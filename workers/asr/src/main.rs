//! Private whisper.cpp worker.
//!
//! Usage: `dictation-asr-worker --model <ggml file> [--threads N] [--dtw PRESET]`
//!
//! The host verifies the model checksum before spawning. After loading, the
//! worker confines itself (see `dictation_worker::sandbox`) and then only reads
//! framed requests from standard input. It never writes diagnostics that could
//! contain transcript text; standard error is discarded by the host.

mod words;

use std::{env, process::ExitCode, time::Instant};

use dictation_worker::{
    messages::{
        AsrSegment, PROTOCOL_VERSION, Ready, Request, Response, TranscribeRequest,
        TranscriptResponse,
    },
    sandbox,
    serve::{Handler, serve},
};
use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperState,
};

use crate::words::{SAMPLES_PER_CENTISECOND, Token, group_words};

struct Arguments {
    model: String,
    threads: i32,
    dtw: Option<DtwModelPreset>,
}

fn parse_preset(name: &str) -> Option<DtwModelPreset> {
    Some(match name {
        "tiny.en" => DtwModelPreset::TinyEn,
        "tiny" => DtwModelPreset::Tiny,
        "base.en" => DtwModelPreset::BaseEn,
        "base" => DtwModelPreset::Base,
        "small.en" => DtwModelPreset::SmallEn,
        "small" => DtwModelPreset::Small,
        "medium.en" => DtwModelPreset::MediumEn,
        "medium" => DtwModelPreset::Medium,
        "large-v3-turbo" => DtwModelPreset::LargeV3Turbo,
        _ => return None,
    })
}

fn parse_arguments() -> Result<Arguments, &'static str> {
    let mut model = None;
    let mut threads = 4;
    let mut dtw = None;
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or("missing_argument_value")?;
        match flag.as_str() {
            "--model" => model = Some(value),
            "--threads" => {
                threads = value.parse().map_err(|_| "invalid_threads")?;
                if !(1..=64).contains(&threads) {
                    return Err("invalid_threads");
                }
            }
            "--dtw" => dtw = Some(parse_preset(&value).ok_or("unknown_dtw_preset")?),
            _ => return Err("unknown_argument"),
        }
    }
    Ok(Arguments {
        model: model.ok_or("missing_model")?,
        threads,
        dtw,
    })
}

struct AsrHandler {
    context: WhisperContext,
    state: WhisperState,
    threads: i32,
}

impl AsrHandler {
    fn transcribe(
        &mut self,
        request: &TranscribeRequest,
        pcm: &[i16],
    ) -> Result<TranscriptResponse, String> {
        let started = Instant::now();
        let audio_samples = pcm.len() as u64;
        let samples: Vec<f32> = pcm
            .iter()
            .map(|sample| f32::from(*sample) / 32_768.0)
            .collect();
        let strategy = if request.final_pass {
            SamplingStrategy::BeamSearch {
                beam_size: 3,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(0.0);
        let language = request.language.as_deref().unwrap_or("auto");
        params.set_language(Some(language));
        if let Some(prompt) = request.initial_prompt.as_deref() {
            params.set_initial_prompt(prompt);
        }
        // Whisper pads to 30 s windows internally; very short input is padded
        // with silence here so the model does not reject it.
        let mut padded = samples;
        if padded.len() < 16_000 {
            padded.resize(16_000, 0.0);
        }
        self.state
            .full(params, &padded)
            .map_err(|_| "inference_failed".to_owned())?;

        let detected = whisper_rs::get_lang_str(self.state.full_lang_id_from_state())
            .unwrap_or("unknown")
            .to_owned();
        let end_of_text = self.context.token_eot();
        let mut segments = Vec::new();
        for segment in self.state.as_iter() {
            let start_sample = to_samples(segment.start_timestamp()).min(audio_samples);
            let end_sample = to_samples(segment.end_timestamp()).min(audio_samples);
            let mut tokens = Vec::new();
            for index in 0..segment.n_tokens() {
                let Some(token) = segment.get_token(index) else {
                    continue;
                };
                if token.token_id() >= end_of_text {
                    continue;
                }
                let data = token.token_data();
                let text = token
                    .to_str_lossy()
                    .map_err(|_| "token_decode_failed".to_owned())?
                    .into_owned();
                tokens.push(Token {
                    text,
                    t0: data.t0,
                    t1: data.t1,
                    t_dtw: data.t_dtw,
                    probability: data.p,
                });
            }
            let words = group_words(&tokens, start_sample, end_sample, audio_samples);
            let text = segment
                .to_str_lossy()
                .map_err(|_| "segment_decode_failed".to_owned())?
                .trim()
                .to_owned();
            if text.is_empty() && words.is_empty() {
                continue;
            }
            segments.push(AsrSegment {
                text,
                start_sample,
                end_sample: end_sample.max(start_sample),
                no_speech_probability: segment.no_speech_probability(),
                words,
            });
        }
        Ok(TranscriptResponse {
            request_id: request.request_id,
            language: detected,
            audio_samples,
            compute_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            segments,
        })
    }
}

fn to_samples(centiseconds: i64) -> u64 {
    u64::try_from(centiseconds.max(0)).unwrap_or(0) * SAMPLES_PER_CENTISECOND
}

impl Handler for AsrHandler {
    fn handle(&mut self, request: &Request, pcm: Option<&[i16]>) -> Result<Response, String> {
        match (request, pcm) {
            (Request::Transcribe(request), Some(pcm)) => {
                self.transcribe(request, pcm).map(Response::Transcript)
            }
            _ => Err("unsupported_request".to_owned()),
        }
    }
}

fn main() -> ExitCode {
    whisper_rs::install_logging_hooks();
    let Ok(arguments) = parse_arguments() else {
        return ExitCode::from(2);
    };
    let started = Instant::now();
    let mut parameters = WhisperContextParameters::default();
    parameters.use_gpu(false);
    if let Some(preset) = arguments.dtw.clone() {
        parameters.dtw_parameters = DtwParameters {
            mode: DtwMode::ModelPreset {
                model_preset: preset,
            },
            ..DtwParameters::default()
        };
    }
    let Ok(context) = WhisperContext::new_with_params(&arguments.model, parameters) else {
        return ExitCode::from(3);
    };
    let Ok(state) = context.create_state() else {
        return ExitCode::from(3);
    };
    let description = format!(
        "whisper.cpp {} multilingual={} dtw={}",
        context
            .model_type_readable_str_lossy()
            .map(|value| value.into_owned())
            .unwrap_or_default(),
        context.is_multilingual(),
        arguments.dtw.is_some()
    );
    let sandbox = sandbox::confine();
    let ready = Ready {
        worker: "asr".to_owned(),
        protocol: PROTOCOL_VERSION,
        model_description: description,
        load_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        sandbox,
    };
    let mut handler = AsrHandler {
        context,
        state,
        threads: arguments.threads,
    };
    match serve(&ready, &mut handler) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::from(4),
    }
}
