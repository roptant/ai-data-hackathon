//! Benchmark harness for the two model roles (implementation plan §4, §12 M1).
//!
//! It drives the real worker executables through the same private IPC used by
//! the application and reports load time, compute time, real-time factor, and
//! the worker's peak resident memory. Results are printed as JSON so they can
//! be recorded with the exact OS, CPU, memory, model, and quantization.

mod privacy_eval;

use std::{
    env,
    ffi::OsString,
    fs,
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

use dictation_worker::{
    messages::{ClassifyRequest, Request, Response, TranscribeRequest},
    persistent::{PersistentWorker, WorkerSpec},
    serve::{MAXIMUM_AUDIO_BYTES, MAXIMUM_CONTROL_BYTES},
};
use serde_json::{Value, json};

struct Options {
    values: Vec<(String, String)>,
    flags: Vec<String>,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Self {
        let mut values = Vec::new();
        let mut flags = Vec::new();
        let mut arguments = arguments.peekable();
        while let Some(argument) = arguments.next() {
            if let Some(name) = argument.strip_prefix("--") {
                match arguments.peek() {
                    Some(next) if !next.starts_with("--") => {
                        values.push((name.to_owned(), arguments.next().unwrap_or_default()));
                    }
                    _ => flags.push(name.to_owned()),
                }
            }
        }
        Self { values, flags }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    fn require(&self, name: &str) -> Result<&str, String> {
        self.get(name).ok_or_else(|| format!("missing --{name}"))
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|flag| flag == name)
    }
}

/// Peak resident memory plus the current anonymous/file-backed split.
///
/// Memory-mapped weights are file-backed and reclaimable, so the plan's
/// memory budget is reported both with and without them.
fn memory_report(process_id: Option<u32>) -> Value {
    let Some(status) = process_id
        .and_then(|id| fs::read_to_string(format!("/proc/{id}/status")).ok())
    else {
        return Value::Null;
    };
    let field = |name: &str| -> Option<f64> {
        let line = status.lines().find(|line| line.starts_with(name))?;
        let kilobytes: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kilobytes / 1024.0)
    };
    json!({
        "peak_rss_mb": field("VmHWM:"),
        "rss_anon_mb": field("RssAnon:"),
        "rss_file_mb": field("RssFile:"),
    })
}

fn spawn(worker: &str, arguments: Vec<OsString>, startup: Duration) -> PersistentWorker {
    let executable = fs::canonicalize(worker).unwrap_or_else(|_| PathBuf::from(worker));
    let working_directory = executable
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    PersistentWorker::new(WorkerSpec {
        executable,
        arguments,
        working_directory,
        maximum_control_bytes: MAXIMUM_CONTROL_BYTES,
        maximum_audio_bytes: MAXIMUM_AUDIO_BYTES,
        startup_timeout: startup,
        environment: Vec::new(),
    })
}

fn absolute(path: &str) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path))
}

fn read_wav(path: &str) -> Result<Vec<i16>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|error| error.to_string())?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("benchmark audio must be 16 kHz mono 16-bit PCM".to_owned());
    }
    reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn asr(options: &Options) -> Result<Value, String> {
    let mut arguments: Vec<OsString> = vec![
        "--model".into(),
        absolute(options.require("model")?).into(),
        "--threads".into(),
        options.get("threads").unwrap_or("4").into(),
    ];
    if let Some(preset) = options.get("dtw") {
        arguments.extend(["--dtw".into(), preset.into()]);
    }
    let samples = read_wav(options.require("wav")?)?;
    let pcm: Vec<u8> = samples.iter().flat_map(|sample| sample.to_le_bytes()).collect();
    let repeat: usize = options
        .get("repeat")
        .unwrap_or("3")
        .parse()
        .map_err(|_| "invalid --repeat")?;
    let spawn_started = Instant::now();
    let mut worker = spawn(
        options.require("worker")?,
        arguments,
        Duration::from_secs(120),
    );
    let ready = worker
        .ensure_started()
        .map_err(|error| error.to_string())?
        .clone();
    let cold_start_ms = millis(spawn_started.elapsed());
    let audio_ms = samples.len() as u64 * 1_000 / 16_000;
    let mut runs = Vec::new();
    let mut last = Value::Null;
    for index in 0..repeat {
        let request_id = worker.next_request_id();
        let request = Request::Transcribe(TranscribeRequest {
            request_id,
            stream_epoch: None,
            language: options.get("language").map(str::to_owned),
            initial_prompt: None,
            final_pass: options.flag("final"),
        });
        let started = Instant::now();
        let response = worker
            .request(&request, Some(&pcm), Duration::from_secs(300))
            .map_err(|error| error.to_string())?;
        let wall_ms = millis(started.elapsed());
        let Response::Transcript(transcript) = response else {
            return Err(format!("unexpected worker response: {response:?}"));
        };
        #[allow(clippy::cast_precision_loss)]
        let rtf = wall_ms as f64 / audio_ms.max(1) as f64;
        runs.push(json!({ "run": index, "wall_ms": wall_ms, "compute_ms": transcript.compute_ms, "real_time_factor": rtf }));
        last = json!({
            "language": transcript.language,
            "text": transcript.segments.iter().map(|segment| segment.text.as_str()).collect::<Vec<_>>().join(" "),
            "words": transcript.segments.iter().flat_map(|segment| segment.words.iter()).map(|word| json!([word.text, word.start_sample, word.end_sample, word.probability, word.timing_unreliable])).collect::<Vec<_>>(),
        });
    }
    let peak = memory_report(worker.process_id());
    Ok(json!({
        "role": "asr",
        "ready": { "model": ready.model_description, "load_ms": ready.load_ms, "sandbox": {
            "network_denied": ready.sandbox.network_denied,
            "filesystem_restricted": ready.sandbox.filesystem_restricted,
            "notes": ready.sandbox.notes } },
        "cold_start_ms": cold_start_ms,
        "audio_ms": audio_ms,
        "runs": runs,
        "worker_memory": peak,
        "output": last,
    }))
}

fn privacy(options: &Options) -> Result<Value, String> {
    let arguments: Vec<OsString> = vec![
        "--model".into(),
        absolute(options.require("model")?).into(),
        "--threads".into(),
        options.get("threads").unwrap_or("4").into(),
        "--context".into(),
        options.get("context").unwrap_or("4096").into(),
    ];
    let cases: Value = serde_json::from_str(
        &fs::read_to_string(options.require("cases")?).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let spawn_started = Instant::now();
    let mut worker = spawn(
        options.require("worker")?,
        arguments,
        Duration::from_secs(300),
    );
    let ready = worker
        .ensure_started()
        .map_err(|error| error.to_string())?
        .clone();
    let cold_start_ms = millis(spawn_started.elapsed());
    let mut results = Vec::new();
    for case in cases.as_array().ok_or("cases must be an array")? {
        let request = Request::Classify(ClassifyRequest {
            request_id: worker.next_request_id(),
            system_prompt: case["system_prompt"].as_str().unwrap_or_default().to_owned(),
            user_prompt: case["user_prompt"].as_str().unwrap_or_default().to_owned(),
            grammar: case["grammar"].as_str().unwrap_or_default().to_owned(),
            max_tokens: 512,
        });
        let started = Instant::now();
        let response = worker
            .request(&request, None, Duration::from_secs(600))
            .map_err(|error| error.to_string())?;
        results.push(match response {
            Response::Classification(output) => json!({
                "wall_ms": millis(started.elapsed()),
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "truncated": output.truncated,
                "output": output.output,
            }),
            other => json!({ "error": format!("{other:?}") }),
        });
    }
    Ok(json!({
        "role": "privacy",
        "ready": { "model": ready.model_description, "load_ms": ready.load_ms, "sandbox": {
            "network_denied": ready.sandbox.network_denied,
            "filesystem_restricted": ready.sandbox.filesystem_restricted,
            "notes": ready.sandbox.notes } },
        "cold_start_ms": cold_start_ms,
        "results": results,
        "worker_memory": memory_report(worker.process_id()),
    }))
}

fn privacy_evaluation(options: &Options) -> Result<Value, String> {
    let corpus = fs::read_to_string(options.require("cases")?).map_err(|error| error.to_string())?;
    let cases = privacy_eval::load(&corpus)?;
    let progress = |index: usize, id: &str| eprintln!("[{}/{}] {id}", index + 1, cases.len());
    if options.flag("rules-only") {
        return privacy_eval::evaluate(&cases, &mut privacy_eval::NoModel, progress);
    }
    let models = PathBuf::from(options.require("models")?);
    let worker_dir = absolute(options.require("worker-dir")?);
    let paths = dictation_engine::workers::WorkerPaths::locate(&worker_dir);
    let verifier = dictation_engine::workers::ModelVerifier::default();
    let spec = dictation_models::default_for(dictation_models::Role::Privacy);
    let mut engine = dictation_engine::workers::PrivacyEngine::new(&paths, &models, spec, &verifier, dictation_engine::workers::Preemption::default())
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut report = privacy_eval::evaluate(&cases, &mut engine, progress)?;
    report["model"] = json!(spec.id);
    report["wall_seconds"] = json!(started.elapsed().as_secs());
    Ok(report)
}

fn main() -> ExitCode {
    let mut arguments = env::args().skip(1);
    let command = arguments.next().unwrap_or_default();
    let options = Options::parse(arguments);
    let result = match command.as_str() {
        "asr" => asr(&options),
        "privacy" => privacy(&options),
        "privacy-eval" => privacy_evaluation(&options),
        _ => Err("usage: dictation-bench <asr|privacy> --worker PATH --model PATH ...".to_owned()),
    };
    match result {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).unwrap_or_default()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
