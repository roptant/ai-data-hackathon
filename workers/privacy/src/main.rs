//! Private llama.cpp sensitive-span classifier.
//!
//! Usage: `dictation-privacy-worker --model <gguf> [--threads N] [--context N]`
//!
//! Decoding is greedy and grammar-constrained so the same window always yields
//! the same answer; the host still validates every answer independently. The
//! worker has no tools. After loading it confines itself so that transcript
//! text containing instructions cannot reach a network or filesystem.

use std::{env, num::NonZeroU32, process::ExitCode, time::Instant};

use dictation_worker::{
    messages::{
        ClassificationResponse, ClassifyRequest, PROTOCOL_VERSION, Ready, Request, Response,
    },
    sandbox,
    serve::{Handler, serve},
};
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};

const BATCH: usize = 512;

struct Arguments {
    model: String,
    threads: i32,
    context: u32,
}

fn parse_arguments() -> Result<Arguments, &'static str> {
    let mut model = None;
    let mut threads = 4;
    let mut context = 4096;
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or("missing_argument_value")?;
        match flag.as_str() {
            "--model" => model = Some(value),
            "--threads" => threads = value.parse().map_err(|_| "invalid_threads")?,
            "--context" => context = value.parse().map_err(|_| "invalid_context")?,
            _ => return Err("unknown_argument"),
        }
    }
    if !(1..=64).contains(&threads) || !(512..=32_768).contains(&context) {
        return Err("invalid_limits");
    }
    Ok(Arguments {
        model: model.ok_or("missing_model")?,
        threads,
        context,
    })
}

struct PrivacyHandler {
    model: &'static LlamaModel,
    context: LlamaContext<'static>,
    /// Tokens currently held in the KV cache. The shared system prompt is
    /// reused across windows; everything after the common prefix is erased
    /// before the next request is evaluated.
    cached: Vec<LlamaToken>,
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

impl PrivacyHandler {
    #[allow(clippy::too_many_lines)]
    fn classify(&mut self, request: &ClassifyRequest) -> Result<ClassificationResponse, String> {
        let started = Instant::now();
        let template = self
            .model
            .chat_template(None)
            .map_err(|_| "missing_chat_template".to_owned())?;
        let messages = [
            LlamaChatMessage::new("system".to_owned(), request.system_prompt.clone())
                .map_err(|_| "invalid_prompt".to_owned())?,
            LlamaChatMessage::new("user".to_owned(), request.user_prompt.clone())
                .map_err(|_| "invalid_prompt".to_owned())?,
        ];
        let prompt = self
            .model
            .apply_chat_template(&template, &messages, true)
            .map_err(|_| "chat_template_failed".to_owned())?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(|_| "tokenize_failed".to_owned())?;
        let capacity = self.context.n_ctx() as usize;
        // Never truncate the input: text that did not fit was not classified.
        if tokens.len() + request.max_tokens as usize > capacity {
            return Err("context_exceeded".to_owned());
        }
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::grammar(self.model, &request.grammar, "root")
                .map_err(|_| "invalid_grammar".to_owned())?,
            LlamaSampler::greedy(),
        ]);
        let common = self
            .cached
            .iter()
            .zip(&tokens)
            .take_while(|(cached, new)| cached == new)
            .count()
            .min(tokens.len().saturating_sub(1));
        self.cached.clear();
        let reused = u32::try_from(common).map_err(|_| "position_overflow".to_owned())?;
        if !matches!(
            self.context.clear_kv_cache_seq(Some(0), Some(reused), None),
            Ok(true)
        ) {
            self.context.clear_kv_cache();
            return Err("kv_cache_reset_failed".to_owned());
        }
        let mut batch = LlamaBatch::new(BATCH, 1);
        let last_index = tokens.len().saturating_sub(1);
        for chunk_start in (common..tokens.len()).step_by(BATCH) {
            batch.clear();
            let chunk_end = (chunk_start + BATCH).min(tokens.len());
            for (position, token) in tokens.iter().enumerate().take(chunk_end).skip(chunk_start) {
                batch
                    .add(
                        *token,
                        i32::try_from(position).map_err(|_| "position_overflow".to_owned())?,
                        &[0],
                        position == last_index,
                    )
                    .map_err(|_| "batch_failed".to_owned())?;
            }
            self.context
                .decode(&mut batch)
                .map_err(|_| "decode_failed".to_owned())?;
        }
        let mut position = tokens.len();
        let mut output = Vec::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut completion_tokens = 0_u32;
        let mut generated = Vec::new();
        let mut truncated = true;
        while completion_tokens < request.max_tokens {
            let token = sampler.sample(&self.context, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                truncated = false;
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|_| "detokenize_failed".to_owned())?;
            output.push(piece);
            generated.push(token);
            completion_tokens += 1;
            batch.clear();
            batch
                .add(
                    token,
                    i32::try_from(position).map_err(|_| "position_overflow".to_owned())?,
                    &[0],
                    true,
                )
                .map_err(|_| "batch_failed".to_owned())?;
            position += 1;
            self.context
                .decode(&mut batch)
                .map_err(|_| "decode_failed".to_owned())?;
        }
        // Every emitted token was decoded, so the cache holds exactly these.
        self.cached = tokens.iter().chain(&generated).copied().collect();
        Ok(ClassificationResponse {
            request_id: request.request_id,
            output: output.concat(),
            prompt_tokens: u32::try_from(tokens.len()).unwrap_or(u32::MAX),
            completion_tokens,
            compute_ms: elapsed_ms(started),
            truncated,
        })
    }
}

impl Handler for PrivacyHandler {
    fn handle(&mut self, request: &Request, pcm: Option<&[i16]>) -> Result<Response, String> {
        match (request, pcm) {
            (Request::Classify(request), None) => {
                self.classify(request).map(Response::Classification)
            }
            _ => Err("unsupported_request".to_owned()),
        }
    }
}

fn main() -> ExitCode {
    let Ok(arguments) = parse_arguments() else {
        return ExitCode::from(2);
    };
    let started = Instant::now();
    let Ok(mut backend) = LlamaBackend::init() else {
        return ExitCode::from(3);
    };
    backend.void_logs();
    let parameters = LlamaModelParams::default().with_n_gpu_layers(0);
    let Ok(model) = LlamaModel::load_from_file(&backend, &arguments.model, &parameters) else {
        return ExitCode::from(3);
    };
    // The model lives for the whole process; leaking it lets the context
    // borrow it without a self-referential struct.
    let model: &'static LlamaModel = Box::leak(Box::new(model));
    let backend: &'static LlamaBackend = Box::leak(Box::new(backend));
    let context_parameters = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(arguments.context))
        .with_n_batch(u32::try_from(BATCH).unwrap_or(512))
        .with_n_threads(arguments.threads)
        .with_n_threads_batch(arguments.threads);
    let Ok(context) = model.new_context(backend, context_parameters) else {
        return ExitCode::from(3);
    };
    let description = format!(
        "llama.cpp params={} size_bytes={} ctx={}",
        model.n_params(),
        model.size(),
        arguments.context
    );
    let sandbox = sandbox::confine();
    let ready = Ready {
        worker: "privacy".to_owned(),
        protocol: PROTOCOL_VERSION,
        model_description: description,
        load_ms: elapsed_ms(started),
        sandbox,
    };
    let mut handler = PrivacyHandler {
        model,
        context,
        cached: Vec::new(),
    };
    match serve(&ready, &mut handler) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::from(4),
    }
}
