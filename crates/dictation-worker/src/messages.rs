//! Typed control messages exchanged with the private model workers.
//!
//! Every request carries a `request_id`; responses that do not echo the
//! outstanding identifier are discarded as stale. Transcript text travels only
//! inside these frames, never through arguments, environment variables, files,
//! or worker diagnostics.

use serde::{Deserialize, Serialize};

/// First frame a worker sends after loading its model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ready {
    pub worker: String,
    pub protocol: u32,
    pub model_description: String,
    pub load_ms: u64,
    pub sandbox: SandboxReport,
}

/// What the worker could confine after start-up. Reported, never assumed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxReport {
    pub network_denied: bool,
    pub filesystem_restricted: bool,
    pub notes: Vec<String>,
}

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Followed by exactly one PCM frame of 16 kHz mono signed 16-bit samples.
    Transcribe(TranscribeRequest),
    Classify(ClassifyRequest),
    Shutdown { request_id: u64 },
}

impl Request {
    #[must_use]
    pub const fn request_id(&self) -> u64 {
        match self {
            Self::Transcribe(request) => request.request_id,
            Self::Classify(request) => request.request_id,
            Self::Shutdown { request_id } => *request_id,
        }
    }

    #[must_use]
    pub const fn carries_audio(&self) -> bool {
        matches!(self, Self::Transcribe(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscribeRequest {
    pub request_id: u64,
    /// ISO 639-1 code, or `None` for detection.
    pub language: Option<String>,
    /// Local vocabulary hint. It stays inside the ASR worker.
    pub initial_prompt: Option<String>,
    /// Partial passes may use cheaper decoding; the final pass never does.
    pub final_pass: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifyRequest {
    pub request_id: u64,
    pub system_prompt: String,
    pub user_prompt: String,
    /// GBNF grammar constraining the answer.
    pub grammar: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Transcript(TranscriptResponse),
    Classification(ClassificationResponse),
    Error(ErrorResponse),
    ShuttingDown { request_id: u64 },
}

impl Response {
    #[must_use]
    pub const fn request_id(&self) -> u64 {
        match self {
            Self::Transcript(response) => response.request_id,
            Self::Classification(response) => response.request_id,
            Self::Error(response) => response.request_id,
            Self::ShuttingDown { request_id } => *request_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptResponse {
    pub request_id: u64,
    pub language: String,
    pub audio_samples: u64,
    pub compute_ms: u64,
    pub segments: Vec<AsrSegment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrSegment {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub no_speech_probability: f32,
    pub words: Vec<AsrWord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrWord {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub probability: f32,
    /// Set when the worker could not align this word reliably.
    pub timing_unreliable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassificationResponse {
    pub request_id: u64,
    pub output: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub compute_ms: u64,
    /// True when generation stopped at `max_tokens` instead of end-of-text.
    pub truncated: bool,
}

/// A coarse, content-free failure code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub request_id: u64,
    pub code: String,
}
