//! Privacy-critical, platform-independent domain logic.
//!
//! This crate deliberately has no Tauri, operating-system, network, model, or
//! filesystem dependencies. It is the first migration boundary from the Python
//! reference implementation and can be tested without a desktop environment.

pub mod asr;
pub mod audio;
pub mod classifier_schema;
pub mod contribution;
pub mod coordinator;
pub mod dataset;
pub mod package;
pub mod platform;
pub mod privacy;
pub mod quality;
pub mod recording;
pub mod removal;
pub mod rules;
pub mod shortcuts;
pub mod time_map;
pub mod transcript;
pub mod vad;
pub mod wer;

pub const CANONICAL_SAMPLE_RATE: u32 = 16_000;
pub const CANONICAL_CHANNELS: u16 = 1;
pub const CANONICAL_SAMPLE_WIDTH: usize = 2;
