//! Orchestration between the pure core, encrypted storage, and model workers.
//!
//! Live dictation uses the ASR worker only. Training copies are processed
//! afterwards by [`training`], and eligible packages are transferred by
//! [`upload`], which holds only the queue database.

pub mod training;
pub mod upload;
pub mod workers;
