//! Wire types between the desktop upload worker and the training service.
//!
//! Tenant identity is never part of a request body: the server derives it from
//! the bearer credential. Bodies carry identifiers and consent facts only.

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

/// Header carrying the client's SHA-256 of the uploaded archive bytes.
pub const CONTENT_SHA256_HEADER: &str = "x-content-sha256";
/// Header carrying the per-job idempotency key.
pub const IDEMPOTENCY_HEADER: &str = "idempotency-key";

pub const MAX_ARCHIVE_BYTES: usize = 33 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub receipt_id: String,
    pub accepted: bool,
    /// Content-free reason code when not accepted.
    pub reason: String,
    /// True when an earlier request with the same idempotency key answered.
    pub duplicate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsentGrant {
    pub consent_id: String,
    pub version: String,
    pub purposes: Vec<String>,
    pub granted_at: i64,
    pub expires_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsentAck {
    pub consent_id: String,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WithdrawalAck {
    pub consent_id: String,
    pub cancelled_training_jobs: u32,
    pub deleted_samples: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum DeletionScope {
    /// All training data and personalized models for the authenticated tenant.
    Everything,
    /// One sample, e.g. after a receipt arrived for a locally withdrawn job.
    Sample { sample_id: String },
    /// A sample identified only by the receipt the client holds.
    Receipt { receipt_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletionRequest {
    pub request_id: String,
    #[serde(flatten)]
    pub scope: DeletionScope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletionReport {
    pub request_id: String,
    pub completed: bool,
    pub deleted_samples: u32,
    pub deleted_artifacts: u32,
    /// Honest limits of the deletion, e.g. backups expiring on schedule.
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: String,
}
