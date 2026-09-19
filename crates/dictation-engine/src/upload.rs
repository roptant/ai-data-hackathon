//! Automatic upload of eligible packages (plan §9).
//!
//! The worker receives only the queue database: it cannot read raw sessions.
//! Consent is rechecked immediately before each transfer. Retries use the same
//! idempotency key with bounded exponential backoff; a failure never falls
//! back to anything other than the package already built. A receipt that
//! arrives after local withdrawal is recorded as not accepted and triggers a
//! server-side deletion request.

use std::time::Duration;

use dictation_core::contribution::{
    JobState, RetrySettings, UploadTarget, VolumeCaps, recheck, retry_delay,
};
use dictation_protocol::{
    CONTENT_SHA256_HEADER, ConsentAck, ConsentGrant, DeletionReport, DeletionRequest,
    DeletionScope, ErrorBody, IDEMPOTENCY_HEADER, Receipt, WithdrawalAck,
};
use dictation_storage::{
    EncryptedStore, StoreError,
    contribution::{DeletionRequest as StoredDeletion, Job, JobUpdate},
    package_scope_id,
};
use sha2::{Digest, Sha256};

use crate::training::{package_archive, random_hex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// Network failure, timeout, 5xx, or 429: retry later.
    Retriable(String),
    /// The server refused the request itself.
    Permanent(String),
    Unauthorized,
}

pub trait Transport {
    /// # Errors
    ///
    /// Returns a classified transport failure.
    fn upload(&mut self, archive: &[u8], sha256: &str, idempotency_key: &str) -> Result<Receipt, TransportError>;
    /// # Errors
    ///
    /// Returns a classified transport failure.
    fn register_consent(&mut self, grant: &ConsentGrant) -> Result<ConsentAck, TransportError>;
    /// # Errors
    ///
    /// Returns a classified transport failure.
    fn withdraw_consent(&mut self, consent_id: &str) -> Result<WithdrawalAck, TransportError>;
    /// # Errors
    ///
    /// Returns a classified transport failure.
    fn request_deletion(&mut self, request: &DeletionRequest) -> Result<DeletionReport, TransportError>;
}

/// Blocking HTTPS transport. Plain HTTP is accepted only for a loopback
/// nonproduction server used in integration tests.
pub struct HttpTransport {
    base_url: String,
    token: String,
    client: reqwest::blocking::Client,
}

impl HttpTransport {
    /// # Errors
    ///
    /// Refuses non-HTTPS endpoints unless they are loopback nonproduction.
    pub fn new(base_url: &str, token: &str, target: UploadTarget) -> Result<Self, TransportError> {
        let loopback = base_url.starts_with("http://127.0.0.1:") || base_url.starts_with("http://[::1]:");
        let allowed = base_url.starts_with("https://")
            || (loopback && target == UploadTarget::NonProduction);
        if !allowed || target == UploadTarget::Disabled {
            return Err(TransportError::Permanent("insecure_or_disabled_endpoint".to_owned()));
        }
        let client = reqwest::blocking::Client::builder()
            .https_only(!loopback)
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(concat!("local-dictation/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| TransportError::Permanent(error.to_string()))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            token: token.to_owned(),
            client,
        })
    }

    fn send<T: serde::de::DeserializeOwned>(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> Result<T, TransportError> {
        let response = request
            .bearer_auth(&self.token)
            .send()
            .map_err(|error| TransportError::Retriable(if error.is_timeout() { "timeout" } else { "network" }.to_owned()))?;
        let status = response.status();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(TransportError::Unauthorized);
        }
        if status.is_server_error() || status.as_u16() == 429 {
            return Err(TransportError::Retriable(format!("http_{}", status.as_u16())));
        }
        if !status.is_success() {
            let code = response
                .json::<ErrorBody>()
                .map_or_else(|_| format!("http_{}", status.as_u16()), |body| body.error);
            return Err(TransportError::Permanent(code));
        }
        response
            .json::<T>()
            .map_err(|_| TransportError::Retriable("malformed_response".to_owned()))
    }
}

impl HttpTransport {
    /// Latest signed personalized-model manifest for this tenant, if any.
    ///
    /// # Errors
    ///
    /// Returns a classified transport failure.
    pub fn latest_model(&self) -> Result<Option<(String, dictation_models::delivery::SignedManifest)>, TransportError> {
        #[derive(serde::Deserialize)]
        struct Latest {
            model_version: Option<String>,
            signed: Option<dictation_models::delivery::SignedManifest>,
        }
        let latest: Latest = self.send(self.client.get(format!("{}/v1/models/latest", self.base_url)))?;
        Ok(latest.model_version.zip(latest.signed))
    }

    /// Downloads a delivered model artifact; verified by the caller.
    ///
    /// # Errors
    ///
    /// Returns a classified transport failure.
    pub fn model_artifact(&self, version: &str) -> Result<Vec<u8>, TransportError> {
        if !version.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-') {
            return Err(TransportError::Permanent("invalid_version".to_owned()));
        }
        let response = self
            .client
            .get(format!("{}/v1/models/{version}/artifact", self.base_url))
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(1_800))
            .send()
            .map_err(|_| TransportError::Retriable("network".to_owned()))?;
        if !response.status().is_success() {
            return Err(TransportError::Permanent(format!("http_{}", response.status().as_u16())));
        }
        response
            .bytes()
            .map(|bytes| bytes.to_vec())
            .map_err(|_| TransportError::Retriable("network".to_owned()))
    }
}

impl Transport for HttpTransport {
    fn upload(&mut self, archive: &[u8], sha256: &str, idempotency_key: &str) -> Result<Receipt, TransportError> {
        self.send(
            self.client
                .post(format!("{}/v1/samples", self.base_url))
                .header(IDEMPOTENCY_HEADER, idempotency_key)
                .header(CONTENT_SHA256_HEADER, sha256)
                .header("content-type", "application/x-tar")
                .body(archive.to_vec()),
        )
    }

    fn register_consent(&mut self, grant: &ConsentGrant) -> Result<ConsentAck, TransportError> {
        self.send(self.client.post(format!("{}/v1/consents", self.base_url)).json(grant))
    }

    fn withdraw_consent(&mut self, consent_id: &str) -> Result<WithdrawalAck, TransportError> {
        self.send(
            self.client
                .post(format!("{}/v1/consents/{consent_id}/withdraw", self.base_url)),
        )
    }

    fn request_deletion(&mut self, request: &DeletionRequest) -> Result<DeletionReport, TransportError> {
        self.send(
            self.client
                .post(format!("{}/v1/deletion-requests", self.base_url))
                .json(request),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadOutcome {
    Acknowledged { job_id: String, receipt_id: String },
    RejectedByServer { job_id: String, reason: String },
    Retrying { job_id: String, reason: String, next_attempt_at: i64 },
    GaveUp { job_id: String, reason: String },
    Cancelled { job_id: String, reason: String },
    Deferred { job_id: String, reason: String },
    LateReceipt { job_id: String, receipt_id: String },
}

pub struct UploadPolicy {
    pub target: UploadTarget,
    pub caps: VolumeCaps,
    pub retry: RetrySettings,
}

fn cancel(queue: &EncryptedStore, job: &Job, reason: &str, now: i64) -> Result<UploadOutcome, StoreError> {
    let _ = queue.transition_job(
        &job.job_id,
        JobState::Deleted,
        &JobUpdate {
            reason: Some(reason),
            ..JobUpdate::default()
        },
        now,
    );
    queue.delete_scope(&package_scope_id(&job.job_id))?;
    Ok(UploadOutcome::Cancelled {
        job_id: job.job_id.clone(),
        reason: reason.to_owned(),
    })
}

fn finish_rejected(queue: &EncryptedStore, job: &Job, reason: &str, now: i64) -> Result<(), StoreError> {
    let _ = queue.transition_job(
        &job.job_id,
        JobState::Rejected,
        &JobUpdate {
            reason: Some(reason),
            ..JobUpdate::default()
        },
        now,
    );
    queue.delete_scope(&package_scope_id(&job.job_id))?;
    Ok(())
}

/// A receipt for a job that is no longer uploading: never re-eligible.
fn late_receipt(queue: &EncryptedStore, job: &Job, receipt: &Receipt, now: i64) -> Result<UploadOutcome, StoreError> {
    queue.insert_receipt(&job.job_id, &receipt.receipt_id, false, now)?;
    let request_id = format!("deletion-{}", random_hex(12));
    queue.insert_deletion_request(
        StoredDeletion {
            request_id: &request_id,
            scope: &format!("receipt:{}", receipt.receipt_id),
        },
        now,
    )?;
    Ok(UploadOutcome::LateReceipt {
        job_id: job.job_id.clone(),
        receipt_id: receipt.receipt_id.clone(),
    })
}

/// Uploads one eligible job.
///
/// # Errors
///
/// Fails on storage errors only; transport failures become outcomes.
pub fn upload_job(
    queue: &EncryptedStore,
    transport: &mut impl Transport,
    policy: &UploadPolicy,
    job: &Job,
    now: impl Fn() -> i64,
) -> Result<UploadOutcome, StoreError> {
    let stamp = now();
    let consent = queue.consent(&job.consent_id)?;
    if let Err(refusal) = recheck(consent.as_ref(), policy.target, stamp) {
        return cancel(queue, job, refusal.code(), stamp);
    }
    let (packages, seconds) = queue.volume_today(stamp)?;
    if packages >= policy.caps.max_packages_per_day || seconds >= policy.caps.max_seconds_per_day {
        let next = (stamp.div_euclid(86_400) + 1) * 86_400;
        queue.reschedule_job(&job.job_id, next, "daily_cap_reached", stamp)?;
        return Ok(UploadOutcome::Deferred {
            job_id: job.job_id.clone(),
            reason: "daily_cap_reached".to_owned(),
        });
    }
    let Some(archive) = package_archive(queue, &job.job_id)? else {
        finish_rejected(queue, job, "package_missing_or_expired", stamp)?;
        return Ok(UploadOutcome::GaveUp {
            job_id: job.job_id.clone(),
            reason: "package_missing_or_expired".to_owned(),
        });
    };
    let job = queue.transition_job(
        &job.job_id,
        JobState::Uploading,
        &JobUpdate {
            attempts: Some(job.attempts.saturating_add(1)),
            ..JobUpdate::default()
        },
        stamp,
    )?;
    let digest = hex::encode(Sha256::digest(&archive));
    let result = transport.upload(&archive, &digest, &job.idempotency_key);
    let stamp = now();
    match result {
        Ok(receipt) if receipt.accepted => {
            match queue.transition_job(&job.job_id, JobState::Acknowledged, &JobUpdate::default(), stamp) {
                Ok(_) => {
                    queue.insert_receipt(&job.job_id, &receipt.receipt_id, true, stamp)?;
                    queue.record_contributed_hash(&job.content_hash, stamp)?;
                    queue.add_volume(stamp, 1, job.duration_ms.div_ceil(1_000))?;
                    queue.delete_scope(&package_scope_id(&job.job_id))?;
                    Ok(UploadOutcome::Acknowledged {
                        job_id: job.job_id,
                        receipt_id: receipt.receipt_id,
                    })
                }
                // Withdrawal cancelled the job while it was in flight.
                Err(StoreError::TransitionConflict) => late_receipt(queue, &job, &receipt, stamp),
                Err(error) => Err(error),
            }
        }
        Ok(receipt) => {
            finish_rejected(queue, &job, &receipt.reason, stamp)?;
            Ok(UploadOutcome::RejectedByServer {
                job_id: job.job_id,
                reason: receipt.reason,
            })
        }
        Err(TransportError::Retriable(reason)) => {
            if let Some(delay) = retry_delay(job.attempts, true, policy.retry) {
                let next = stamp + delay;
                let moved = queue.transition_job(
                    &job.job_id,
                    JobState::Eligible,
                    &JobUpdate {
                        reason: Some(&reason),
                        next_attempt_at: Some(next),
                        ..JobUpdate::default()
                    },
                    stamp,
                );
                match moved {
                    Ok(_) => Ok(UploadOutcome::Retrying {
                        job_id: job.job_id,
                        reason,
                        next_attempt_at: next,
                    }),
                    Err(StoreError::TransitionConflict) => Ok(UploadOutcome::Cancelled {
                        job_id: job.job_id,
                        reason: "withdrawn_during_transfer".to_owned(),
                    }),
                    Err(error) => Err(error),
                }
            } else {
                finish_rejected(queue, &job, &reason, stamp)?;
                Ok(UploadOutcome::GaveUp { job_id: job.job_id, reason })
            }
        }
        Err(TransportError::Permanent(reason)) => {
            finish_rejected(queue, &job, &reason, stamp)?;
            Ok(UploadOutcome::GaveUp { job_id: job.job_id, reason })
        }
        Err(TransportError::Unauthorized) => {
            // Credentials were revoked: stop, keep nothing upload-ready.
            finish_rejected(queue, &job, "unauthorized", stamp)?;
            Ok(UploadOutcome::GaveUp {
                job_id: job.job_id,
                reason: "unauthorized".to_owned(),
            })
        }
    }
}

/// Uploads every eligible job whose retry time has come.
///
/// # Errors
///
/// Fails on storage errors.
pub fn upload_ready(
    queue: &EncryptedStore,
    transport: &mut impl Transport,
    policy: &UploadPolicy,
    now: impl Fn() -> i64,
) -> Result<Vec<UploadOutcome>, StoreError> {
    let stamp = now();
    let ready: Vec<Job> = queue
        .jobs_in(&[JobState::Eligible])?
        .into_iter()
        .filter(|job| job.next_attempt_at <= stamp)
        .collect();
    let mut outcomes = Vec::with_capacity(ready.len());
    for job in ready {
        outcomes.push(upload_job(queue, transport, policy, &job, &now)?);
    }
    Ok(outcomes)
}

/// Sends pending deletion requests; unsent ones are retried next cycle.
///
/// # Errors
///
/// Fails on storage errors.
pub fn send_deletion_requests(
    queue: &EncryptedStore,
    transport: &mut impl Transport,
    now: i64,
) -> Result<usize, StoreError> {
    let mut completed = 0;
    for (request_id, scope) in queue.pending_deletion_requests()? {
        let scope = if scope == "everything" {
            DeletionScope::Everything
        } else if let Some(receipt_id) = scope.strip_prefix("receipt:") {
            DeletionScope::Receipt {
                receipt_id: receipt_id.to_owned(),
            }
        } else if let Some(sample_id) = scope.strip_prefix("sample:") {
            DeletionScope::Sample {
                sample_id: sample_id.to_owned(),
            }
        } else {
            continue;
        };
        if let Ok(report) = transport.request_deletion(&DeletionRequest {
            request_id: request_id.clone(),
            scope,
        }) {
            if report.completed {
                queue.complete_deletion_request(&request_id, now)?;
                completed += 1;
            }
        }
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use dictation_core::contribution::{CONSENT_VERSION, ConsentRecord, Purpose};
    use dictation_storage::{DataClass, MasterKeyProvider};

    use super::*;

    struct Keys;
    impl MasterKeyProvider for Keys {
        fn load_or_create(&self) -> Result<Option<[u8; 32]>, String> {
            Ok(Some([5; 32]))
        }
    }

    #[derive(Default)]
    struct Fake {
        responses: Vec<Result<Receipt, TransportError>>,
        keys_seen: Vec<String>,
        deletions: Vec<DeletionRequest>,
        on_upload: Option<Box<dyn FnMut()>>,
    }

    impl Transport for Fake {
        fn upload(&mut self, _: &[u8], sha256: &str, key: &str) -> Result<Receipt, TransportError> {
            assert_eq!(sha256.len(), 64);
            self.keys_seen.push(key.to_owned());
            if let Some(hook) = self.on_upload.as_mut() {
                hook();
            }
            self.responses.remove(0)
        }
        fn register_consent(&mut self, grant: &ConsentGrant) -> Result<ConsentAck, TransportError> {
            Ok(ConsentAck { consent_id: grant.consent_id.clone(), active: true })
        }
        fn withdraw_consent(&mut self, id: &str) -> Result<WithdrawalAck, TransportError> {
            Ok(WithdrawalAck { consent_id: id.to_owned(), cancelled_training_jobs: 0, deleted_samples: 0 })
        }
        fn request_deletion(&mut self, request: &DeletionRequest) -> Result<DeletionReport, TransportError> {
            self.deletions.push(request.clone());
            Ok(DeletionReport {
                request_id: request.request_id.clone(),
                completed: true,
                deleted_samples: 1,
                deleted_artifacts: 0,
                notes: Vec::new(),
            })
        }
    }

    fn receipt(accepted: bool) -> Receipt {
        Receipt {
            receipt_id: "receipt-1".to_owned(),
            accepted,
            reason: if accepted { String::new() } else { "invalid_package".to_owned() },
            duplicate: false,
        }
    }

    fn eligible_queue() -> EncryptedStore {
        let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
        queue
            .insert_consent(&ConsentRecord {
                consent_id: "consent-1".to_owned(),
                version: CONSENT_VERSION.to_owned(),
                purposes: vec![Purpose::CustomerPersonalization],
                granted_at: 0,
                expires_at: 1_000_000,
                revoked_at: None,
                paused: false,
                policy_version: "p".to_owned(),
            })
            .unwrap();
        queue.insert_job("job-1", "s1", "consent-1", 1, 1, 1_000_000).unwrap();
        for state in [JobState::Analyzing, JobState::Building] {
            queue.transition_job("job-1", state, &JobUpdate::default(), 2).unwrap();
        }
        let scope = package_scope_id("job-1");
        queue.create_scope(&scope, DataClass::EligiblePackage, 1_000_000, 2).unwrap();
        queue.put_artifact(&scope, &format!("{scope}:archive"), "package_archive", b"archive", 2).unwrap();
        queue
            .transition_job(
                "job-1",
                JobState::Eligible,
                &JobUpdate {
                    idempotency_key: Some("idem-1"),
                    content_hash: Some("hash-1"),
                    duration_ms: Some(4_000),
                    ..JobUpdate::default()
                },
                3,
            )
            .unwrap();
        queue
    }

    fn policy() -> UploadPolicy {
        UploadPolicy {
            target: UploadTarget::NonProduction,
            caps: VolumeCaps::default(),
            retry: RetrySettings::default(),
        }
    }

    #[test]
    fn acknowledged_upload_deletes_the_local_package() {
        let queue = eligible_queue();
        let mut transport = Fake { responses: vec![Ok(receipt(true))], ..Fake::default() };
        let outcomes = upload_ready(&queue, &mut transport, &policy(), || 10).unwrap();
        assert!(matches!(outcomes[0], UploadOutcome::Acknowledged { .. }));
        assert_eq!(queue.job("job-1").unwrap().unwrap().state, JobState::Acknowledged);
        assert!(package_archive(&queue, "job-1").unwrap().is_none());
        assert!(queue.contributed_hashes().unwrap().contains("hash-1"));
        assert_eq!(queue.volume_today(10).unwrap(), (1, 4));
    }

    #[test]
    fn retries_reuse_the_idempotency_key_with_backoff() {
        let queue = eligible_queue();
        let mut transport = Fake {
            responses: vec![Err(TransportError::Retriable("http_503".to_owned())), Ok(receipt(true))],
            ..Fake::default()
        };
        let first = upload_ready(&queue, &mut transport, &policy(), || 10).unwrap();
        assert!(matches!(first[0], UploadOutcome::Retrying { next_attempt_at: 40, .. }));
        assert!(upload_ready(&queue, &mut transport, &policy(), || 20).unwrap().is_empty());
        let second = upload_ready(&queue, &mut transport, &policy(), || 41).unwrap();
        assert!(matches!(second[0], UploadOutcome::Acknowledged { .. }));
        assert_eq!(transport.keys_seen, vec!["idem-1", "idem-1"]);
    }

    #[test]
    fn server_rejection_deletes_and_never_retries() {
        let queue = eligible_queue();
        let mut transport = Fake { responses: vec![Ok(receipt(false))], ..Fake::default() };
        let outcomes = upload_ready(&queue, &mut transport, &policy(), || 10).unwrap();
        assert!(matches!(outcomes[0], UploadOutcome::RejectedByServer { .. }));
        assert_eq!(queue.job("job-1").unwrap().unwrap().state, JobState::Rejected);
        assert!(package_archive(&queue, "job-1").unwrap().is_none());
    }

    #[test]
    fn withdrawn_consent_cancels_before_transfer() {
        let queue = eligible_queue();
        queue.revoke_consent("consent-1", 5).unwrap();
        let mut transport = Fake::default();
        let outcomes = upload_ready(&queue, &mut transport, &policy(), || 10).unwrap();
        assert!(matches!(outcomes[0], UploadOutcome::Cancelled { .. }));
        assert!(transport.keys_seen.is_empty());
        assert!(package_archive(&queue, "job-1").unwrap().is_none());
    }

    #[test]
    fn production_is_refused_while_gates_are_unmet() {
        let queue = eligible_queue();
        let mut transport = Fake::default();
        let outcomes = upload_ready(
            &queue,
            &mut transport,
            &UploadPolicy { target: UploadTarget::Production, ..policy() },
            || 10,
        )
        .unwrap();
        assert!(matches!(outcomes[0], UploadOutcome::Cancelled { ref reason, .. } if reason == "upload_gates_unmet"));
        assert!(transport.keys_seen.is_empty());
    }

    #[test]
    fn receipt_after_withdrawal_requests_server_deletion() {
        let queue = std::rc::Rc::new(eligible_queue());
        let during = std::rc::Rc::clone(&queue);
        let mut transport = Fake {
            responses: vec![Ok(receipt(true))],
            on_upload: Some(Box::new(move || {
                // Withdrawal lands while the package is in flight.
                during
                    .transition_job("job-1", JobState::Deleted, &JobUpdate::default(), 6)
                    .unwrap();
            })),
            ..Fake::default()
        };
        let job = queue.job("job-1").unwrap().unwrap();
        let outcome = upload_job(&queue, &mut transport, &policy(), &job, || 10).unwrap();
        assert!(matches!(outcome, UploadOutcome::LateReceipt { .. }));
        assert_eq!(queue.job("job-1").unwrap().unwrap().state, JobState::Deleted);
        let pending = queue.pending_deletion_requests().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].1, "receipt:receipt-1");
        assert!(!queue.contributed_hashes().unwrap().contains("hash-1"));
    }

    #[test]
    fn deletion_requests_are_sent_and_completed() {
        let queue = eligible_queue();
        queue
            .insert_deletion_request(StoredDeletion { request_id: "d1", scope: "everything" }, 1)
            .unwrap();
        let mut transport = Fake::default();
        assert_eq!(send_deletion_requests(&queue, &mut transport, 2).unwrap(), 1);
        assert_eq!(transport.deletions[0].scope, DeletionScope::Everything);
        assert!(queue.pending_deletion_requests().unwrap().is_empty());
    }
}
