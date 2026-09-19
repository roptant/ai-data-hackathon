//! Asynchronous privacy-filtered training copies (plan §7, §9).
//!
//! Runs after delivery, on a copy, when dictation is idle. Consent is checked
//! at enqueue and rechecked before the package is stored. Every failure path
//! deletes what it produced; raw working data is deleted as soon as a package
//! exists or the session is rejected, and otherwise expires after 24 hours.

use std::collections::BTreeSet;

use dictation_core::{
    contribution::{
        ConsentRecord, ELIGIBILITY_VERSION, JobState, Refusal, SessionFacts, UploadTarget,
        check_session, recheck,
    },
    dataset::{DatasetSettings, Recheck, build_dataset},
    package::{PackageHeader, PackageLimits, build_package, encode_archive},
    privacy::{Classifier, POLICY_VERSION, PrivacySettings, analyze},
    transcript::{FrozenTranscript, Word},
    vad::{SessionActivity, VadSettings},
};
use dictation_storage::{
    DataClass, EncryptedStore, StoreError,
    contribution::{Job, JobUpdate},
    package_scope_id, session_scope_id,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Raw session lifetime when processing never completes (plan §6).
pub const RAW_SESSION_SECONDS: i64 = 24 * 3600;
/// Eligible package lifetime while offline (plan §6).
pub const PACKAGE_SECONDS: i64 = 7 * 24 * 3600;

#[must_use]
pub fn random_hex(bytes: usize) -> String {
    let mut buffer = vec![0_u8; bytes];
    rand::rngs::OsRng.fill_bytes(&mut buffer);
    hex::encode(buffer)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StoredWord {
    id: u64,
    text: String,
    display: String,
    start: u64,
    end: u64,
    confidence: f32,
    unreliable: bool,
}

/// Serialized header of a raw session. Audio follows as a separate artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StoredSession {
    session_id: String,
    started_at: i64,
    language: String,
    sample_rate: u32,
    total_samples: u64,
    asr_model: String,
    asr_model_revision: String,
    words: Vec<StoredWord>,
}

/// One finished dictation's training copy, held only in encrypted storage.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSession {
    pub session_id: String,
    pub started_at: i64,
    pub opted_out: bool,
    pub asr_model: String,
    pub asr_model_revision: String,
    pub transcript: FrozenTranscript,
    pub pcm: Vec<i16>,
}

fn encode_header(session: &RawSession) -> Result<Vec<u8>, StoreError> {
    let transcript = &session.transcript;
    serde_json::to_vec(&StoredSession {
        session_id: session.session_id.clone(),
        started_at: session.started_at,
        language: transcript.language().to_owned(),
        sample_rate: transcript.sample_rate(),
        total_samples: transcript.total_samples(),
        asr_model: session.asr_model.clone(),
        asr_model_revision: session.asr_model_revision.clone(),
        words: transcript
            .words()
            .iter()
            .map(|word| StoredWord {
                id: word.id(),
                text: word.text().to_owned(),
                display: word.display_text().to_owned(),
                start: word.interval().start(),
                end: word.interval().end(),
                confidence: word.confidence(),
                unreliable: word.timing_unreliable(),
            })
            .collect(),
    })
    .map_err(|_| StoreError::InvalidMetadata)
}

fn decode_session(header: &[u8], audio: &[u8]) -> Option<RawSession> {
    let stored: StoredSession = serde_json::from_slice(header).ok()?;
    let words = stored
        .words
        .into_iter()
        .map(|word| {
            Word::new(word.id, word.text, word.start, word.end, word.confidence)
                .ok()
                .map(|built| built.with_display(word.display).with_unreliable_timing(word.unreliable))
        })
        .collect::<Option<Vec<_>>>()?;
    let transcript = FrozenTranscript::new(
        stored.session_id.clone(),
        1,
        words,
        stored.sample_rate,
        stored.total_samples,
        stored.language,
    )
    .ok()?;
    if audio.len() % 2 != 0 {
        return None;
    }
    Some(RawSession {
        session_id: stored.session_id,
        started_at: stored.started_at,
        opted_out: false,
        asr_model: stored.asr_model,
        asr_model_revision: stored.asr_model_revision,
        transcript,
        pcm: audio
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect(),
    })
}

/// The two databases the pipeline touches. The upload worker never receives
/// the sessions store.
pub struct Stores<'a> {
    pub sessions: &'a mut EncryptedStore,
    pub queue: &'a mut EncryptedStore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnqueueOutcome {
    Enqueued { job_id: String },
    Refused(Refusal),
}

/// Stores a session's training copy if, and only if, consent allows it.
///
/// # Errors
///
/// Fails on storage errors; nothing is persisted when consent refuses.
pub fn enqueue_session(
    stores: &mut Stores<'_>,
    session: &RawSession,
    target: UploadTarget,
    now: i64,
) -> Result<EnqueueOutcome, StoreError> {
    let consent = stores.queue.current_consent()?;
    let facts = SessionFacts {
        started_at: session.started_at,
        opted_out: session.opted_out,
        language: session.transcript.language(),
    };
    let consent_id = match check_session(consent.as_ref(), facts, target, now) {
        Ok(id) => id,
        Err(refusal) => return Ok(EnqueueOutcome::Refused(refusal)),
    };
    let scope = session_scope_id(&session.session_id);
    let expires_at = session.started_at.min(now) + RAW_SESSION_SECONDS;
    stores
        .sessions
        .create_scope(&scope, DataClass::RawSession, expires_at, now)?;
    let audio: Vec<u8> = session.pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
    let stored_session = stores
        .sessions
        .put_artifact(&scope, &format!("{scope}:header"), "session_header", &encode_header(session)?, now)
        .and_then(|()| {
            stores
                .sessions
                .put_artifact(&scope, &format!("{scope}:audio"), "session_audio", &audio, now)
        });
    if let Err(error) = stored_session {
        let _ = stores.sessions.delete_scope(&scope);
        return Err(error);
    }
    let job_id = format!("job-{}", random_hex(16));
    if let Err(error) = stores.queue.insert_job(
        &job_id,
        &session.session_id,
        &consent_id,
        session.started_at,
        now,
        expires_at,
    ) {
        let _ = stores.sessions.delete_scope(&scope);
        return Err(error);
    }
    Ok(EnqueueOutcome::Enqueued { job_id })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobOutcome {
    Eligible { job_id: String, sample_id: String, duration_ms: u64 },
    Rejected { job_id: String, reason: String },
    /// Dictation started; the job returns to the queue unchanged.
    Preempted { job_id: String },
    Idle,
}

/// Rules plus the privacy model over retained text only.
struct CombinedRecheck<'a, C: Classifier> {
    classifier: &'a mut C,
    settings: PrivacySettings,
    private_terms: &'a [String],
}

impl<C: Classifier> Recheck for CombinedRecheck<'_, C> {
    fn verify_clean(&mut self, retained: &FrozenTranscript) -> Result<(), &'static str> {
        let analysis = analyze(retained, self.classifier, self.settings, self.private_terms);
        if let Some(reason) = analysis.rejected_reason {
            return Err(if reason == "empty_transcript" {
                "empty_after_removal"
            } else {
                reason
            });
        }
        if analysis.uncertain {
            return Err("uncertain_after_removal");
        }
        if !analysis.spans.is_empty() {
            return Err("sensitive_after_removal");
        }
        Ok(())
    }
}

fn load_session(sessions: &EncryptedStore, session_id: &str) -> Result<Option<RawSession>, StoreError> {
    let scope = session_scope_id(session_id);
    let header = sessions.get_artifact(&format!("{scope}:header"))?;
    let audio = sessions.get_artifact(&format!("{scope}:audio"))?;
    Ok(header.zip(audio).and_then(|(header, audio)| decode_session(&header, &audio)))
}

fn reject(
    stores: &mut Stores<'_>,
    job: &Job,
    reason: &str,
    now: i64,
) -> Result<JobOutcome, StoreError> {
    let _ = stores.queue.transition_job(
        &job.job_id,
        JobState::Rejected,
        &JobUpdate {
            reason: Some(reason),
            ..JobUpdate::default()
        },
        now,
    );
    // No review bucket: rejected material is deleted at the decision.
    stores.sessions.delete_scope(&session_scope_id(&job.session_id))?;
    stores.queue.delete_scope(&package_scope_id(&job.job_id))?;
    Ok(JobOutcome::Rejected {
        job_id: job.job_id.clone(),
        reason: reason.to_owned(),
    })
}

pub struct TrainingSettings {
    pub target: UploadTarget,
    pub dataset: DatasetSettings,
    pub vad: VadSettings,
    pub private_terms: Vec<String>,
    pub privacy_model_revision: String,
}

/// Processes the oldest pending job, if any.
///
/// # Errors
///
/// Fails on storage errors. Classifier and dataset failures reject the job.
#[allow(clippy::too_many_lines)]
pub fn process_next_job<C: Classifier>(
    stores: &mut Stores<'_>,
    classifier: &mut C,
    settings: &TrainingSettings,
    is_preempted: impl Fn() -> bool,
    now: impl Fn() -> i64,
) -> Result<JobOutcome, StoreError> {
    let Some(job) = stores.queue.jobs_in(&[JobState::LocalPending])?.into_iter().next() else {
        return Ok(JobOutcome::Idle);
    };
    if is_preempted() {
        return Ok(JobOutcome::Preempted { job_id: job.job_id });
    }
    let consent: Option<ConsentRecord> = stores.queue.consent(&job.consent_id)?;
    if let Err(refusal) = recheck(consent.as_ref(), settings.target, now()) {
        return reject(stores, &job, refusal.code(), now());
    }
    let job = stores
        .queue
        .transition_job(&job.job_id, JobState::Analyzing, &JobUpdate::default(), now())?;
    let Some(session) = load_session(stores.sessions, &job.session_id)? else {
        return reject(stores, &job, "raw_session_missing_or_expired", now());
    };
    let analysis = analyze(
        &session.transcript,
        classifier,
        settings.dataset.privacy,
        &settings.private_terms,
    );
    if is_preempted() {
        stores
            .queue
            .transition_job(&job.job_id, JobState::LocalPending, &JobUpdate::default(), now())?;
        return Ok(JobOutcome::Preempted { job_id: job.job_id });
    }
    stores
        .queue
        .transition_job(&job.job_id, JobState::Building, &JobUpdate::default(), now())?;
    let activity = SessionActivity::analyze(&session.pcm, session.transcript.sample_rate(), settings.vad);
    let seen: BTreeSet<String> = stores.queue.contributed_hashes()?;
    let mut recheck_stage = CombinedRecheck {
        classifier,
        settings: settings.dataset.privacy,
        private_terms: &settings.private_terms,
    };
    let result = build_dataset(
        &session.transcript,
        &analysis,
        &session.pcm,
        &activity,
        settings.dataset,
        &seen,
        Some(&mut recheck_stage),
    );
    if is_preempted() {
        stores
            .queue
            .transition_job(&job.job_id, JobState::LocalPending, &JobUpdate::default(), now())?;
        return Ok(JobOutcome::Preempted { job_id: job.job_id });
    }
    if let Some(reason) = result.rejection() {
        return reject(stores, &job, reason, now());
    }
    // Recheck consent before anything is written as upload-ready.
    let consent = stores.queue.consent(&job.consent_id)?;
    if let Err(refusal) = recheck(consent.as_ref(), settings.target, now()) {
        return reject(stores, &job, refusal.code(), now());
    }
    let sample_id = format!("sample-{}", random_hex(16));
    let header = PackageHeader {
        eligibility_version: ELIGIBILITY_VERSION,
        sample_id: sample_id.clone(),
        language: session.transcript.language().to_owned(),
        asr_model: session.asr_model.clone(),
        asr_model_revision: session.asr_model_revision.clone(),
        privacy_model: analysis.classifier_model.clone(),
        privacy_model_revision: settings.privacy_model_revision.clone(),
        policy_version: POLICY_VERSION.to_owned(),
        consent_version: consent.as_ref().map(|c| c.version.clone()).unwrap_or_default(),
        consent_reference: job.consent_id.clone(),
    };
    let package = match build_package(
        &result,
        &header,
        session.transcript.sample_rate(),
        PackageLimits::default(),
    )
    .and_then(|package| encode_archive(&package))
    {
        Ok(bytes) => bytes,
        Err(error) => return reject(stores, &job, error.code, now()),
    };
    let stamp = now();
    let scope = package_scope_id(&job.job_id);
    stores
        .queue
        .create_scope(&scope, DataClass::EligiblePackage, stamp + PACKAGE_SECONDS, stamp)?;
    stores
        .queue
        .put_artifact(&scope, &format!("{scope}:archive"), "package_archive", &package, stamp)?;
    let idempotency_key = format!("idem-{}", random_hex(16));
    let duration_ms = result.metrics.duration_ms;
    let transitioned = stores.queue.transition_job(
        &job.job_id,
        JobState::Eligible,
        &JobUpdate {
            reason: Some(""),
            sample_id: Some(&sample_id),
            idempotency_key: Some(&idempotency_key),
            content_hash: Some(&result.content_hash),
            duration_ms: Some(duration_ms),
            ..JobUpdate::default()
        },
        stamp,
    );
    if let Err(error) = transitioned {
        // Withdrawal won the race: leave nothing upload-ready behind.
        stores.queue.delete_scope(&scope)?;
        stores.sessions.delete_scope(&session_scope_id(&job.session_id))?;
        return Err(error);
    }
    // Raw working data goes now that the package exists (plan §6).
    stores.sessions.delete_scope(&session_scope_id(&job.session_id))?;
    Ok(JobOutcome::Eligible {
        job_id: job.job_id,
        sample_id,
        duration_ms,
    })
}

/// Loads a job's archive for transfer. Only the queue store is needed.
///
/// # Errors
///
/// Fails on storage errors.
pub fn package_archive(queue: &EncryptedStore, job_id: &str) -> Result<Option<Vec<u8>>, StoreError> {
    let scope = package_scope_id(job_id);
    queue.get_artifact(&format!("{scope}:archive"))
}

#[cfg(test)]
mod tests {
    use dictation_core::{
        contribution::{CONSENT_VERSION, Purpose},
        package::{PackageLimits, decode_archive, validate_package},
        privacy::ClassifierFailure,
    };
    use dictation_storage::MasterKeyProvider;

    use super::*;

    struct Keys;
    impl MasterKeyProvider for Keys {
        fn load_or_create(&self) -> Result<Option<[u8; 32]>, String> {
            Ok(Some([4; 32]))
        }
    }

    struct Clean;
    impl Classifier for Clean {
        fn model_id(&self) -> String {
            "clean-test".to_owned()
        }
        fn classify(&mut self, _: &str, _: &str, _: &str, _: u32) -> Result<String, ClassifierFailure> {
            Ok(r#"{"spans":[],"uncertain":false}"#.to_owned())
        }
    }

    struct Broken;
    impl Classifier for Broken {
        fn model_id(&self) -> String {
            "broken".to_owned()
        }
        fn classify(&mut self, _: &str, _: &str, _: &str, _: u32) -> Result<String, ClassifierFailure> {
            Err(ClassifierFailure::Timeout)
        }
    }

    fn session(started_at: i64) -> RawSession {
        let text = "Send the parcel to Jane at 14 Oak Street. The meeting starts tomorrow.";
        let mut words = Vec::new();
        let mut pcm = vec![20_i16; 8_000];
        for (index, token) in text.split_whitespace().enumerate() {
            let start = pcm.len() as u64;
            pcm.extend((0..6_400).map(|n| if n % 2 == 0 { 9_000 } else { -9_000 }));
            let end = pcm.len() as u64;
            pcm.extend(std::iter::repeat_n(20_i16, 1_600));
            if token.ends_with('.') {
                pcm.extend(std::iter::repeat_n(20_i16, 12_000));
            }
            let spoken = token.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_owned();
            words.push(Word::new(index as u64, spoken, start, end, 0.95).unwrap().with_display(token));
        }
        pcm.extend(std::iter::repeat_n(20_i16, 8_000));
        let total = pcm.len() as u64;
        RawSession {
            session_id: format!("session-{started_at}"),
            started_at,
            opted_out: false,
            asr_model: "whisper-base-q5_1".to_owned(),
            asr_model_revision: "r".to_owned(),
            transcript: FrozenTranscript::new(format!("session-{started_at}"), 1, words, 16_000, total, "en").unwrap(),
            pcm,
        }
    }

    fn consent(granted_at: i64) -> ConsentRecord {
        ConsentRecord {
            consent_id: "consent-1".to_owned(),
            version: CONSENT_VERSION.to_owned(),
            purposes: vec![Purpose::CustomerPersonalization],
            granted_at,
            expires_at: granted_at + 100_000,
            revoked_at: None,
            paused: false,
            policy_version: POLICY_VERSION.to_owned(),
        }
    }

    fn settings() -> TrainingSettings {
        TrainingSettings {
            target: UploadTarget::NonProduction,
            dataset: DatasetSettings::default(),
            vad: VadSettings::default(),
            private_terms: Vec::new(),
            privacy_model_revision: "r".to_owned(),
        }
    }

    #[test]
    fn without_consent_nothing_is_stored() {
        let mut sessions = EncryptedStore::open_in_memory(&Keys).unwrap();
        let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
        let mut stores = Stores { sessions: &mut sessions, queue: &mut queue };
        let outcome = enqueue_session(&mut stores, &session(100), UploadTarget::NonProduction, 100).unwrap();
        assert_eq!(outcome, EnqueueOutcome::Refused(Refusal::NoConsent));
        assert!(queue.jobs_in(&JobState::ALL).unwrap().is_empty());
        assert!(sessions.get_artifact("session:session-100:header").unwrap().is_none());
    }

    #[test]
    fn opted_in_session_becomes_a_valid_package_and_raw_data_is_deleted() {
        let mut sessions = EncryptedStore::open_in_memory(&Keys).unwrap();
        let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
        queue.insert_consent(&consent(50)).unwrap();
        let mut stores = Stores { sessions: &mut sessions, queue: &mut queue };
        let EnqueueOutcome::Enqueued { job_id } =
            enqueue_session(&mut stores, &session(100), UploadTarget::NonProduction, 100).unwrap()
        else {
            panic!("expected enqueue");
        };
        let outcome = process_next_job(&mut stores, &mut Clean, &settings(), || false, || 200).unwrap();
        assert!(matches!(outcome, JobOutcome::Eligible { .. }), "{outcome:?}");
        assert!(sessions.get_artifact("session:session-100:audio").unwrap().is_none());
        let archive = package_archive(&queue, &job_id).unwrap().unwrap();
        let (manifest, payloads) = decode_archive(&archive, PackageLimits::default()).unwrap();
        validate_package(&manifest, &payloads, &BTreeSet::from([ELIGIBILITY_VERSION]), PackageLimits::default()).unwrap();
        let text = String::from_utf8_lossy(&archive);
        assert!(!text.contains("Jane") && !text.contains("Oak"));
        assert_eq!(queue.job(&job_id).unwrap().unwrap().state, JobState::Eligible);
    }

    #[test]
    fn classifier_failure_rejects_and_deletes() {
        let mut sessions = EncryptedStore::open_in_memory(&Keys).unwrap();
        let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
        queue.insert_consent(&consent(50)).unwrap();
        let mut stores = Stores { sessions: &mut sessions, queue: &mut queue };
        enqueue_session(&mut stores, &session(100), UploadTarget::NonProduction, 100).unwrap();
        let outcome = process_next_job(&mut stores, &mut Broken, &settings(), || false, || 200).unwrap();
        assert!(matches!(outcome, JobOutcome::Rejected { ref reason, .. } if reason == "classifier_timeout"));
        assert!(sessions.get_artifact("session:session-100:audio").unwrap().is_none());
    }

    #[test]
    fn earlier_sessions_are_not_swept_up_and_withdrawal_wins() {
        let mut sessions = EncryptedStore::open_in_memory(&Keys).unwrap();
        let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
        queue.insert_consent(&consent(150)).unwrap();
        let mut stores = Stores { sessions: &mut sessions, queue: &mut queue };
        assert_eq!(
            enqueue_session(&mut stores, &session(100), UploadTarget::NonProduction, 160).unwrap(),
            EnqueueOutcome::Refused(Refusal::SessionBeforeConsent)
        );
        enqueue_session(&mut stores, &session(170), UploadTarget::NonProduction, 170).unwrap();
        stores.queue.revoke_consent("consent-1", 180).unwrap();
        let outcome = process_next_job(&mut stores, &mut Clean, &settings(), || false, || 190).unwrap();
        assert!(matches!(outcome, JobOutcome::Rejected { ref reason, .. } if reason == "consent_expired_or_revoked"));
    }

    #[test]
    fn preemption_returns_the_job_to_the_queue() {
        let mut sessions = EncryptedStore::open_in_memory(&Keys).unwrap();
        let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
        queue.insert_consent(&consent(50)).unwrap();
        let mut stores = Stores { sessions: &mut sessions, queue: &mut queue };
        let EnqueueOutcome::Enqueued { job_id } =
            enqueue_session(&mut stores, &session(100), UploadTarget::NonProduction, 100).unwrap()
        else {
            panic!("expected enqueue");
        };
        let calls = std::cell::Cell::new(0);
        let outcome = process_next_job(
            &mut stores,
            &mut Clean,
            &settings(),
            || {
                calls.set(calls.get() + 1);
                calls.get() > 1
            },
            || 200,
        )
        .unwrap();
        assert!(matches!(outcome, JobOutcome::Preempted { .. }));
        assert_eq!(queue.job(&job_id).unwrap().unwrap().state, JobState::LocalPending);
        assert!(sessions.get_artifact("session:session-100:audio").unwrap().is_some());
    }
}
