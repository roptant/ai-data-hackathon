//! Supervised background work: retention cleanup, privacy processing,
//! uploads, deletion requests, and personalized-model delivery.
//!
//! Privacy processing runs only while no dictation is active; new dictation
//! preempts it (plan §4). On low-memory machines the recognition worker is
//! released before the privacy model loads and reloads afterwards.

use std::{
    sync::{Arc, mpsc::Receiver},
    time::{Duration, Instant},
};

use dictation_core::{
    contribution::{JobState, RetrySettings, VolumeCaps},
    dataset::DatasetSettings,
    vad::VadSettings,
};
use dictation_engine::{
    training::{JobOutcome, Stores as EngineStores, TrainingSettings, process_next_job},
    upload::{HttpTransport, UploadPolicy, send_deletion_requests, upload_ready},
    workers::{ModelVerifier, PrivacyEngine, WorkerPaths, is_low_memory},
};
use dictation_models::{
    InstallState, Role,
    delivery::{PersonalModels, verify_manifest},
};

use crate::{
    asr_thread::{AsrHandle, AsrRequest},
    settings::ContributionTarget,
    state::Shared,
};

/// Terminal job rows are kept this long as content-free history.
const JOB_HISTORY_SECONDS: i64 = 90 * 24 * 3600;
const MODEL_CHECK_INTERVAL: Duration = Duration::from_secs(6 * 3600);

pub fn spawn(
    shared: Arc<Shared>,
    wake: Receiver<()>,
    workers: WorkerPaths,
    verifier: Arc<ModelVerifier>,
    asr: AsrHandle,
) {
    std::thread::Builder::new()
        .name("background".to_owned())
        .spawn(move || run(&shared, &wake, &workers, &verifier, &asr))
        .expect("the background thread can be spawned");
}

fn run(shared: &Shared, wake: &Receiver<()>, workers: &WorkerPaths, verifier: &ModelVerifier, asr: &AsrHandle) {
    let mut privacy: Option<PrivacyEngine> = None;
    let mut last_model_check: Option<Instant> = None;
    loop {
        let _ = wake.recv_timeout(Duration::from_secs(30));
        while wake.try_recv().is_ok() {}
        cleanup(shared);
        if shared.preemption.is_set() {
            continue;
        }
        let settings = shared.settings();
        if settings.contribution_target != ContributionTarget::Disabled {
            process_jobs(shared, workers, verifier, asr, &mut privacy);
            transfer(shared);
        }
        if privacy.is_some() && !has_pending(shared) {
            if let Some(mut engine) = privacy.take() {
                engine.release();
            }
            shared.publish(|status| status.privacy_model_ready = false);
        }
        if last_model_check.is_none_or(|checked| checked.elapsed() >= MODEL_CHECK_INTERVAL) {
            last_model_check = Some(Instant::now());
            check_personal_model(shared, asr);
        }
    }
}

fn cleanup(shared: &Shared) {
    let Some(stores) = &shared.stores else { return };
    let now = crate::unix_now();
    for store in [&stores.sessions, &stores.queue, &stores.metadata] {
        if let Ok(mut store) = store.lock() {
            let _ = store.cleanup_expired(now);
        }
    }
    if let Ok(queue) = stores.queue.lock() {
        if let Ok(jobs) = queue.jobs_in(&JobState::ALL) {
            for job in jobs {
                if job.state.is_terminal() && now - job.updated_at > JOB_HISTORY_SECONDS {
                    let _ = queue.purge_job(&job.job_id);
                } else if !job.state.is_terminal() && job.expires_at <= now {
                    let _ = queue.transition_job(
                        &job.job_id,
                        JobState::Deleted,
                        &dictation_storage::contribution::JobUpdate {
                            reason: Some("expired"),
                            ..Default::default()
                        },
                        now,
                    );
                }
            }
        }
    }
}

fn has_pending(shared: &Shared) -> bool {
    shared.stores.as_ref().is_some_and(|stores| {
        stores
            .queue
            .lock()
            .ok()
            .and_then(|queue| queue.jobs_in(&[JobState::LocalPending]).ok())
            .is_some_and(|jobs| !jobs.is_empty())
    })
}

fn process_jobs(
    shared: &Shared,
    workers: &WorkerPaths,
    verifier: &ModelVerifier,
    asr: &AsrHandle,
    privacy: &mut Option<PrivacyEngine>,
) {
    let Some(stores) = &shared.stores else { return };
    if !has_pending(shared) {
        return;
    }
    let spec = dictation_models::default_for(Role::Privacy);
    if dictation_models::state(&shared.layout.models(), spec) != InstallState::Installed {
        shared.publish(|status| status.training_status = "privacy_model_not_installed".to_owned());
        return;
    }
    if privacy.is_none() {
        if is_low_memory() {
            asr.send(AsrRequest::Release);
        }
        match PrivacyEngine::new(workers, &shared.layout.models(), spec, verifier, shared.preemption.clone()) {
            Ok(engine) => *privacy = Some(engine),
            Err(_) => {
                shared.publish(|status| status.training_status = "privacy_worker_unavailable".to_owned());
                return;
            }
        }
        shared.publish(|status| status.privacy_model_ready = true);
    }
    let Some(engine) = privacy.as_mut() else { return };
    let settings = shared.settings();
    let training = TrainingSettings {
        target: settings.contribution_target.upload_target(),
        dataset: DatasetSettings::default(),
        vad: VadSettings::default(),
        private_terms: settings.private_terms.clone(),
        privacy_model_revision: spec.revision.to_owned(),
    };
    while !shared.preemption.is_set() {
        shared.publish(|status| status.training_status = "filtering".to_owned());
        let (Ok(mut sessions), Ok(mut queue)) = (stores.sessions.lock(), stores.queue.lock()) else { return };
        let mut engine_stores = EngineStores { sessions: &mut sessions, queue: &mut queue };
        let outcome = process_next_job(&mut engine_stores, engine, &training, || shared.preemption.is_set(), crate::unix_now);
        drop(engine_stores);
        engine.clear_preempted();
        let label = match outcome {
            Ok(JobOutcome::Idle) => break,
            Ok(JobOutcome::Eligible { .. }) => "eligible_waiting_for_upload".to_owned(),
            Ok(JobOutcome::Rejected { reason, .. }) => format!("discarded:{reason}"),
            Ok(JobOutcome::Preempted { .. }) => "paused_for_dictation".to_owned(),
            Err(_) => "storage_error".to_owned(),
        };
        shared.publish(|status| status.training_status = label);
    }
    if is_low_memory() && !shared.preemption.is_set() {
        // Give the memory back to recognition before the next dictation.
        if let Some(mut released) = privacy.take() {
            released.release();
        }
        asr.send(AsrRequest::Warm);
    }
}

fn transport(shared: &Shared) -> Option<HttpTransport> {
    let settings = shared.settings();
    if settings.server_url.is_empty() || settings.server_token.is_empty() {
        return None;
    }
    HttpTransport::new(&settings.server_url, &settings.server_token, settings.contribution_target.upload_target()).ok()
}

fn transfer(shared: &Shared) {
    let (Some(stores), Some(mut transport)) = (&shared.stores, transport(shared)) else { return };
    let Ok(queue) = stores.queue.lock() else { return };
    // Idempotent: the server records each consent once and rechecks it at
    // admission and before training.
    if let Ok(Some(consent)) = queue.current_consent() {
        let _ = dictation_engine::upload::Transport::register_consent(
            &mut transport,
            &dictation_protocol::ConsentGrant {
                consent_id: consent.consent_id.clone(),
                version: consent.version.clone(),
                purposes: consent.purposes.iter().map(|purpose| purpose.as_str().to_owned()).collect(),
                granted_at: consent.granted_at,
                expires_at: consent.expires_at,
            },
        );
    }
    let policy = UploadPolicy {
        target: shared.settings().contribution_target.upload_target(),
        caps: VolumeCaps::default(),
        retry: RetrySettings::default(),
    };
    if let Ok(outcomes) = upload_ready(&queue, &mut transport, &policy, crate::unix_now) {
        if !outcomes.is_empty() {
            let label = format!("uploaded_or_retried:{}", outcomes.len());
            drop(queue);
            shared.publish(|status| status.training_status = label);
        }
    }
    if let Ok(queue) = stores.queue.lock() {
        let _ = send_deletion_requests(&queue, &mut transport, crate::unix_now());
    }
}

/// Installs a newly delivered personalized model, or removes a revoked one.
fn check_personal_model(shared: &Shared, asr: &AsrHandle) {
    let settings = shared.settings();
    let Some(transport) = transport(shared) else { return };
    let Ok(key_bytes) = hex_key(&settings.delivery_public_key) else { return };
    let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes) else { return };
    let personal = PersonalModels::new(&shared.layout.models());
    match transport.latest_model() {
        Ok(Some((version, signed))) => {
            let installed = personal.active().and_then(|path| path.file_stem().map(|stem| stem.to_string_lossy().into_owned()));
            if installed.as_deref() == Some(version.as_str()) {
                return;
            }
            let budget = crate::device_budget_mib();
            let Ok(manifest) = verify_manifest(&signed, &key, &settings.asr_model, budget) else { return };
            let Ok(artifact) = transport.model_artifact(&version) else { return };
            if personal.install(&manifest, &artifact).is_ok() {
                asr.send(AsrRequest::Reload);
            }
        }
        Ok(None) => {
            // The server revoked or deleted the personalized model.
            if personal.active().is_some() && personal.delete_all().is_ok() {
                asr.send(AsrRequest::Reload);
            }
        }
        Err(_) => {}
    }
}

fn hex_key(text: &str) -> Result<[u8; 32], ()> {
    let bytes = hex::decode(text).map_err(|_| ())?;
    bytes.try_into().map_err(|_| ())
}
