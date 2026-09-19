//! Narrow command boundary between the WebView and the trusted core.
//!
//! The WebView receives status, settings (without secrets), the disclosure,
//! content-free history, and the in-memory result panel it explicitly asks
//! for. It never gets storage handles, audio, filesystem access, or tokens
//! other than a pairing token it cannot see (released only to the client).

use std::{
    collections::BTreeSet,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use dictation_core::contribution::{CONSENT_VERSION, ConsentRecord, DEFAULT_CONSENT_DAYS, JobState, Purpose, disclosure};
use dictation_engine::training::random_hex;
use dictation_models::{MODELS, delivery::PersonalModels};
use dictation_platform::clipboard;
use dictation_protocol::ConsentGrant;
use dictation_storage::{api_clients::Scope, contribution::DeletionRequest as StoredDeletion, session_scope_id};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use crate::{
    asr_thread::{AsrHandle, AsrRequest},
    controller::{Command, UiAction},
    settings::{SettingsUpdate, SettingsView},
    state::{CapabilityView, ResultPanel, Shared, StatusView},
};

type Shared_ = Arc<Shared>;

#[tauri::command]
pub fn get_status(shared: State<'_, Shared_>) -> StatusView {
    shared.status()
}

#[tauri::command]
pub fn get_settings(shared: State<'_, Shared_>) -> SettingsView {
    SettingsView::from(&shared.settings())
}

#[tauri::command]
pub fn update_settings(
    app: AppHandle,
    shared: State<'_, Shared_>,
    asr: State<'_, AsrHandle>,
    update: SettingsUpdate,
) -> Result<SettingsView, String> {
    let before = shared.settings();
    {
        let mut settings = shared.settings.lock().map_err(|_| "settings unavailable")?;
        settings.apply(update)?;
    }
    shared.save_settings();
    let after = shared.settings();
    if after.shortcuts != before.shortcuts {
        crate::shortcut_setup::register(&app, &shared)?;
    }
    if after.automatic_insertion && !before.automatic_insertion {
        shared.desktop.enable_accessibility().map_err(|error| format!("Accessibility could not be enabled: {error}"))?;
    }
    if after.asr_model != before.asr_model {
        asr.send(AsrRequest::Reload);
    }
    if (after.api_enabled, after.api_port) != (before.api_enabled, before.api_port) {
        crate::api_bridge::apply(&shared);
    }
    crate::refresh_capabilities(&shared);
    Ok(SettingsView::from(&after))
}

#[tauri::command]
pub fn capability_matrix(shared: State<'_, Shared_>) -> Option<CapabilityView> {
    crate::refresh_capabilities(&shared);
    shared.status().capabilities
}

#[tauri::command]
pub fn recording_action(shared: State<'_, Shared_>, action: String) -> Result<(), String> {
    let action = match action.as_str() {
        "toggle" => UiAction::Toggle,
        "stop" => UiAction::Stop,
        "cancel" => UiAction::Cancel,
        "lock" => UiAction::Lock,
        _ => return Err("unknown action".to_owned()),
    };
    shared.commands.send(Command::Ui(action)).map_err(|_| "controller unavailable".to_owned())
}

/// Marks a session "do not contribute"; removes it from the queue if already
/// enqueued and not yet uploaded.
#[tauri::command]
pub fn do_not_contribute(shared: State<'_, Shared_>, session_id: String) -> Result<(), String> {
    if let Ok(mut sessions) = shared.opted_out_sessions.lock() {
        sessions.push(session_id.clone());
        let excess = sessions.len().saturating_sub(100);
        sessions.drain(..excess);
    }
    let Some(stores) = &shared.stores else { return Ok(()) };
    let now = crate::unix_now();
    if let Ok(queue) = stores.queue.lock() {
        for job in queue.jobs_in(&JobState::ALL).map_err(|error| error.to_string())? {
            if job.session_id == session_id && job.state.is_cancellable() && job.state != JobState::Uploading {
                let _ = queue.transition_job(
                    &job.job_id,
                    JobState::Deleted,
                    &dictation_storage::contribution::JobUpdate { reason: Some("session_opted_out"), ..Default::default() },
                    now,
                );
                let _ = queue.delete_scope(&dictation_storage::package_scope_id(&job.job_id));
            }
        }
    }
    if let Ok(sessions) = stores.sessions.lock() {
        let _ = sessions.delete_scope(&session_scope_id(&session_id));
    }
    Ok(())
}

#[tauri::command]
pub fn get_result(shared: State<'_, Shared_>) -> Option<ResultPanel> {
    shared.result.lock().ok().and_then(|slot| slot.clone())
}

/// Copies the result at the user's request, then forgets it.
#[tauri::command]
pub fn copy_result(shared: State<'_, Shared_>) -> Result<(), String> {
    let result = shared.result.lock().map_err(|_| "unavailable")?.take();
    let Some(result) = result else { return Err("There is no result to copy.".to_owned()) };
    clipboard::place(&result.text).map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn dismiss_result(shared: State<'_, Shared_>) {
    if let Ok(mut slot) = shared.result.lock() {
        *slot = None;
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    id: &'static str,
    role: dictation_models::Role,
    display_name: &'static str,
    size_bytes: u64,
    quantization: &'static str,
    license: &'static str,
    provenance: &'static str,
    notes: &'static str,
    measured_peak_rss_mib: u32,
    state: dictation_models::InstallState,
}

#[tauri::command]
pub fn list_models(shared: State<'_, Shared_>) -> Vec<ModelInfo> {
    MODELS
        .iter()
        .map(|spec| ModelInfo {
            id: spec.id,
            role: spec.role,
            display_name: spec.display_name,
            size_bytes: spec.size_bytes,
            quantization: spec.quantization,
            license: spec.license,
            provenance: spec.conversion_provenance,
            notes: spec.notes,
            measured_peak_rss_mib: spec.measured_peak_rss_mib,
            state: dictation_models::state(&shared.layout.models(), spec),
        })
        .collect()
}

static DOWNLOAD_CANCEL: AtomicBool = AtomicBool::new(false);

/// Downloads a pinned model. This is an explicit, visible network operation.
#[tauri::command]
pub fn install_model(app: AppHandle, shared: State<'_, Shared_>, asr: State<'_, AsrHandle>, id: String) -> Result<(), String> {
    let spec = dictation_models::spec(&id).ok_or("unknown model")?;
    let directory = shared.layout.models();
    let asr = asr.inner().clone();
    DOWNLOAD_CANCEL.store(false, Ordering::SeqCst);
    std::thread::spawn(move || {
        let mut last = 0_u64;
        let result = dictation_models::download(spec, &directory, &DOWNLOAD_CANCEL, |received, total| {
            if received - last >= 4 * 1024 * 1024 || received == total {
                last = received;
                let _ = app.emit("model-progress", (spec.id, received, total));
            }
        });
        let _ = app.emit("model-installed", (spec.id, result.as_ref().err().map(ToString::to_string)));
        if result.is_ok() && spec.role == dictation_models::Role::Asr {
            asr.send(AsrRequest::Reload);
        }
    });
    Ok(())
}

#[tauri::command]
pub fn cancel_model_download() {
    DOWNLOAD_CANCEL.store(true, Ordering::SeqCst);
}

/// Installs a model file the user obtained separately; verified against the pin.
#[tauri::command]
pub fn import_model(shared: State<'_, Shared_>, asr: State<'_, AsrHandle>, id: String, path: String) -> Result<(), String> {
    let spec = dictation_models::spec(&id).ok_or("unknown model")?;
    dictation_models::import_local(&PathBuf::from(path), &shared.layout.models(), spec).map_err(|error| error.to_string())?;
    if spec.role == dictation_models::Role::Asr {
        asr.send(AsrRequest::Reload);
    }
    Ok(())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DisclosureView {
    version: &'static str,
    sections: Vec<(&'static str, &'static str)>,
    gates_met: bool,
}

#[tauri::command]
pub fn get_disclosure() -> DisclosureView {
    let text = disclosure();
    DisclosureView { version: text.version, sections: text.sections.to_vec(), gates_met: text.gates_met }
}

fn with_queue<T>(shared: &Shared, work: impl FnOnce(&mut dictation_storage::EncryptedStore) -> Result<T, String>) -> Result<T, String> {
    let stores = shared.stores.as_ref().ok_or("Secure storage is unavailable, so contribution cannot be enabled.")?;
    let mut queue = stores.queue.lock().map_err(|_| "storage unavailable")?;
    work(&mut queue)
}

/// Records explicit opt-in to exactly the disclosed version.
#[tauri::command]
pub fn grant_consent(shared: State<'_, Shared_>, accepted_version: String) -> Result<(), String> {
    if accepted_version != CONSENT_VERSION {
        return Err("The consent text changed; please review it again.".to_owned());
    }
    let now = crate::unix_now();
    let record = ConsentRecord {
        consent_id: format!("consent-{}", random_hex(12)),
        version: CONSENT_VERSION.to_owned(),
        purposes: vec![Purpose::CustomerPersonalization],
        granted_at: now,
        expires_at: now + DEFAULT_CONSENT_DAYS * 86_400,
        revoked_at: None,
        paused: false,
        policy_version: dictation_core::privacy::POLICY_VERSION.to_owned(),
    };
    with_queue(&shared, |queue| queue.insert_consent(&record).map_err(|error| error.to_string()))?;
    if let Some(mut transport) = crate::background_transport(&shared) {
        let _ = dictation_engine::upload::Transport::register_consent(
            &mut transport,
            &ConsentGrant {
                consent_id: record.consent_id,
                version: record.version,
                purposes: vec![Purpose::CustomerPersonalization.as_str().to_owned()],
                granted_at: record.granted_at,
                expires_at: record.expires_at,
            },
        );
    }
    Ok(())
}

#[tauri::command]
pub fn set_contribution_paused(shared: State<'_, Shared_>, paused: bool) -> Result<(), String> {
    with_queue(&shared, |queue| {
        let consent = queue.current_consent().map_err(|error| error.to_string())?.ok_or("No active consent.")?;
        queue.set_consent_paused(&consent.consent_id, paused).map_err(|error| error.to_string())?;
        Ok(())
    })
}

fn cancel_everything_locally(shared: &Shared, reason: &str) -> Result<(), String> {
    let now = crate::unix_now();
    let cancelled = with_queue(shared, |queue| queue.cancel_all_jobs(reason, now).map_err(|error| error.to_string()))?;
    if let Some(stores) = &shared.stores {
        if let Ok(sessions) = stores.sessions.lock() {
            for job in cancelled {
                let _ = sessions.delete_scope(&session_scope_id(&job.session_id));
            }
        }
    }
    Ok(())
}

/// Withdrawal: revokes consent, cancels queued and in-flight work, and starts
/// deletion of what the server received.
#[tauri::command]
pub fn withdraw_consent(shared: State<'_, Shared_>) -> Result<(), String> {
    let now = crate::unix_now();
    let consent = with_queue(&shared, |queue| queue.current_consent().map_err(|error| error.to_string()))?;
    if let Some(consent) = &consent {
        with_queue(&shared, |queue| queue.revoke_consent(&consent.consent_id, now).map(|_| ()).map_err(|error| error.to_string()))?;
    }
    cancel_everything_locally(&shared, "consent_withdrawn")?;
    let notified = consent.as_ref().zip(crate::background_transport(&shared)).is_some_and(|(consent, mut transport)| {
        dictation_engine::upload::Transport::withdraw_consent(&mut transport, &consent.consent_id).is_ok()
    });
    if !notified {
        // Retried by the background worker until the server confirms.
        with_queue(&shared, |queue| {
            queue
                .insert_deletion_request(StoredDeletion { request_id: &format!("deletion-{}", random_hex(12)), scope: "everything" }, now)
                .map_err(|error| error.to_string())
        })?;
    }
    let _ = shared.background_wake.send(());
    Ok(())
}

/// "Delete my training data and personalized model."
#[tauri::command]
pub fn delete_training_data(shared: State<'_, Shared_>, asr: State<'_, AsrHandle>) -> Result<(), String> {
    let now = crate::unix_now();
    cancel_everything_locally(&shared, "deletion_requested")?;
    with_queue(&shared, |queue| {
        queue
            .insert_deletion_request(StoredDeletion { request_id: &format!("deletion-{}", random_hex(12)), scope: "everything" }, now)
            .map_err(|error| error.to_string())
    })?;
    PersonalModels::new(&shared.layout.models()).delete_all().map_err(|error| error.to_string())?;
    asr.send(AsrRequest::Reload);
    let _ = shared.background_wake.send(());
    Ok(())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryView {
    consents: Vec<ConsentView>,
    jobs: Vec<JobView>,
    pending_deletions: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsentView {
    consent_id: String,
    version: String,
    granted_at: i64,
    expires_at: i64,
    revoked_at: Option<i64>,
    paused: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobView {
    state: &'static str,
    reason: String,
    duration_ms: u64,
    created_at: i64,
    updated_at: i64,
}

/// Contribution history without raw sensitive text.
#[tauri::command]
pub fn contribution_history(shared: State<'_, Shared_>) -> Result<HistoryView, String> {
    with_queue(&shared, |queue| {
        let consents = queue
            .consent_history()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|record| ConsentView {
                consent_id: record.consent_id,
                version: record.version,
                granted_at: record.granted_at,
                expires_at: record.expires_at,
                revoked_at: record.revoked_at,
                paused: record.paused,
            })
            .collect();
        let jobs = queue
            .jobs_in(&JobState::ALL)
            .map_err(|error| error.to_string())?
            .into_iter()
            .rev()
            .take(200)
            .map(|job| JobView {
                state: job.state.as_str(),
                reason: job.reason,
                duration_ms: job.duration_ms,
                created_at: job.created_at,
                updated_at: job.updated_at,
            })
            .collect();
        let pending_deletions = queue.pending_deletion_requests().map_err(|error| error.to_string())?.len();
        Ok(HistoryView { consents, jobs, pending_deletions })
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairingView {
    pairing_id: String,
    client_name: String,
    requested_scopes: Vec<&'static str>,
    verification_code: String,
}

#[tauri::command]
pub fn list_pairings(shared: State<'_, Shared_>) -> Vec<PairingView> {
    shared
        .pairings
        .pending()
        .into_iter()
        .map(|request| PairingView {
            pairing_id: request.pairing_id,
            client_name: request.client_name,
            requested_scopes: request.requested_scopes.iter().map(|scope| scope.as_str()).collect(),
            verification_code: request.verification_code,
        })
        .collect()
}

/// Approves a pairing with at most the requested scopes.
#[tauri::command]
pub fn approve_pairing(shared: State<'_, Shared_>, pairing_id: String, scopes: Vec<String>) -> Result<(), String> {
    let request = shared.pairings.request(&pairing_id).ok_or("That pairing request expired.")?;
    let requested: BTreeSet<Scope> = request.requested_scopes.iter().copied().collect();
    let granted: Vec<Scope> = scopes.iter().filter_map(|scope| Scope::parse(scope)).filter(|scope| requested.contains(scope)).collect();
    if granted.is_empty() {
        return Err("Grant at least one requested permission, or deny the request.".to_owned());
    }
    let stores = shared.stores.as_ref().ok_or("Secure storage is unavailable.")?;
    let token = format!("ldc_{}", random_hex(32));
    let client_id = format!("client-{}", random_hex(8));
    stores
        .metadata
        .lock()
        .map_err(|_| "storage unavailable")?
        .insert_api_client(&client_id, &request.client_name, &token, &granted, crate::unix_now())
        .map_err(|error| error.to_string())?;
    shared.pairings.approve(&pairing_id, token);
    Ok(())
}

#[tauri::command]
pub fn deny_pairing(shared: State<'_, Shared_>, pairing_id: String) {
    shared.pairings.deny(&pairing_id);
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientView {
    client_id: String,
    display_name: String,
    scopes: Vec<&'static str>,
    created_at: i64,
    last_used_at: Option<i64>,
    revoked: bool,
}

#[tauri::command]
pub fn list_clients(shared: State<'_, Shared_>) -> Result<Vec<ClientView>, String> {
    let stores = shared.stores.as_ref().ok_or("Secure storage is unavailable.")?;
    let clients = stores.metadata.lock().map_err(|_| "storage unavailable")?.api_clients().map_err(|error| error.to_string())?;
    Ok(clients
        .into_iter()
        .map(|client| ClientView {
            client_id: client.client_id,
            display_name: client.display_name,
            scopes: client.scopes.iter().map(|scope| scope.as_str()).collect(),
            created_at: client.created_at,
            last_used_at: client.last_used_at,
            revoked: client.revoked_at.is_some(),
        })
        .collect())
}

#[tauri::command]
pub fn revoke_client(shared: State<'_, Shared_>, client_id: String) -> Result<(), String> {
    let stores = shared.stores.as_ref().ok_or("Secure storage is unavailable.")?;
    stores
        .metadata
        .lock()
        .map_err(|_| "storage unavailable")?
        .revoke_api_client(&client_id, crate::unix_now())
        .map_err(|error| error.to_string())?;
    shared.bump_revocations();
    Ok(())
}

/// Lets the UI wait briefly for the portal consent dialog to be answered.
#[tauri::command]
pub fn retry_shortcuts(app: AppHandle, shared: State<'_, Shared_>) -> Result<(), String> {
    std::thread::sleep(Duration::from_millis(50));
    crate::shortcut_setup::register(&app, &shared)
}
