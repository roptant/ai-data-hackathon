//! Local Dictation desktop shell.
//!
//! Microphone capture, secrets, storage, inference, and upload policy live in
//! Rust threads and isolated workers. The `WebView` gets a narrow command set
//! (see `capabilities/`) and no filesystem, shell, or network permission.

// Tauri command handlers and serialized view structs intentionally use owned
// parameters and several booleans because those shapes are the IPC contract.
#![allow(clippy::needless_pass_by_value, clippy::struct_excessive_bools)]
// These explicit drops release structs that borrow locked stores before the
// following status/update work.
#![allow(clippy::drop_non_drop)]

mod api_bridge;
mod asr_thread;
mod background;
mod commands;
mod controller;
mod delivery;
mod indicator;
mod settings;
mod shortcut_setup;
mod state;
mod training_copy;
mod tray;

use std::{
    sync::{Arc, Mutex, atomic::AtomicU64, mpsc::channel},
    time::{SystemTime, UNIX_EPOCH},
};

use dictation_api::{EventBus, Pairings};
use dictation_engine::workers::{ModelVerifier, Preemption, WorkerPaths, total_memory_mib};
use dictation_platform::desktop::DesktopServices;
use dictation_storage::{EncryptedStore, keys::OsKeyring, layout::DataLayout};
use tauri::{AppHandle, Manager, WindowEvent};

use crate::{
    controller::{Command, Controller},
    settings::AppSettings,
    state::{CapabilityView, Shared, StatusView, Stores},
};

#[must_use]
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX))
}

/// Memory a personalized recognition model may use on this device. Provisional
/// rule: a quarter of physical memory, capped at 1.5 GB, 1 GB when unknown.
#[must_use]
pub fn device_budget_mib() -> u32 {
    total_memory_mib().map_or(1_024, |total| u32::try_from((total / 4).min(1_536)).unwrap_or(1_024))
}

pub fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

pub(crate) fn background_transport(shared: &Shared) -> Option<dictation_engine::upload::HttpTransport> {
    let settings = shared.settings();
    if settings.server_url.is_empty() || settings.server_token.is_empty() {
        return None;
    }
    dictation_engine::upload::HttpTransport::new(
        &settings.server_url,
        &settings.server_token,
        settings.contribution_target.upload_target(),
    )
    .ok()
}

pub(crate) fn refresh_capabilities(shared: &Shared) {
    let capabilities = shared
        .desktop
        .capabilities(shared.portal_shortcuts_active.load(std::sync::atomic::Ordering::SeqCst));
    let lifecycle = shared.status().capabilities.map_or_else(dictation_platform::LifecycleReport::default, |view| {
        dictation_platform::LifecycleReport { sleep: view.sleep_watch, screen_lock: view.screen_lock_watch }
    });
    let view = CapabilityView::from(capabilities, lifecycle, shared.desktop.session.label());
    shared.publish(|status| status.capabilities = Some(view));
}

fn open_stores_now(layout: &DataLayout) -> Option<Stores> {
    layout.ensure().ok()?;
    let keys = OsKeyring;
    Some(Stores {
        sessions: Mutex::new(EncryptedStore::open(&layout.sessions_database(), &keys).ok()?),
        queue: Mutex::new(EncryptedStore::open(&layout.queue_database(), &keys).ok()?),
        metadata: Mutex::new(EncryptedStore::open(&layout.metadata_database(), &keys).ok()?),
    })
}

/// The credential store may show an unlock prompt; startup does not wait on
/// it indefinitely. Without secure storage, dictation still works.
fn open_stores(layout: &DataLayout) -> Option<Stores> {
    let (sender, receiver) = channel();
    let layout = layout.clone();
    std::thread::spawn(move || {
        let _ = sender.send(open_stores_now(&layout));
    });
    receiver.recv_timeout(std::time::Duration::from_secs(20)).ok().flatten()
}

fn load_settings(stores: Option<&Stores>) -> AppSettings {
    stores
        .and_then(|stores| stores.metadata.lock().ok()?.get_setting("app_settings").ok()?)
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn setup(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let root = app.path().app_data_dir()?;
    let layout = DataLayout::new(root);
    let stores = open_stores(&layout);
    #[allow(unused_mut)]
    let mut settings = load_settings(stores.as_ref());
    // Debug builds only: apply a settings update from the environment so the
    // end-to-end harness can run unattended.
    #[cfg(debug_assertions)]
    if let Some(update) = std::env::var("LOCAL_DICTATION_DEV_SETTINGS")
        .ok()
        .and_then(|json| serde_json::from_str::<settings::SettingsUpdate>(&json).ok())
    {
        let _ = settings.apply(update);
    }
    let (commands, command_receiver) = channel();
    let (background_wake, wake_receiver) = channel();
    let shared = Arc::new(Shared {
        app: app.clone(),
        layout,
        storage_ready: stores.is_some(),
        stores,
        settings: Mutex::new(settings.clone()),
        status: Mutex::new(StatusView::default()),
        result: Mutex::new(None),
        desktop: DesktopServices::new()?,
        bus: EventBus::new(256),
        pairings: Arc::new(Pairings::default()),
        revocations: AtomicU64::new(0),
        preemption: Preemption::default(),
        commands,
        background_wake,
        portal_shortcuts_active: std::sync::atomic::AtomicBool::new(false),
        opted_out_sessions: Mutex::new(Vec::new()),
        api: Mutex::new(None),
    });
    app.manage(Arc::clone(&shared));
    let storage_ready = shared.storage_ready;
    let contribution = settings.contribution_target != settings::ContributionTarget::Disabled;
    shared.publish(|status| {
        status.storage_available = storage_ready;
        status.contribution_enabled = contribution;
        status.max_session_seconds = settings.max_session_seconds;
        if !storage_ready {
            status.notice = Some("secure_storage_unavailable".to_owned());
        }
    });

    // LOCAL_DICTATION_WORKER_DIR lets development builds use release workers.
    let executable_dir = std::env::var_os("LOCAL_DICTATION_WORKER_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::current_exe().ok()?.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_default();
    let workers = WorkerPaths::locate(&executable_dir);
    let verifier = Arc::new(ModelVerifier::default());
    let asr = asr_thread::spawn(Arc::clone(&shared), workers.clone(), Arc::clone(&verifier));
    app.manage(asr.clone());
    let controller = Controller::new(Arc::clone(&shared), asr.clone());
    std::thread::Builder::new()
        .name("controller".to_owned())
        .spawn(move || controller.run(&command_receiver))?;
    background::spawn(Arc::clone(&shared), wake_receiver, workers, verifier, asr.clone());

    let (lifecycle_sender, mut lifecycle_receiver) = tokio::sync::mpsc::unbounded_channel();
    let report = shared.desktop.start_lifecycle_watch(lifecycle_sender);
    let forward = Arc::clone(&shared);
    std::thread::spawn(move || {
        while let Some(event) = lifecycle_receiver.blocking_recv() {
            let _ = forward.commands.send(Command::Lifecycle(event));
        }
    });
    let capabilities = shared.desktop.capabilities(false);
    shared.publish(|status| status.capabilities = Some(CapabilityView::from(capabilities, report, shared.desktop.session.label())));

    indicator::create(app)?;
    tray::install(app);
    api_bridge::apply(&shared);
    // Portal and accessibility setup may wait for the user to answer a
    // system dialog, so they run off the UI thread.
    let integration = Arc::clone(&shared);
    let handle = app.clone();
    std::thread::Builder::new().name("integration-setup".to_owned()).spawn(move || {
        if settings.automatic_insertion {
            if let Err(error) = integration.desktop.enable_accessibility() {
                integration.publish(|status| status.notice = Some(format!("accessibility_unavailable:{error}")));
            }
        }
        if let Err(error) = shortcut_setup::register(&handle, &integration) {
            integration.publish(|status| {
                status.shortcuts_active = false;
                status.notice = Some(format!("shortcuts_unavailable:{error}"));
            });
        }
        refresh_capabilities(&integration);
    })?;
    if dictation_models::state(&shared.layout.models(), dictation_models::spec(&settings.asr_model).unwrap_or_else(|| dictation_models::default_for(dictation_models::Role::Asr)))
        == dictation_models::InstallState::Installed
    {
        asr.send(asr_thread::AsrRequest::Warm);
    }
    if !settings.onboarding_complete {
        show_main_window(app);
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
/// Starts the desktop application.
///
/// # Panics
///
/// Panics when Tauri cannot initialize the application shell.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| setup(app.handle()))
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    // Closing the window keeps dictation available from the tray.
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_status,
            commands::get_settings,
            commands::update_settings,
            commands::capability_matrix,
            commands::recording_action,
            commands::do_not_contribute,
            commands::get_result,
            commands::copy_result,
            commands::dismiss_result,
            commands::list_models,
            commands::install_model,
            commands::cancel_model_download,
            commands::import_model,
            commands::install_custom_model,
            commands::choose_model_file,
            commands::get_disclosure,
            commands::grant_consent,
            commands::set_contribution_paused,
            commands::withdraw_consent,
            commands::delete_training_data,
            commands::contribution_history,
            commands::list_pairings,
            commands::approve_pairing,
            commands::deny_pairing,
            commands::list_clients,
            commands::revoke_client,
            commands::retry_shortcuts,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build the Local Dictation shell")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(shared) = app.try_state::<Arc<Shared>>() {
                    let _ = shared.commands.send(Command::Shutdown);
                }
            }
        });
}
