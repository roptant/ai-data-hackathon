//! Shared application state.
//!
//! The `WebView` never receives storage handles, audio, or the server token.
//! Transcript text crosses to the UI only for the local result panel and the
//! overlay's live caption, both of which the user explicitly sees.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::Sender,
};

use dictation_api::{EventBus, Pairings};
use dictation_core::platform::{Capability, PlatformCapabilities};
use dictation_engine::workers::Preemption;
use dictation_platform::desktop::DesktopServices;
use dictation_storage::{EncryptedStore, layout::DataLayout};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::{controller::Command, settings::AppSettings};

/// The three encrypted databases. `None` when secure key storage is
/// unavailable: dictation still works, persistent contribution does not.
pub struct Stores {
    pub sessions: Mutex<EncryptedStore>,
    pub queue: Mutex<EncryptedStore>,
    pub metadata: Mutex<EncryptedStore>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityView {
    pub microphone: &'static str,
    pub global_shortcut: &'static str,
    pub focus_tracking: &'static str,
    pub native_insertion: &'static str,
    pub clipboard: &'static str,
    pub nonactivating_indicator: &'static str,
    pub sleep_watch: bool,
    pub screen_lock_watch: bool,
    pub session: &'static str,
}

#[must_use]
pub const fn capability_name(capability: Capability) -> &'static str {
    match capability {
        Capability::Available => "available",
        Capability::PermissionRequired => "needs_permission",
        Capability::Unavailable => "unavailable",
    }
}

impl CapabilityView {
    #[must_use]
    pub fn from(
        capabilities: PlatformCapabilities,
        lifecycle: dictation_platform::LifecycleReport,
        session: &'static str,
    ) -> Self {
        Self {
            microphone: capability_name(capabilities.microphone),
            global_shortcut: capability_name(capabilities.global_shortcut),
            focus_tracking: capability_name(capabilities.focus_tracking),
            native_insertion: capability_name(capabilities.native_insertion),
            clipboard: capability_name(capabilities.clipboard),
            nonactivating_indicator: capability_name(capabilities.nonactivating_indicator),
            sleep_watch: lifecycle.sleep,
            screen_lock_watch: lifecycle.screen_lock,
            session,
        }
    }
}

/// Coarse state for the UI, indicator, tray, and API. No transcript text.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StatusView {
    pub recording_state: &'static str,
    pub session_id: Option<String>,
    pub elapsed_ms: u64,
    pub max_session_seconds: u64,
    pub locked: bool,
    /// Short content-free notice, e.g. `microphone_failure`.
    pub notice: Option<String>,
    pub asr_ready: bool,
    pub asr_loading: bool,
    pub privacy_model_ready: bool,
    pub storage_available: bool,
    pub contribution_enabled: bool,
    pub training_status: String,
    pub api_listening: Option<u16>,
    pub shortcuts_active: bool,
    pub shortcut_triggers: Vec<(String, String)>,
    pub capabilities: Option<CapabilityView>,
}

impl Default for StatusView {
    fn default() -> Self {
        Self {
            recording_state: "idle",
            session_id: None,
            elapsed_ms: 0,
            max_session_seconds: 600,
            locked: false,
            notice: None,
            asr_ready: false,
            asr_loading: false,
            privacy_model_ready: false,
            storage_available: false,
            contribution_enabled: false,
            training_status: "idle".to_owned(),
            api_listening: None,
            shortcuts_active: false,
            shortcut_triggers: Vec::new(),
            capabilities: None,
        }
    }
}

/// An undelivered result kept only in memory until copied or dismissed.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ResultPanel {
    pub session_id: String,
    pub text: String,
    /// Why it was not inserted automatically (content-free code).
    pub reason: String,
}

pub struct Shared {
    pub app: AppHandle,
    pub layout: DataLayout,
    pub storage_ready: bool,
    pub stores: Option<Stores>,
    pub settings: Mutex<AppSettings>,
    pub status: Mutex<StatusView>,
    pub result: Mutex<Option<ResultPanel>>,
    pub desktop: DesktopServices,
    pub bus: EventBus,
    pub pairings: Arc<Pairings>,
    pub revocations: AtomicU64,
    pub preemption: Preemption,
    pub commands: Sender<Command>,
    pub background_wake: Sender<()>,
    pub portal_shortcuts_active: AtomicBool,
    /// Session IDs the user marked "do not contribute".
    pub opted_out_sessions: Mutex<Vec<String>>,
    pub api: Mutex<Option<dictation_api::ApiHandle>>,
}

impl Shared {
    /// Publishes status to every window and the tray.
    pub fn publish(&self, update: impl FnOnce(&mut StatusView)) {
        let snapshot = {
            let Ok(mut status) = self.status.lock() else { return };
            update(&mut status);
            status.clone()
        };
        if std::env::var_os("LOCAL_DICTATION_DEBUG").is_some() {
            // Status never contains transcript text, so this trace is safe.
            eprintln!("status {}", serde_json::to_string(&snapshot).unwrap_or_default());
        }
        let _ = self.app.emit("status", &snapshot);
        crate::tray::refresh(&self.app, &snapshot);
        let active = matches!(snapshot.recording_state, "starting" | "recording" | "recording_locked" | "finalizing");
        crate::shortcut_setup::sync_cancel(&self.app, self, active);
    }

    #[must_use]
    pub fn status(&self) -> StatusView {
        self.status.lock().map(|status| status.clone()).unwrap_or_default()
    }

    #[must_use]
    pub fn settings(&self) -> AppSettings {
        self.settings.lock().map(|settings| settings.clone()).unwrap_or_default()
    }

    /// Persists settings encrypted; without secure storage they stay in memory.
    pub fn save_settings(&self) {
        let (Some(stores), Ok(settings)) = (&self.stores, self.settings.lock()) else {
            return;
        };
        if let (Ok(bytes), Ok(metadata)) = (serde_json::to_vec(&*settings), stores.metadata.lock()) {
            let _ = metadata.put_setting("app_settings", &bytes, crate::unix_now());
        }
    }

    pub fn bump_revocations(&self) {
        self.revocations.fetch_add(1, Ordering::SeqCst);
    }
}
