//! Minimal-permission Tauri shell for the Rust migration.
//!
//! Capture, storage, inference, and upload are intentionally not exposed to the
//! `WebView`. New capabilities are added only when their Rust implementation and
//! permission boundary are ready for review.

use std::sync::Mutex;

use dictation_core::{
    platform::Capability,
    recording::{RecordingStateMachine, State as RecordingState},
};
use dictation_platform::probe_capabilities;
use serde::Serialize;
use tauri::State;

struct DesktopState {
    recording: Mutex<RecordingStateMachine>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShellStatus {
    app_version: &'static str,
    migration_phase: &'static str,
    recording_state: &'static str,
    components: ComponentStatus,
    contribution_enabled: bool,
    notice: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComponentStatus {
    microphone_ready: bool,
    asr_ready: bool,
    privacy_model_ready: bool,
}

fn recording_state_name(state: RecordingState) -> &'static str {
    match state {
        RecordingState::Idle => "idle",
        RecordingState::Starting => "starting",
        RecordingState::RecordingHeld => "recording_held",
        RecordingState::RecordingLocked => "recording_locked",
        RecordingState::Finalizing => "finalizing",
        RecordingState::Delivering => "delivering",
        RecordingState::Cancelled => "cancelled",
        RecordingState::Error => "error",
    }
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)] // Tauri's command extractor requires owned `State`.
fn shell_status(state: State<'_, DesktopState>) -> Result<ShellStatus, String> {
    let recording = state
        .recording
        .lock()
        .map_err(|_| "recording_state_unavailable".to_owned())?;
    let capabilities = probe_capabilities();
    Ok(ShellStatus {
        app_version: env!("CARGO_PKG_VERSION"),
        migration_phase: "platform_capture",
        recording_state: recording_state_name(recording.state()),
        components: ComponentStatus {
            microphone_ready: capabilities.microphone == Capability::Available,
            asr_ready: false,
            privacy_model_ready: false,
        },
        contribution_enabled: false,
        notice: "Microphone probing is live; ASR, delivery, and uploads remain disabled.",
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
/// Starts the desktop application.
///
/// # Panics
///
/// Panics when Tauri cannot initialize or run the application shell.
pub fn run() {
    tauri::Builder::default()
        .manage(DesktopState {
            recording: Mutex::new(RecordingStateMachine::default()),
        })
        .invoke_handler(tauri::generate_handler![shell_status])
        .run(tauri::generate_context!())
        .expect("failed to run the Local Dictation desktop shell");
}
