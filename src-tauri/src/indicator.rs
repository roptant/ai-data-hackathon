//! The unobtrusive recording overlay (plan §5).
//!
//! A small undecorated, always-on-top window that is created unfocused and
//! non-focusable so it never takes focus from the target application. It
//! shows state as text and an icon, not color alone, plus elapsed time, and
//! announces changes through an ARIA live region. Wayland compositors may
//! ignore stacking or placement hints; the tray remains the authoritative
//! indicator there and the capability matrix says so.

use dictation_core::recording::State;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};

pub const LABEL: &str = "indicator";

pub fn create(app: &AppHandle) -> tauri::Result<()> {
    if app.get_webview_window(LABEL).is_some() {
        return Ok(());
    }
    let builder = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::App("indicator.html".into()))
        .title("Local Dictation indicator")
        .inner_size(300.0, 64.0)
        .resizable(false)
        .decorations(false);
    // Transparent WKWebView windows require Tauri's `macos-private-api`
    // feature, which is unsuitable for an App Store-capable build. A solid,
    // nonactivating indicator is safer there.
    #[cfg(not(target_os = "macos"))]
    let builder = builder.transparent(true);
    let window = builder
        .always_on_top(true)
        .visible_on_all_workspaces(true)
        .skip_taskbar(true)
        .shadow(false)
        .focused(false)
        .focusable(false)
        .visible(false)
        .build()?;
    if let Ok(Some(monitor)) = window.primary_monitor() {
        let size = monitor.size();
        let scale = monitor.scale_factor();
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let x = (f64::from(size.width) / scale / 2.0 - 150.0) as i32;
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let y = (f64::from(size.height) / scale - 120.0) as i32;
        let _ = window.set_position(tauri::LogicalPosition::new(x, y));
    }
    Ok(())
}

/// Shows the overlay while a session is active and hides it afterwards.
/// `show()` on this window never requests focus.
pub fn update(app: &AppHandle, state: State) {
    // Measured on KWin/Wayland: compositors activate regular toplevel windows
    // regardless of the focusable hint, which would move focus away from the
    // target field. There the tray is the indicator until a layer-shell
    // overlay is available.
    if dictation_platform::desktop::SessionKind::detect() == dictation_platform::desktop::SessionKind::Wayland
        || std::env::var_os("LOCAL_DICTATION_NO_OVERLAY").is_some()
    {
        return;
    }
    let Some(window) = app.get_webview_window(LABEL) else { return };
    let visible = matches!(
        state,
        State::Starting | State::RecordingHeld | State::RecordingLocked | State::Finalizing | State::Delivering | State::Error
    );
    let _ = if visible { window.show() } else { window.hide() };
}
