//! Cross-platform desktop integration facade.
//!
//! Capabilities are probed and reported as observed. Focus is captured at
//! recording start and compared at delivery; insertion never happens into a
//! changed or unverifiable target (the core policy decides). Text is inserted
//! without newlines, so no Enter/submit is ever synthesized.

use std::sync::Mutex;

use dictation_core::platform::{Capability, FocusTarget, LifecycleEvent, PlatformCapabilities};
use tokio::sync::mpsc::UnboundedSender;

use crate::{probe_capabilities, shortcuts::{ShortcutBindings, ShortcutEvent}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    Wayland,
    X11,
    Windows,
    MacOs,
    Unknown,
}

impl SessionKind {
    #[must_use]
    pub fn detect() -> Self {
        if cfg!(target_os = "windows") {
            return Self::Windows;
        }
        if cfg!(target_os = "macos") {
            return Self::MacOs;
        }
        match std::env::var("XDG_SESSION_TYPE").as_deref() {
            Ok("wayland") => Self::Wayland,
            Ok("x11") => Self::X11,
            _ if std::env::var_os("WAYLAND_DISPLAY").is_some() => Self::Wayland,
            _ if std::env::var_os("DISPLAY").is_some() => Self::X11,
            _ => Self::Unknown,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Wayland => "linux-wayland",
            Self::X11 => "linux-x11",
            Self::Windows => "windows",
            Self::MacOs => "macos",
            Self::Unknown => "unknown",
        }
    }
}

/// Where focus was when recording started, in the terms each backend can
/// re-verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeFocus {
    #[cfg(target_os = "linux")]
    Accessible(crate::linux::accessibility::FocusedObject),
    Window { id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FocusSnapshot {
    pub target: FocusTarget,
    pub native: NativeFocus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertedVia {
    Accessibility,
    Keystrokes,
    PortalKeyboard,
}

#[derive(Debug)]
pub struct InsertError(pub String);

impl std::fmt::Display for InsertError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "insertion failed: {}", self.0)
    }
}

impl std::error::Error for InsertError {}

/// Removes characters that could submit or restructure input (newlines,
/// tabs, control characters). Dictation is inserted as one line of text.
#[must_use]
pub fn sanitize_for_insertion(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut last_space = false;
    for character in text.chars() {
        if character.is_control() || character.is_whitespace() {
            if !last_space && !output.is_empty() {
                output.push(' ');
            }
            last_space = true;
        } else {
            output.push(character);
            last_space = false;
        }
    }
    output.trim_end().to_owned()
}

/// Reverse-DNS application identifier shared with the Tauri bundle.
pub const APP_ID: &str = "app.localdictation.desktop";

pub struct DesktopServices {
    pub session: SessionKind,
    #[cfg(target_os = "linux")]
    runtime: tokio::runtime::Runtime,
    #[cfg(target_os = "linux")]
    accessibility: Mutex<Option<crate::linux::accessibility::Accessibility>>,
    #[cfg(target_os = "linux")]
    portal_keyboard: Mutex<Option<crate::linux::remote_desktop::PortalKeyboard>>,
    #[cfg(target_os = "linux")]
    portal_shortcuts: Mutex<Option<crate::linux::portal_shortcuts::PortalShortcuts>>,
    lifecycle: Mutex<crate::LifecycleReport>,
}

impl DesktopServices {
    /// # Errors
    ///
    /// Fails only when the Linux D-Bus runtime cannot be created.
    pub fn new() -> std::io::Result<Self> {
        let services = Self {
            session: SessionKind::detect(),
            #[cfg(target_os = "linux")]
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("desktop-integration")
                .enable_all()
                .build()?,
            #[cfg(target_os = "linux")]
            accessibility: Mutex::new(None),
            #[cfg(target_os = "linux")]
            portal_keyboard: Mutex::new(None),
            #[cfg(target_os = "linux")]
            portal_shortcuts: Mutex::new(None),
            lifecycle: Mutex::new(crate::LifecycleReport::default()),
        };
        #[cfg(target_os = "linux")]
        {
            // Unsandboxed apps identify themselves to xdg-desktop-portal
            // (>= 1.19) before any portal call; it must match an installed
            // `<APP_ID>.desktop` file.
            if let Ok(app_id) = ashpd::AppID::try_from(APP_ID) {
                let _ = services.runtime.block_on(ashpd::register_host_app(app_id));
            }
        }
        Ok(services)
    }

    /// Observed capabilities for the capability matrix and the core policy.
    #[must_use]
    pub fn capabilities(&self, portal_shortcuts_active: bool) -> PlatformCapabilities {
        let mut capabilities = probe_capabilities();
        capabilities.clipboard = if arboard::Clipboard::new().is_ok() {
            Capability::Available
        } else {
            Capability::Unavailable
        };
        match self.session {
            SessionKind::Wayland => {
                #[cfg(target_os = "linux")]
                {
                    let accessible = self.accessibility.lock().is_ok_and(|slot| slot.is_some());
                    capabilities.focus_tracking = if accessible { Capability::Available } else { Capability::PermissionRequired };
                    capabilities.native_insertion = if accessible { Capability::Available } else { Capability::PermissionRequired };
                }
                capabilities.global_shortcut = if portal_shortcuts_active { Capability::Available } else { Capability::PermissionRequired };
                // Measured on KWin: a regular overlay window takes focus, so
                // it is not shown; the tray (StatusNotifierItem) indicates.
                capabilities.nonactivating_indicator = Capability::Unavailable;
            }
            SessionKind::X11 | SessionKind::Windows | SessionKind::MacOs => {
                capabilities.focus_tracking = Capability::Available;
                capabilities.native_insertion = Capability::Available;
                capabilities.global_shortcut = Capability::Available;
                capabilities.nonactivating_indicator = Capability::Available;
            }
            SessionKind::Unknown => {}
        }
        capabilities
    }

    /// Enables toolkit accessibility and focus tracking (Linux). Called only
    /// after the user turns on automatic insertion.
    ///
    /// # Errors
    ///
    /// Fails when the accessibility bus is unavailable.
    pub fn enable_accessibility(&self) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            let service = self
                .runtime
                .block_on(crate::linux::accessibility::Accessibility::start())
                .map_err(|error| error.to_string())?;
            if let Ok(mut slot) = self.accessibility.lock() {
                *slot = Some(service);
            }
        }
        Ok(())
    }

    /// Registers portal shortcuts (Wayland). Other sessions use the Tauri
    /// global-shortcut plugin in the desktop shell.
    ///
    /// # Errors
    ///
    /// Fails when the portal is unavailable or the user declines.
    pub fn start_portal_shortcuts(
        &self,
        bindings: &ShortcutBindings,
        sender: UnboundedSender<ShortcutEvent>,
    ) -> Result<Vec<(String, String)>, String> {
        #[cfg(target_os = "linux")]
        {
            let shortcuts = self
                .runtime
                .block_on(crate::linux::portal_shortcuts::start(bindings, sender))?;
            let bound = shortcuts.bound.clone();
            if let Ok(mut slot) = self.portal_shortcuts.lock() {
                *slot = Some(shortcuts);
            }
            Ok(bound)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (bindings, sender);
            Err("portal_shortcuts_linux_only".to_owned())
        }
    }

    /// Watches for sleep and screen lock.
    pub fn start_lifecycle_watch(&self, sender: UnboundedSender<LifecycleEvent>) -> crate::LifecycleReport {
        #[cfg(target_os = "linux")]
        let report = {
            let watch = self.runtime.block_on(crate::linux::session_watch::start(sender));
            crate::LifecycleReport {
                sleep: watch.sleep,
                screen_lock: watch.screen_lock,
            }
        };
        #[cfg(not(target_os = "linux"))]
        let report = {
            let _ = sender;
            crate::LifecycleReport::default()
        };
        if let Ok(mut slot) = self.lifecycle.lock() {
            *slot = report;
        }
        report
    }

    /// Current focus in re-verifiable terms, when the platform reveals it.
    #[must_use]
    pub fn focus(&self) -> Option<FocusSnapshot> {
        #[cfg(target_os = "linux")]
        {
            if let Some(object) = self
                .accessibility
                .lock()
                .ok()
                .and_then(|slot| slot.as_ref().and_then(crate::linux::accessibility::Accessibility::focused))
            {
                return Some(FocusSnapshot {
                    target: object.target(),
                    native: NativeFocus::Accessible(object),
                });
            }
            if self.session == SessionKind::X11 {
                return crate::linux::x11::active_window().map(|(id, class)| FocusSnapshot {
                    target: FocusTarget {
                        application_id: class,
                        window_id: id.clone(),
                        editable_id: None,
                        protected: false,
                    },
                    native: NativeFocus::Window { id },
                });
            }
            None
        }
        #[cfg(not(target_os = "linux"))]
        {
            let window = active_win_pos_rs::get_active_window().ok()?;
            let id = window.window_id.clone();
            Some(FocusSnapshot {
                target: FocusTarget {
                    application_id: window.process_path.to_string_lossy().into_owned(),
                    window_id: id.clone(),
                    editable_id: None,
                    protected: false,
                },
                native: NativeFocus::Window { id },
            })
        }
    }

    /// Inserts into the verified target. The caller has already confirmed via
    /// the core policy that focus is unchanged and the field is not excluded.
    ///
    /// # Errors
    ///
    /// Fails without partial typing when the backend cannot insert safely.
    pub fn insert(&self, snapshot: &FocusSnapshot, text: &str) -> Result<InsertedVia, InsertError> {
        let text = sanitize_for_insertion(text);
        if text.is_empty() {
            return Err(InsertError("empty_text".to_owned()));
        }
        match &snapshot.native {
            #[cfg(target_os = "linux")]
            NativeFocus::Accessible(object) => {
                let guard = self.accessibility.lock().map_err(|_| InsertError("lock".to_owned()))?;
                let service = guard.as_ref().ok_or_else(|| InsertError("accessibility_disabled".to_owned()))?;
                match self.runtime.block_on(service.insert(object, &text)) {
                    Ok(()) => Ok(InsertedVia::Accessibility),
                    Err(crate::linux::accessibility::AccessibilityError::NotEditable | crate::linux::accessibility::AccessibilityError::Refused)
                        if self.session == SessionKind::Wayland =>
                    {
                        // The widget has focus but no editable-text support
                        // (common in terminals): consent-based typing.
                        drop(guard);
                        self.type_via_portal(&text).map(|()| InsertedVia::PortalKeyboard)
                    }
                    Err(crate::linux::accessibility::AccessibilityError::NotEditable | crate::linux::accessibility::AccessibilityError::Refused)
                        if self.session == SessionKind::X11 =>
                    {
                        type_with_enigo(&text).map(|()| InsertedVia::Keystrokes)
                    }
                    Err(error) => Err(InsertError(error.to_string())),
                }
            }
            NativeFocus::Window { .. } => type_with_enigo(&text).map(|()| InsertedVia::Keystrokes),
        }
    }

    #[cfg(target_os = "linux")]
    fn type_via_portal(&self, text: &str) -> Result<(), InsertError> {
        let mut slot = self.portal_keyboard.lock().map_err(|_| InsertError("lock".to_owned()))?;
        if slot.is_none() {
            let keyboard = self
                .runtime
                .block_on(crate::linux::remote_desktop::PortalKeyboard::open(None))
                .map_err(InsertError)?;
            *slot = Some(keyboard);
        }
        let keyboard = slot.as_ref().ok_or_else(|| InsertError("portal_keyboard".to_owned()))?;
        self.runtime.block_on(keyboard.type_text(text)).map_err(InsertError)
    }
}

fn type_with_enigo(text: &str) -> Result<(), InsertError> {
    use enigo::{Enigo, Keyboard, Settings};
    let mut enigo = Enigo::new(&Settings::default()).map_err(|error| InsertError(error.to_string()))?;
    enigo.text(text).map_err(|error| InsertError(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_text_never_contains_submit_characters() {
        assert_eq!(sanitize_for_insertion("hello\nworld\r\n"), "hello world");
        assert_eq!(sanitize_for_insertion("a\tb  c"), "a b c");
        assert_eq!(sanitize_for_insertion("\n\n"), "");
    }
}
