//! Shortcut registration per platform.
//!
//! Wayland: the GlobalShortcuts portal (the compositor owns the grab and the
//! user remaps triggers in system settings). X11, Windows, macOS: the Tauri
//! global-shortcut plugin with press/release states. Escape is registered
//! only while recording so it is never taken from other applications
//! otherwise.

use std::sync::{Arc, atomic::Ordering};

use dictation_core::shortcuts::{Chord, ShortcutAction, ShortcutEvent};
use dictation_platform::desktop::SessionKind;
use tauri::AppHandle;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use crate::{controller::Command, state::Shared};

/// `global-hotkey` spelling of a portable chord.
fn plugin_accelerator(chord: &Chord) -> String {
    let mut parts = Vec::new();
    if chord.modifiers.control {
        parts.push("Control".to_owned());
    }
    if chord.modifiers.alt {
        parts.push("Alt".to_owned());
    }
    if chord.modifiers.shift {
        parts.push("Shift".to_owned());
    }
    if chord.modifiers.super_key {
        parts.push("Super".to_owned());
    }
    let key = match chord.key.as_str() {
        "Return" => "Enter".to_owned(),
        "BackSpace" => "Backspace".to_owned(),
        key if key.len() == 1 && key.chars().all(|c| c.is_ascii_alphabetic()) => format!("Key{key}"),
        key if key.len() == 1 && key.chars().all(|c| c.is_ascii_digit()) => format!("Digit{key}"),
        key => key.to_owned(),
    };
    parts.push(key);
    parts.join("+")
}

fn forward(shared: &Shared, action: ShortcutAction, pressed: bool) {
    let _ = shared.commands.send(Command::Shortcut(ShortcutEvent { action, pressed }));
}

/// Registers the configured shortcuts, replacing any previous registration.
pub fn register(app: &AppHandle, shared: &Arc<Shared>) -> Result<(), String> {
    let bindings = shared.settings().shortcuts.bindings()?;
    if shared.desktop.session == SessionKind::Wayland {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let bound = shared.desktop.start_portal_shortcuts(&bindings, sender)?;
        shared.portal_shortcuts_active.store(true, Ordering::SeqCst);
        let forwarder = Arc::clone(shared);
        std::thread::Builder::new()
            .name("portal-shortcuts".to_owned())
            .spawn(move || {
                while let Some(event) = receiver.blocking_recv() {
                    forward(&forwarder, event.action, event.pressed);
                }
            })
            .map_err(|error| error.to_string())?;
        shared.publish(|status| {
            status.shortcuts_active = true;
            status.shortcut_triggers = bound;
        });
        return Ok(());
    }
    let manager = app.global_shortcut();
    let _ = manager.unregister_all();
    let mut triggers = Vec::new();
    for (action, chord) in [
        (ShortcutAction::Hold, &bindings.hold),
        (ShortcutAction::Toggle, &bindings.toggle),
        (ShortcutAction::Lock, &bindings.lock),
    ] {
        let accelerator = plugin_accelerator(chord);
        let shortcut: Shortcut = accelerator.parse().map_err(|error| format!("{accelerator}: {error}"))?;
        let target = Arc::clone(shared);
        manager
            .on_shortcut(shortcut, move |_, _, event| {
                forward(&target, action, event.state() == ShortcutState::Pressed);
            })
            .map_err(|error| format!("{} is unavailable (already used by another app?): {error}", chord.display()))?;
        triggers.push((format!("{action:?}").to_lowercase(), chord.display()));
    }
    shared.publish(|status| {
        status.shortcuts_active = true;
        status.shortcut_triggers = triggers;
    });
    Ok(())
}

/// Grabs Escape only while a session is active (non-portal platforms).
pub fn sync_cancel(app: &AppHandle, shared: &Shared, active: bool) {
    if shared.desktop.session == SessionKind::Wayland {
        return;
    }
    let Ok(bindings) = shared.settings().shortcuts.bindings() else { return };
    let Ok(shortcut) = plugin_accelerator(&bindings.cancel).parse::<Shortcut>() else { return };
    let manager = app.global_shortcut();
    let registered = manager.is_registered(shortcut);
    if active && !registered {
        let commands = shared.commands.clone();
        let _ = manager.on_shortcut(shortcut, move |_, _, event| {
            let pressed = event.state() == ShortcutState::Pressed;
            let _ = commands.send(Command::Shortcut(ShortcutEvent { action: ShortcutAction::Cancel, pressed }));
        });
    } else if !active && registered {
        let _ = manager.unregister(shortcut);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chords_map_to_plugin_accelerators() {
        assert_eq!(plugin_accelerator(&Chord::parse("Ctrl+Alt+Space").unwrap()), "Control+Alt+Space");
        assert_eq!(plugin_accelerator(&Chord::parse("Ctrl+Alt+C").unwrap()), "Control+Alt+KeyC");
        assert_eq!(plugin_accelerator(&Chord::parse("Escape").unwrap()), "Escape");
        for text in ["Ctrl+Alt+Space", "Ctrl+Alt+Period", "Ctrl+Alt+Shift+Space", "Escape", "Ctrl+Alt+C"] {
            let accelerator = plugin_accelerator(&Chord::parse(text).unwrap());
            assert!(accelerator.parse::<Shortcut>().is_ok(), "{accelerator}");
        }
    }
}
