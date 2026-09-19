//! Delivery of the full dictation to the target the user started in.
//!
//! The core policy decides; this module executes. Anything not delivered
//! automatically is kept in memory for the local result panel, where the user
//! copies it and chooses where to paste. No Enter key is ever synthesized.

use std::{collections::BTreeSet, time::Duration};

use dictation_core::platform::{DeliveryMethod, DeliveryReason, InsertionPolicy, choose_delivery};
use dictation_platform::{clipboard, desktop::FocusSnapshot};
use tauri::Emitter;

use crate::state::{ResultPanel, Shared};

const fn reason_code(reason: DeliveryReason) -> &'static str {
    match reason {
        DeliveryReason::FocusUnverifiable => "focus_unverifiable",
        DeliveryReason::FocusChanged => "focus_changed",
        DeliveryReason::ProtectedField => "protected_field",
        DeliveryReason::ApplicationExcluded => "application_excluded",
        DeliveryReason::NativeInsertionAvailable => "inserted",
        DeliveryReason::DisclosedClipboardFallback => "clipboard",
        DeliveryReason::NoSafeAutomaticMethod => "no_automatic_method",
    }
}

fn to_panel(shared: &Shared, session_id: &str, text: &str, reason: &str) {
    if let Ok(mut slot) = shared.result.lock() {
        *slot = Some(ResultPanel {
            session_id: session_id.to_owned(),
            text: text.to_owned(),
            reason: reason.to_owned(),
        });
    }
    // The overlay and tray announce the result; the main window is not
    // raised automatically so focus is never taken from the user's app.
    let _ = shared.app.emit("result-ready", reason);
    shared.publish(|status| status.notice = Some(format!("result_ready:{reason}")));
}

pub fn deliver(shared: &Shared, session_id: &str, original: Option<&FocusSnapshot>, text: &str) {
    let settings = shared.settings();
    if !settings.automatic_insertion {
        to_panel(shared, session_id, text, "automatic_insertion_off");
        return;
    }
    let capabilities = shared
        .desktop
        .capabilities(shared.portal_shortcuts_active.load(std::sync::atomic::Ordering::SeqCst));
    let current = shared.desktop.focus();
    let (Some(original), Some(current)) = (original, current.as_ref()) else {
        to_panel(shared, session_id, text, reason_code(DeliveryReason::FocusUnverifiable));
        return;
    };
    let policy = InsertionPolicy {
        excluded_applications: settings.excluded_applications.iter().cloned().collect::<BTreeSet<_>>(),
        clipboard_fallback_disclosed: settings.clipboard_fallback_disclosed,
    };
    let decision = choose_delivery(&original.target, &current.target, capabilities, &policy);
    match decision.method {
        DeliveryMethod::NativeInsertion => match shared.desktop.insert(original, text) {
            Ok(_) => shared.publish(|status| status.notice = Some("inserted".to_owned())),
            Err(error) => {
                // Insertion errors are content-free codes.
                if std::env::var_os("LOCAL_DICTATION_DEBUG").is_some() {
                    eprintln!("insertion {error}");
                }
                if settings.clipboard_fallback_disclosed {
                    clipboard_fallback(shared, session_id, text);
                } else {
                    to_panel(shared, session_id, text, "insertion_failed");
                }
            }
        },
        DeliveryMethod::ClipboardFallback => clipboard_fallback(shared, session_id, text),
        DeliveryMethod::ResultPanel => to_panel(shared, session_id, text, reason_code(decision.reason)),
    }
}

fn clipboard_fallback(shared: &Shared, session_id: &str, text: &str) {
    match clipboard::place(text) {
        Ok(lease) => {
            // Restore the user's previous clipboard after a minute, but only
            // if it still holds the dictation.
            lease.restore_later(Duration::from_secs(60));
            shared.publish(|status| status.notice = Some("copied_to_clipboard".to_owned()));
        }
        Err(_) => to_panel(shared, session_id, text, "clipboard_unavailable"),
    }
}
