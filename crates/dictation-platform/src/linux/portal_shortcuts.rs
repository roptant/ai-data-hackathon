//! Global shortcuts through the xdg-desktop-portal `GlobalShortcuts` interface
//! (Wayland). The compositor owns the key grab; users remap the triggers in
//! their desktop settings. `Activated`/`Deactivated` give press and release,
//! which hold-to-record needs.

use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
use futures_util::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::shortcuts::{ShortcutAction, ShortcutBindings, ShortcutEvent};

/// Keeps the portal session alive; dropping it releases the shortcuts.
pub struct PortalShortcuts {
    _tasks: Vec<tokio::task::JoinHandle<()>>,
    pub bound: Vec<(String, String)>,
}

fn action_for(id: &str) -> Option<ShortcutAction> {
    match id {
        "hold" => Some(ShortcutAction::Hold),
        "toggle" => Some(ShortcutAction::Toggle),
        "lock" => Some(ShortcutAction::Lock),
        "cancel" => Some(ShortcutAction::Cancel),
        _ => None,
    }
}

/// Registers the four actions and forwards press/release events.
///
/// # Errors
///
/// Fails when the portal is missing or the user declines the binding.
pub async fn start(
    bindings: &ShortcutBindings,
    sender: UnboundedSender<ShortcutEvent>,
) -> Result<PortalShortcuts, String> {
    let portal = GlobalShortcuts::new().await.map_err(|error| error.to_string())?;
    let session = portal
        .create_session(Default::default())
        .await
        .map_err(|error| error.to_string())?;
    let shortcuts = [
        NewShortcut::new("hold", "Hold to dictate").preferred_trigger(Some(bindings.hold.portal_trigger().as_str())),
        NewShortcut::new("toggle", "Start or stop dictation").preferred_trigger(Some(bindings.toggle.portal_trigger().as_str())),
        NewShortcut::new("lock", "Keep recording after releasing the hold key").preferred_trigger(Some(bindings.lock.portal_trigger().as_str())),
        NewShortcut::new("cancel", "Cancel dictation without inserting").preferred_trigger(Some(bindings.cancel_portal.portal_trigger().as_str())),
    ];
    let response = portal
        .bind_shortcuts(&session, &shortcuts, None, Default::default())
        .await
        .map_err(|error| error.to_string())?
        .response()
        .map_err(|error| error.to_string())?;
    let bound = response
        .shortcuts()
        .iter()
        .map(|shortcut| (shortcut.id().to_owned(), shortcut.trigger_description().to_owned()))
        .collect();
    let mut activated = portal.receive_activated().await.map_err(|error| error.to_string())?;
    let mut deactivated = portal.receive_deactivated().await.map_err(|error| error.to_string())?;
    let press = sender.clone();
    let keep_session = tokio::spawn(async move {
        let _portal = portal;
        let _session = session;
        std::future::pending::<()>().await;
    });
    let pressed = tokio::spawn(async move {
        while let Some(event) = activated.next().await {
            if let Some(action) = action_for(event.shortcut_id()) {
                let _ = press.send(ShortcutEvent { action, pressed: true });
            }
        }
    });
    let released = tokio::spawn(async move {
        while let Some(event) = deactivated.next().await {
            if let Some(action) = action_for(event.shortcut_id()) {
                let _ = sender.send(ShortcutEvent { action, pressed: false });
            }
        }
    });
    Ok(PortalShortcuts {
        _tasks: vec![keep_session, pressed, released],
        bound,
    })
}
