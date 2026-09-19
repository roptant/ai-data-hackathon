//! Sleep and screen-lock notifications (logind and the freedesktop
//! ScreenSaver interface). Either one cancels an active recording without
//! delivery (plan §5).

use dictation_core::platform::LifecycleEvent;
use futures_util::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

#[zbus::proxy(
    interface = "org.freedesktop.login1.Manager",
    default_service = "org.freedesktop.login1",
    default_path = "/org/freedesktop/login1"
)]
trait LoginManager {
    #[zbus(signal)]
    fn prepare_for_sleep(&self, start: bool) -> zbus::Result<()>;
}

#[zbus::proxy(
    interface = "org.freedesktop.ScreenSaver",
    default_service = "org.freedesktop.ScreenSaver",
    default_path = "/org/freedesktop/ScreenSaver"
)]
trait ScreenSaver {
    #[zbus(signal)]
    fn active_changed(&self, active: bool) -> zbus::Result<()>;
}

/// Which watchers are active; unavailable ones are reported, not assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WatchReport {
    pub sleep: bool,
    pub screen_lock: bool,
}

/// Starts both watchers on the current runtime.
pub async fn start(sender: UnboundedSender<LifecycleEvent>) -> WatchReport {
    let mut report = WatchReport::default();
    if let Ok(system) = zbus::Connection::system().await {
        if let Ok(manager) = LoginManagerProxy::new(&system).await {
            if let Ok(mut signals) = manager.receive_prepare_for_sleep().await {
                report.sleep = true;
                let sleep_sender = sender.clone();
                tokio::spawn(async move {
                    while let Some(signal) = signals.next().await {
                        if signal.args().is_ok_and(|args| args.start) {
                            let _ = sleep_sender.send(LifecycleEvent::SystemSleeping);
                        }
                    }
                });
            }
        }
    }
    if let Ok(session) = zbus::Connection::session().await {
        if let Ok(saver) = ScreenSaverProxy::new(&session).await {
            if let Ok(mut signals) = saver.receive_active_changed().await {
                report.screen_lock = true;
                tokio::spawn(async move {
                    while let Some(signal) = signals.next().await {
                        if signal.args().is_ok_and(|args| args.active) {
                            let _ = sender.send(LifecycleEvent::ScreenLocked);
                        }
                    }
                });
            }
        }
    }
    report
}
