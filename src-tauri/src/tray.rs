//! Tray indicator with text status (not color alone) and controls.
//!
//! Linux uses a StatusNotifierItem (KDE, and GNOME with the AppIndicator
//! extension) implemented over D-Bus; Windows and macOS use Tauri's tray.

use std::sync::{Mutex, OnceLock};

use tauri::AppHandle;

use crate::state::StatusView;

#[must_use]
pub fn describe(status: &StatusView) -> String {
    let state = match status.recording_state {
        "recording" => "Recording (hold)",
        "recording_locked" => "Recording (locked)",
        "starting" => "Starting…",
        "finalizing" | "delivering" => "Finishing…",
        "error" => "Error",
        "cancelled" => "Cancelled",
        _ => "Idle",
    };
    match &status.notice {
        Some(notice) if status.recording_state == "error" || status.recording_state == "idle" => {
            format!("Local Dictation — {state} ({})", notice.replace('_', " "))
        }
        _ => format!("Local Dictation — {state}"),
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use ksni::TrayMethods;
    use tauri::{AppHandle, Manager};

    use crate::controller::{Command, UiAction};

    pub struct Tray {
        pub app: AppHandle,
        pub title: String,
        pub recording: bool,
    }

    fn send(app: &AppHandle, action: UiAction) {
        if let Some(shared) = app.try_state::<std::sync::Arc<crate::state::Shared>>() {
            let _ = shared.commands.send(Command::Ui(action));
        }
    }

    impl ksni::Tray for Tray {
        fn id(&self) -> String {
            "app.localdictation.desktop".to_owned()
        }

        fn title(&self) -> String {
            self.title.clone()
        }

        fn icon_name(&self) -> String {
            if self.recording { "media-record".to_owned() } else { "audio-input-microphone".to_owned() }
        }

        fn tool_tip(&self) -> ksni::ToolTip {
            ksni::ToolTip {
                title: self.title.clone(),
                ..Default::default()
            }
        }

        fn status(&self) -> ksni::Status {
            if self.recording { ksni::Status::NeedsAttention } else { ksni::Status::Active }
        }

        fn activate(&mut self, _x: i32, _y: i32) {
            crate::show_main_window(&self.app);
        }

        fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
            use ksni::menu::StandardItem;
            vec![
                StandardItem { label: self.title.clone(), enabled: false, ..Default::default() }.into(),
                ksni::MenuItem::Separator,
                StandardItem {
                    label: if self.recording { "Stop and insert".to_owned() } else { "Start dictation".to_owned() },
                    activate: Box::new(|tray: &mut Self| send(&tray.app, if tray.recording { UiAction::Stop } else { UiAction::Toggle })),
                    ..Default::default()
                }
                .into(),
                StandardItem {
                    label: "Cancel dictation".to_owned(),
                    enabled: self.recording,
                    activate: Box::new(|tray: &mut Self| send(&tray.app, UiAction::Cancel)),
                    ..Default::default()
                }
                .into(),
                ksni::MenuItem::Separator,
                StandardItem {
                    label: "Open Local Dictation".to_owned(),
                    activate: Box::new(|tray: &mut Self| crate::show_main_window(&tray.app)),
                    ..Default::default()
                }
                .into(),
                StandardItem {
                    label: "Quit".to_owned(),
                    activate: Box::new(|tray: &mut Self| tray.app.exit(0)),
                    ..Default::default()
                }
                .into(),
            ]
        }
    }

    pub type Handle = ksni::Handle<Tray>;

    pub fn spawn(app: &AppHandle) -> Option<Handle> {
        let tray = Tray { app: app.clone(), title: "Local Dictation — Idle".to_owned(), recording: false };
        tauri::async_runtime::block_on(tray.spawn()).ok()
    }
}

#[cfg(target_os = "linux")]
static HANDLE: OnceLock<Mutex<Option<platform::Handle>>> = OnceLock::new();

#[cfg(not(target_os = "linux"))]
static HANDLE: OnceLock<Mutex<Option<tauri::tray::TrayIcon>>> = OnceLock::new();

pub fn install(app: &AppHandle) {
    #[cfg(target_os = "linux")]
    let handle = platform::spawn(app);
    #[cfg(not(target_os = "linux"))]
    let handle = {
        use tauri::{
            menu::{Menu, MenuItem},
            tray::TrayIconBuilder,
        };
        let build = || -> tauri::Result<tauri::tray::TrayIcon> {
            let start = MenuItem::with_id(app, "toggle", "Start / stop dictation", true, None::<&str>)?;
            let cancel = MenuItem::with_id(app, "cancel", "Cancel dictation", true, None::<&str>)?;
            let open = MenuItem::with_id(app, "open", "Open Local Dictation", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&start, &cancel, &open, &quit])?;
            TrayIconBuilder::with_id("main")
                .tooltip("Local Dictation — Idle")
                .menu(&menu)
                .on_menu_event(|app, event| {
                    use tauri::Manager;
                    let send = |action| {
                        if let Some(shared) = app.try_state::<std::sync::Arc<crate::state::Shared>>() {
                            let _ = shared.commands.send(crate::controller::Command::Ui(action));
                        }
                    };
                    match event.id().as_ref() {
                        "toggle" => send(crate::controller::UiAction::Toggle),
                        "cancel" => send(crate::controller::UiAction::Cancel),
                        "open" => crate::show_main_window(app),
                        "quit" => app.exit(0),
                        _ => {}
                    }
                })
                .build(app)
        };
        build().ok()
    };
    let _ = HANDLE.set(Mutex::new(handle));
}

pub fn refresh(app: &AppHandle, status: &StatusView) {
    let _ = app;
    let title = describe(status);
    let recording = matches!(status.recording_state, "recording" | "recording_locked" | "starting");
    let Some(slot) = HANDLE.get() else { return };
    let Ok(guard) = slot.lock() else { return };
    let Some(handle) = guard.as_ref() else { return };
    #[cfg(target_os = "linux")]
    {
        let handle = handle.clone();
        drop(guard);
        tauri::async_runtime::spawn(async move {
            let _ = handle
                .update(move |tray| {
                    tray.title = title;
                    tray.recording = recording;
                })
                .await;
        });
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = recording;
        let _ = handle.set_tooltip(Some(title));
    }
}
