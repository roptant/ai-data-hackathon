//! Disclosed clipboard fallback (plan §5).
//!
//! The dictation is placed on the clipboard for the user to paste manually.
//! Where supported it is marked to be excluded from clipboard history. The
//! previous text is restored later only if the clipboard still holds exactly
//! the app's own replacement; external clipboard managers may still have
//! observed it, which the UI states rather than promising erasure.

use std::time::Duration;

#[derive(Debug)]
pub struct ClipboardError(pub String);

impl std::fmt::Display for ClipboardError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "clipboard unavailable: {}", self.0)
    }
}

impl std::error::Error for ClipboardError {}

/// Restores the previous clipboard text when dropped or when `restore` runs,
/// but only if nothing else replaced the dictation in the meantime.
pub struct ClipboardLease {
    placed: String,
    previous: Option<String>,
}

fn clipboard() -> Result<arboard::Clipboard, ClipboardError> {
    arboard::Clipboard::new().map_err(|error| ClipboardError(error.to_string()))
}

/// Places `text` on the clipboard.
///
/// # Errors
///
/// Fails when no clipboard is reachable.
pub fn place(text: &str) -> Result<ClipboardLease, ClipboardError> {
    let mut board = clipboard()?;
    let previous = board.get_text().ok();
    let setter = board.set();
    #[cfg(target_os = "linux")]
    let setter = {
        use arboard::SetExtLinux;
        setter.exclude_from_history()
    };
    #[cfg(target_os = "macos")]
    let setter = {
        use arboard::SetExtApple;
        setter.exclude_from_history()
    };
    #[cfg(target_os = "windows")]
    let setter = {
        use arboard::SetExtWindows;
        setter.exclude_from_history().exclude_from_cloud()
    };
    setter
        .text(text.to_owned())
        .map_err(|error| ClipboardError(error.to_string()))?;
    Ok(ClipboardLease {
        placed: text.to_owned(),
        previous,
    })
}

impl ClipboardLease {
    /// Restores the previous text after `delay` on a background thread.
    pub fn restore_later(self, delay: Duration) {
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            self.restore();
        });
    }

    /// Restores now if the clipboard still holds the dictation.
    pub fn restore(self) {
        let Ok(mut board) = clipboard() else { return };
        if board.get_text().ok().as_deref() != Some(self.placed.as_str()) {
            return; // The user or another app replaced it: leave it alone.
        }
        match self.previous {
            Some(previous) => {
                let _ = board.set_text(previous);
            }
            None => {
                let _ = board.clear();
            }
        }
    }
}
