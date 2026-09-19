//! Consent-based keyboard input through the RemoteDesktop portal (Wayland).
//!
//! The compositor asks the user once; a restore token lets later sessions
//! reuse that consent. Text is typed as keysyms. Characters without a keysym
//! mapping make the whole insertion fail rather than typing partial text.
//! Enter is never synthesized: newlines are refused so dictation cannot
//! submit a chat message or run a shell command.

use ashpd::{
    desktop::{
        PersistMode, Session,
        remote_desktop::{DeviceType, KeyState, RemoteDesktop, SelectDevicesOptions},
    },
    enumflags2::BitFlags,
};

pub struct PortalKeyboard {
    portal: RemoteDesktop,
    session: Session<RemoteDesktop>,
    pub restore_token: Option<String>,
}

/// X11 keysym for a character: Latin-1 maps directly, the rest use the
/// Unicode keysym range. Control characters have no safe mapping.
#[must_use]
pub fn keysym_for(character: char) -> Option<i32> {
    let code = u32::from(character);
    match code {
        0x20..=0x7e | 0xa0..=0xff => i32::try_from(code).ok(),
        0x100..=0x10_ffff => i32::try_from(0x0100_0000 + code).ok(),
        _ => None,
    }
}

impl PortalKeyboard {
    /// Opens a keyboard-only session, prompting the user unless a stored
    /// restore token is still valid.
    ///
    /// # Errors
    ///
    /// Fails when the portal is missing or the user declines.
    pub async fn open(restore_token: Option<&str>) -> Result<Self, String> {
        let portal = RemoteDesktop::new().await.map_err(|error| error.to_string())?;
        let session = portal
            .create_session(Default::default())
            .await
            .map_err(|error| error.to_string())?;
        portal
            .select_devices(
                &session,
                SelectDevicesOptions::default()
                    .set_devices(BitFlags::from(DeviceType::Keyboard))
                    .set_persist_mode(PersistMode::ExplicitlyRevoked)
                    .set_restore_token(restore_token),
            )
            .await
            .map_err(|error| error.to_string())?;
        let selected = portal
            .start(&session, None, Default::default())
            .await
            .map_err(|error| error.to_string())?
            .response()
            .map_err(|error| error.to_string())?;
        if !selected.devices().contains(DeviceType::Keyboard) {
            return Err("keyboard_access_not_granted".to_owned());
        }
        Ok(Self {
            restore_token: selected.restore_token().map(str::to_owned),
            portal,
            session,
        })
    }

    /// Types `text`; refuses newlines and unmappable characters up front.
    ///
    /// # Errors
    ///
    /// Fails before typing anything when the text cannot be typed safely.
    pub async fn type_text(&self, text: &str) -> Result<(), String> {
        let keysyms: Option<Vec<i32>> = text.chars().map(keysym_for).collect();
        let keysyms = keysyms.ok_or_else(|| "unsupported_character".to_owned())?;
        for keysym in keysyms {
            for state in [KeyState::Pressed, KeyState::Released] {
                self.portal
                    .notify_keyboard_keysym(&self.session, keysym, state, Default::default())
                    .await
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newlines_and_controls_have_no_keysym() {
        assert_eq!(keysym_for('a'), Some(0x61));
        assert_eq!(keysym_for('é'), Some(0xe9));
        assert_eq!(keysym_for('€'), Some(0x0100_20ac));
        assert_eq!(keysym_for('\n'), None);
        assert_eq!(keysym_for('\r'), None);
        assert_eq!(keysym_for('\t'), None);
    }
}
