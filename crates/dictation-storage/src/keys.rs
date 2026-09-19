//! Master key held in the operating-system credential store.
//!
//! macOS Keychain, Windows Credential Manager, or the freedesktop Secret
//! Service on Linux. When none is available, [`OsKeyring::load_or_create`]
//! returns `Ok(None)` and persistent contribution storage stays disabled; there
//! is no plaintext fallback (plan §6).

use rand::RngCore;
use zeroize::Zeroizing;

use crate::{KEY_LEN, MasterKeyProvider};

const SERVICE: &str = "app.localdictation.desktop";
/// Stored as 64 hexadecimal characters: some Secret Service backends do not
/// round-trip arbitrary binary secrets.
const ACCOUNT: &str = "storage-master-key-v2";

#[derive(Debug, Clone, Default)]
pub struct OsKeyring;

impl MasterKeyProvider for OsKeyring {
    fn load_or_create(&self) -> Result<Option<[u8; KEY_LEN]>, String> {
        let entry = match keyring::Entry::new(SERVICE, ACCOUNT) {
            Ok(entry) => entry,
            Err(keyring::Error::PlatformFailure(_) | keyring::Error::NoStorageAccess(_)) => {
                return Ok(None);
            }
            Err(error) => return Err(error_kind(&error).to_owned()),
        };
        match entry.get_password() {
            Ok(text) => {
                let text = Zeroizing::new(text);
                decode_key(&text).map(Some).ok_or_else(|| "stored_master_key_is_malformed".to_owned())
            }
            Err(keyring::Error::NoEntry) => {
                let mut key = Zeroizing::new([0_u8; KEY_LEN]);
                rand::rngs::OsRng.fill_bytes(key.as_mut());
                let encoded = Zeroizing::new(encode_key(&key));
                match entry.set_password(&encoded) {
                    Ok(()) => Ok(Some(*key)),
                    Err(keyring::Error::PlatformFailure(_) | keyring::Error::NoStorageAccess(_)) => {
                        Ok(None)
                    }
                    Err(error) => Err(error_kind(&error).to_owned()),
                }
            }
            Err(keyring::Error::PlatformFailure(_) | keyring::Error::NoStorageAccess(_)) => Ok(None),
            Err(error) => Err(error_kind(&error).to_owned()),
        }
    }
}

fn encode_key(key: &[u8; KEY_LEN]) -> String {
    use std::fmt::Write as _;
    key.iter().fold(String::with_capacity(KEY_LEN * 2), |mut text, byte| {
        let _ = write!(text, "{byte:02x}");
        text
    })
}

fn decode_key(text: &str) -> Option<[u8; KEY_LEN]> {
    if text.len() != KEY_LEN * 2 || !text.is_ascii() {
        return None;
    }
    let mut key = [0_u8; KEY_LEN];
    for (index, slot) in key.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(key)
}

impl OsKeyring {
    /// Destroys the master key, making every stored ciphertext unreadable.
    /// This is cryptographic erasure of this store only; it is not a promise
    /// of physical erasure from SSDs or backups.
    ///
    /// # Errors
    ///
    /// Returns an opaque provider error.
    pub fn destroy(&self) -> Result<(), String> {
        let entry = keyring::Entry::new(SERVICE, ACCOUNT).map_err(|error| error_kind(&error).to_owned())?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error_kind(&error).to_owned()),
        }
    }
}

const fn error_kind(error: &keyring::Error) -> &'static str {
    match error {
        keyring::Error::PlatformFailure(_) => "keyring_platform_failure",
        keyring::Error::NoStorageAccess(_) => "keyring_no_storage_access",
        keyring::Error::NoEntry => "keyring_no_entry",
        keyring::Error::BadEncoding(_) => "keyring_bad_encoding",
        keyring::Error::TooLong(..) => "keyring_too_long",
        keyring::Error::Invalid(..) => "keyring_invalid",
        keyring::Error::Ambiguous(_) => "keyring_ambiguous",
        _ => "keyring_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_encoding_round_trips() {
        let key = [0xab; KEY_LEN];
        assert_eq!(decode_key(&encode_key(&key)), Some(key));
        assert_eq!(decode_key("zz"), None);
    }
}
