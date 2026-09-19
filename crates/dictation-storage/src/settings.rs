//! Encrypted user settings: private terms, excluded applications, shortcuts.
//!
//! Private terms are exactly the words a user considers sensitive, so the
//! whole settings document is encrypted under the master key and bound to its
//! key name.

use rusqlite::{OptionalExtension, params};

use crate::{EncryptedStore, StoreError, open, seal};

pub(crate) const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS settings (
  name TEXT PRIMARY KEY,
  ciphertext BLOB NOT NULL,
  updated_at INTEGER NOT NULL
);
";

fn aad(name: &str) -> String {
    format!("setting\0{name}")
}

impl EncryptedStore {
    /// Stores one encrypted settings document.
    ///
    /// # Errors
    ///
    /// Fails on encryption or database errors.
    pub fn put_setting(&self, name: &str, plaintext: &[u8], now: i64) -> Result<(), StoreError> {
        let ciphertext = seal(&self.master_key, plaintext, aad(name).as_bytes())?;
        self.connection.execute(
            "INSERT INTO settings(name,ciphertext,updated_at) VALUES(?1,?2,?3)
             ON CONFLICT(name) DO UPDATE SET ciphertext=excluded.ciphertext, updated_at=excluded.updated_at",
            params![name, ciphertext, now],
        )?;
        Ok(())
    }

    /// Loads and authenticates one settings document.
    ///
    /// # Errors
    ///
    /// Fails when the ciphertext was modified or bound to another name.
    pub fn get_setting(&self, name: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let ciphertext: Option<Vec<u8>> = self
            .connection
            .query_row(
                "SELECT ciphertext FROM settings WHERE name=?1",
                params![name],
                |row| row.get(0),
            )
            .optional()?;
        ciphertext
            .map(|ciphertext| open(&self.master_key, &ciphertext, aad(name).as_bytes()))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::store;

    #[test]
    fn settings_round_trip_and_are_bound_to_their_name() {
        let store = store();
        store.put_setting("private_terms", b"[\"falcon\"]", 1).unwrap();
        assert_eq!(
            store.get_setting("private_terms").unwrap(),
            Some(b"[\"falcon\"]".to_vec())
        );
        store
            .connection
            .execute("UPDATE settings SET name='other' WHERE name='private_terms'", [])
            .unwrap();
        assert!(store.get_setting("other").is_err());
        assert_eq!(store.get_setting("missing").unwrap(), None);
    }
}
