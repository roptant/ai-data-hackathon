//! Encrypted, application-owned persistence with independent retention scopes.
//!
//! Content is encrypted before it reaches SQLite. A caller-provided master-key
//! source represents the OS credential store; opening persistent storage fails
//! closed when that source is unavailable.

use std::{fmt, path::Path};

use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, AeadCore, KeyInit, OsRng, Payload},
};
use rand::RngCore;
use rusqlite::{Connection, OptionalExtension, params};
use zeroize::Zeroizing;

const KEY_LEN: usize = 32;
const NONCE_LEN: usize = 24;

pub trait MasterKeyProvider {
    /// Loads or creates the application master key in secure OS storage.
    ///
    /// # Errors
    ///
    /// Returns an opaque provider error without secret material.
    fn load_or_create(&self) -> Result<Option<[u8; KEY_LEN]>, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataClass {
    RawSession,
    EligiblePackage,
}

impl DataClass {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RawSession => "raw_session",
            Self::EligiblePackage => "eligible_package",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CleanupReport {
    pub expired_scopes: usize,
    pub expired_metadata: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationalMetadata<'a> {
    pub metadata_id: &'a str,
    pub state: &'a str,
    pub reason_code: Option<&'a str>,
    pub duration_ms: u64,
    pub word_count: u64,
    pub expires_at: i64,
    pub created_at: i64,
}

#[derive(Debug)]
pub enum StoreError {
    KeyUnavailable,
    KeyProvider(String),
    Database(rusqlite::Error),
    Crypto,
    MissingScope,
    InvalidKeyEnvelope,
    InvalidMetadata,
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyUnavailable => write!(formatter, "secure master-key storage is unavailable"),
            Self::KeyProvider(message) => {
                write!(formatter, "master-key provider failed: {message}")
            }
            Self::Database(error) => write!(formatter, "storage database failed: {error}"),
            Self::Crypto => write!(formatter, "authenticated encryption failed"),
            Self::MissingScope => write!(formatter, "encryption scope does not exist"),
            Self::InvalidKeyEnvelope => write!(formatter, "wrapped key has an invalid envelope"),
            Self::InvalidMetadata => {
                write!(formatter, "operational metadata contains an invalid token")
            }
        }
    }
}

impl std::error::Error for StoreError {}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

pub struct EncryptedStore {
    connection: Connection,
    master_key: Zeroizing<[u8; KEY_LEN]>,
}

impl EncryptedStore {
    /// Opens an application-owned database using a securely sourced master key.
    ///
    /// # Errors
    ///
    /// Fails when secure key storage is unavailable, the database cannot open,
    /// or its schema cannot be initialized. No plaintext fallback is provided.
    pub fn open(path: &Path, provider: &impl MasterKeyProvider) -> Result<Self, StoreError> {
        let key = provider
            .load_or_create()
            .map_err(StoreError::KeyProvider)?
            .ok_or(StoreError::KeyUnavailable)?;
        let connection = Connection::open(path)?;
        let store = Self {
            connection,
            master_key: Zeroizing::new(key),
        };
        store.initialize()?;
        Ok(store)
    }

    /// Opens an in-memory encrypted database, primarily for integration tests.
    ///
    /// # Errors
    ///
    /// Fails when secure key storage is unavailable or schema setup fails.
    pub fn open_in_memory(provider: &impl MasterKeyProvider) -> Result<Self, StoreError> {
        let key = provider
            .load_or_create()
            .map_err(StoreError::KeyProvider)?
            .ok_or(StoreError::KeyUnavailable)?;
        let connection = Connection::open_in_memory()?;
        let store = Self {
            connection,
            master_key: Zeroizing::new(key),
        };
        store.initialize()?;
        Ok(store)
    }

    fn initialize(&self) -> Result<(), StoreError> {
        self.connection.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA secure_delete = ON;
             CREATE TABLE IF NOT EXISTS encryption_scopes (
               scope_id TEXT PRIMARY KEY,
               data_class TEXT NOT NULL CHECK(data_class IN ('raw_session','eligible_package')),
               wrapped_key BLOB NOT NULL,
               expires_at INTEGER NOT NULL,
               created_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS encrypted_artifacts (
               artifact_id TEXT PRIMARY KEY,
               scope_id TEXT NOT NULL REFERENCES encryption_scopes(scope_id) ON DELETE CASCADE,
               kind TEXT NOT NULL,
               ciphertext BLOB NOT NULL,
               created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS artifacts_scope_idx
               ON encrypted_artifacts(scope_id);
             CREATE TABLE IF NOT EXISTS operational_metadata (
               metadata_id TEXT PRIMARY KEY,
               state TEXT NOT NULL,
               reason_code TEXT,
               duration_ms INTEGER NOT NULL,
               word_count INTEGER NOT NULL,
               expires_at INTEGER NOT NULL,
               created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS scopes_expiry_idx ON encryption_scopes(expires_at);
             CREATE INDEX IF NOT EXISTS metadata_expiry_idx ON operational_metadata(expires_at);",
        )?;
        Ok(())
    }

    /// Creates a separately expiring encryption scope.
    ///
    /// # Errors
    ///
    /// Fails on duplicate identifiers, random generation failure, encryption
    /// failure, or database failure.
    pub fn create_scope(
        &self,
        scope_id: &str,
        data_class: DataClass,
        expires_at: i64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        let mut data_key = Zeroizing::new([0_u8; KEY_LEN]);
        OsRng.fill_bytes(data_key.as_mut());
        let wrapped = seal(&self.master_key, data_key.as_ref(), scope_id.as_bytes())?;
        self.connection.execute(
            "INSERT INTO encryption_scopes(scope_id,data_class,wrapped_key,expires_at,created_at)
             VALUES(?1,?2,?3,?4,?5)",
            params![
                scope_id,
                data_class.as_str(),
                wrapped,
                expires_at,
                created_at
            ],
        )?;
        Ok(())
    }

    /// Stores content encrypted under its scope key.
    ///
    /// # Errors
    ///
    /// Fails if the scope does not exist, decryption/encryption fails, the
    /// artifact ID already exists, or SQLite rejects the write.
    pub fn put_artifact(
        &self,
        scope_id: &str,
        artifact_id: &str,
        kind: &str,
        plaintext: &[u8],
        created_at: i64,
    ) -> Result<(), StoreError> {
        let data_key = self.scope_key(scope_id)?;
        let aad = artifact_aad(scope_id, artifact_id, kind);
        let ciphertext = seal(&data_key, plaintext, aad.as_bytes())?;
        self.connection.execute(
            "INSERT INTO encrypted_artifacts(artifact_id,scope_id,kind,ciphertext,created_at)
             VALUES(?1,?2,?3,?4,?5)",
            params![artifact_id, scope_id, kind, ciphertext, created_at],
        )?;
        Ok(())
    }

    /// Decrypts one artifact after authenticating its identity and kind.
    ///
    /// # Errors
    ///
    /// Fails if the artifact or scope is absent, its envelope was modified, or
    /// database access fails.
    pub fn get_artifact(&self, artifact_id: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let row: Option<(String, String, Vec<u8>)> = self
            .connection
            .query_row(
                "SELECT scope_id,kind,ciphertext FROM encrypted_artifacts WHERE artifact_id=?1",
                params![artifact_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((scope_id, kind, ciphertext)) = row else {
            return Ok(None);
        };
        let key = self.scope_key(&scope_id)?;
        let aad = artifact_aad(&scope_id, artifact_id, &kind);
        open(&key, &ciphertext, aad.as_bytes()).map(Some)
    }

    /// Stores bounded, non-content operational metadata with its own expiry.
    ///
    /// # Errors
    ///
    /// Fails when SQLite rejects the row.
    pub fn put_metadata(&self, metadata: OperationalMetadata<'_>) -> Result<(), StoreError> {
        if !safe_metadata_token(metadata.metadata_id, 128)
            || !safe_metadata_token(metadata.state, 64)
            || metadata
                .reason_code
                .is_some_and(|reason| !safe_metadata_token(reason, 64))
        {
            return Err(StoreError::InvalidMetadata);
        }
        let duration_ms =
            i64::try_from(metadata.duration_ms).map_err(|_| StoreError::InvalidMetadata)?;
        let word_count =
            i64::try_from(metadata.word_count).map_err(|_| StoreError::InvalidMetadata)?;
        self.connection.execute(
            "INSERT INTO operational_metadata(
               metadata_id,state,reason_code,duration_ms,word_count,expires_at,created_at
             ) VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                metadata.metadata_id,
                metadata.state,
                metadata.reason_code,
                duration_ms,
                word_count,
                metadata.expires_at,
                metadata.created_at,
            ],
        )?;
        Ok(())
    }

    /// Deletes expired content scopes and metadata in one atomic transaction.
    ///
    /// # Errors
    ///
    /// Fails when SQLite cannot complete the transaction.
    pub fn cleanup_expired(&mut self, now: i64) -> Result<CleanupReport, StoreError> {
        let transaction = self.connection.transaction()?;
        let expired_scopes = transaction.execute(
            "DELETE FROM encryption_scopes WHERE expires_at <= ?1",
            params![now],
        )?;
        let expired_metadata = transaction.execute(
            "DELETE FROM operational_metadata WHERE expires_at <= ?1",
            params![now],
        )?;
        transaction.commit()?;
        Ok(CleanupReport {
            expired_scopes,
            expired_metadata,
        })
    }

    /// Deletes a scope and all of its artifacts immediately.
    ///
    /// # Errors
    ///
    /// Fails when SQLite cannot execute the deletion.
    pub fn delete_scope(&self, scope_id: &str) -> Result<bool, StoreError> {
        Ok(self.connection.execute(
            "DELETE FROM encryption_scopes WHERE scope_id=?1",
            params![scope_id],
        )? > 0)
    }

    fn scope_key(&self, scope_id: &str) -> Result<Zeroizing<[u8; KEY_LEN]>, StoreError> {
        let wrapped: Option<Vec<u8>> = self
            .connection
            .query_row(
                "SELECT wrapped_key FROM encryption_scopes WHERE scope_id=?1",
                params![scope_id],
                |row| row.get(0),
            )
            .optional()?;
        let wrapped = wrapped.ok_or(StoreError::MissingScope)?;
        let plaintext = open(&self.master_key, &wrapped, scope_id.as_bytes())?;
        let key: [u8; KEY_LEN] = plaintext
            .try_into()
            .map_err(|_| StoreError::InvalidKeyEnvelope)?;
        Ok(Zeroizing::new(key))
    }
}

fn artifact_aad(scope_id: &str, artifact_id: &str, kind: &str) -> String {
    format!("artifact\0{scope_id}\0{artifact_id}\0{kind}")
}

fn safe_metadata_token(value: &str, maximum_length: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum_length
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'-' | b'_' | b':' | b'.')
        })
}

fn seal(key: &[u8; KEY_LEN], plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, StoreError> {
    let cipher = XChaCha20Poly1305::new(key.into());
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| StoreError::Crypto)?;
    let mut envelope = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    envelope.extend_from_slice(&nonce);
    envelope.extend_from_slice(&ciphertext);
    Ok(envelope)
}

fn open(key: &[u8; KEY_LEN], envelope: &[u8], aad: &[u8]) -> Result<Vec<u8>, StoreError> {
    if envelope.len() <= NONCE_LEN {
        return Err(StoreError::InvalidKeyEnvelope);
    }
    let (nonce, ciphertext) = envelope.split_at(NONCE_LEN);
    XChaCha20Poly1305::new(key.into())
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| StoreError::Crypto)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestKeys(Option<[u8; KEY_LEN]>);

    impl MasterKeyProvider for TestKeys {
        fn load_or_create(&self) -> Result<Option<[u8; KEY_LEN]>, String> {
            Ok(self.0)
        }
    }

    fn store() -> EncryptedStore {
        EncryptedStore::open_in_memory(&TestKeys(Some([7; KEY_LEN]))).unwrap()
    }

    #[test]
    fn secure_key_unavailability_disables_persistence() {
        assert!(matches!(
            EncryptedStore::open_in_memory(&TestKeys(None)),
            Err(StoreError::KeyUnavailable)
        ));
    }

    #[test]
    fn artifacts_round_trip_through_authenticated_encryption() {
        let store = store();
        store
            .create_scope("raw-1", DataClass::RawSession, 100, 1)
            .unwrap();
        store
            .put_artifact("raw-1", "audio-1", "audio", b"private audio", 1)
            .unwrap();
        assert_eq!(
            store.get_artifact("audio-1").unwrap(),
            Some(b"private audio".to_vec())
        );
    }

    #[test]
    fn raw_and_package_expire_independently() {
        let mut store = store();
        store
            .create_scope("raw-1", DataClass::RawSession, 10, 1)
            .unwrap();
        store
            .create_scope("package-1", DataClass::EligiblePackage, 100, 1)
            .unwrap();
        store
            .put_artifact("raw-1", "audio-1", "audio", b"raw", 1)
            .unwrap();
        store
            .put_artifact("package-1", "zip-1", "package", b"eligible", 1)
            .unwrap();
        store
            .put_metadata(OperationalMetadata {
                metadata_id: "meta-1",
                state: "acknowledged",
                reason_code: None,
                duration_ms: 1_000,
                word_count: 3,
                expires_at: 50,
                created_at: 1,
            })
            .unwrap();

        let report = store.cleanup_expired(20).unwrap();
        assert_eq!(report.expired_scopes, 1);
        assert_eq!(report.expired_metadata, 0);
        assert_eq!(store.get_artifact("audio-1").unwrap(), None);
        assert_eq!(
            store.get_artifact("zip-1").unwrap(),
            Some(b"eligible".to_vec())
        );

        let report = store.cleanup_expired(60).unwrap();
        assert_eq!(report.expired_scopes, 0);
        assert_eq!(report.expired_metadata, 1);
        assert_eq!(
            store.get_artifact("zip-1").unwrap(),
            Some(b"eligible".to_vec())
        );
    }

    #[test]
    fn deleting_scope_cascades_to_content() {
        let store = store();
        store
            .create_scope("raw-1", DataClass::RawSession, 100, 1)
            .unwrap();
        store
            .put_artifact("raw-1", "text-1", "transcript", b"secret", 1)
            .unwrap();
        assert!(store.delete_scope("raw-1").unwrap());
        assert_eq!(store.get_artifact("text-1").unwrap(), None);
    }

    #[test]
    fn artifact_identity_is_authenticated() {
        let store = store();
        store
            .create_scope("raw-1", DataClass::RawSession, 100, 1)
            .unwrap();
        store
            .put_artifact("raw-1", "text-1", "transcript", b"secret", 1)
            .unwrap();
        store
            .connection
            .execute(
                "UPDATE encrypted_artifacts SET kind='audio' WHERE artifact_id='text-1'",
                [],
            )
            .unwrap();
        assert!(matches!(
            store.get_artifact("text-1"),
            Err(StoreError::Crypto)
        ));
    }

    #[test]
    fn operational_metadata_rejects_free_form_content() {
        let store = store();
        let result = store.put_metadata(OperationalMetadata {
            metadata_id: "meta-1",
            state: "the customer said a secret",
            reason_code: None,
            duration_ms: 1_000,
            word_count: 3,
            expires_at: 50,
            created_at: 1,
        });
        assert!(matches!(result, Err(StoreError::InvalidMetadata)));
    }
}
