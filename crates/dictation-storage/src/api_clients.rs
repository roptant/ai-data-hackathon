//! Paired local-API clients (plan §8).
//!
//! Only a salted hash of each bearer token is stored. Scopes are an explicit
//! allowlist; reading captions never implies session control.

use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::{EncryptedStore, StoreError};

pub(crate) const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS api_clients (
  client_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  token_hash BLOB NOT NULL UNIQUE,
  scopes TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  revoked_at INTEGER,
  last_used_at INTEGER
);
";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    TranscriptLive,
    TranscriptFinal,
    SessionControl,
    StatusRead,
}

impl Scope {
    pub const ALL: [Self; 4] = [
        Self::TranscriptLive,
        Self::TranscriptFinal,
        Self::SessionControl,
        Self::StatusRead,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TranscriptLive => "transcript:live",
            Self::TranscriptFinal => "transcript:final",
            Self::SessionControl => "session:control",
            Self::StatusRead => "status:read",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|scope| scope.as_str() == value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiClient {
    pub client_id: String,
    pub display_name: String,
    pub scopes: Vec<Scope>,
    pub created_at: i64,
    pub revoked_at: Option<i64>,
    pub last_used_at: Option<i64>,
}

impl ApiClient {
    #[must_use]
    pub fn has(&self, scope: Scope) -> bool {
        self.revoked_at.is_none() && self.scopes.contains(&scope)
    }
}

/// Domain-separated token digest. Tokens carry 256 bits of randomness, so a
/// fast hash is sufficient; no password stretching is needed.
#[must_use]
pub fn token_hash(token: &str) -> Vec<u8> {
    let mut digest = Sha256::new();
    digest.update(b"local-dictation-api-token-v1\0");
    digest.update(token.as_bytes());
    digest.finalize().to_vec()
}

fn scopes_text(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(|scope| scope.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn row_to_client(row: &rusqlite::Row<'_>) -> rusqlite::Result<ApiClient> {
    let scopes: String = row.get(2)?;
    Ok(ApiClient {
        client_id: row.get(0)?,
        display_name: row.get(1)?,
        scopes: scopes.split(' ').filter_map(Scope::parse).collect(),
        created_at: row.get(3)?,
        revoked_at: row.get(4)?,
        last_used_at: row.get(5)?,
    })
}

const COLUMNS: &str = "client_id,display_name,scopes,created_at,revoked_at,last_used_at";

impl EncryptedStore {
    /// # Errors
    ///
    /// Fails on database errors or a colliding token hash.
    pub fn insert_api_client(
        &self,
        client_id: &str,
        display_name: &str,
        token: &str,
        scopes: &[Scope],
        now: i64,
    ) -> Result<(), StoreError> {
        let name: String = display_name
            .chars()
            .filter(|character| !character.is_control())
            .take(64)
            .collect();
        self.connection.execute(
            "INSERT INTO api_clients(client_id,display_name,token_hash,scopes,created_at)
             VALUES(?1,?2,?3,?4,?5)",
            params![client_id, name, token_hash(token), scopes_text(scopes), now],
        )?;
        Ok(())
    }

    /// Resolves an unrevoked client by bearer token.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn authenticate_api_client(
        &self,
        token: &str,
        now: i64,
    ) -> Result<Option<ApiClient>, StoreError> {
        let client = self
            .connection
            .query_row(
                &format!(
                    "SELECT {COLUMNS} FROM api_clients WHERE token_hash=?1 AND revoked_at IS NULL"
                ),
                params![token_hash(token)],
                row_to_client,
            )
            .optional()?;
        if let Some(client) = &client {
            self.connection.execute(
                "UPDATE api_clients SET last_used_at=?2 WHERE client_id=?1",
                params![client.client_id, now],
            )?;
        }
        Ok(client)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn api_clients(&self) -> Result<Vec<ApiClient>, StoreError> {
        let mut statement = self
            .connection
            .prepare(&format!("SELECT {COLUMNS} FROM api_clients ORDER BY created_at"))?;
        let rows = statement.query_map([], row_to_client)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn revoke_api_client(&self, client_id: &str, now: i64) -> Result<bool, StoreError> {
        Ok(self.connection.execute(
            "UPDATE api_clients SET revoked_at=?2 WHERE client_id=?1 AND revoked_at IS NULL",
            params![client_id, now],
        )? > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::store;

    #[test]
    fn tokens_are_hashed_scoped_and_revocable() {
        let store = store();
        store
            .insert_api_client("client-1", "Captions", "secret-token", &[Scope::TranscriptLive], 1)
            .unwrap();
        let stored: Vec<u8> = store
            .connection
            .query_row("SELECT token_hash FROM api_clients", [], |row| row.get(0))
            .unwrap();
        assert_ne!(stored, b"secret-token".to_vec());
        let client = store.authenticate_api_client("secret-token", 2).unwrap().unwrap();
        assert!(client.has(Scope::TranscriptLive));
        assert!(!client.has(Scope::SessionControl));
        assert!(store.authenticate_api_client("wrong", 2).unwrap().is_none());
        assert!(store.revoke_api_client("client-1", 3).unwrap());
        assert!(store.authenticate_api_client("secret-token", 4).unwrap().is_none());
    }
}
