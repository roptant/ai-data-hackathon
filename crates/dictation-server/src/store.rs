//! Server persistence: tenants, consent, admission receipts, encrypted sample
//! objects, lineage, tombstones, and deliveries.
//!
//! Tenant identity comes only from the authenticated credential. Objects are
//! encrypted at rest under per-tenant keys derived from the server master key
//! and bound to their tenant and sample identifiers. Logs and errors carry
//! identifiers and reason codes, never package contents.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, AeadCore, KeyInit, OsRng, Payload},
};
use hkdf::Hkdf;
use rand::RngCore;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const SCHEMA: &str = "
PRAGMA foreign_keys = ON;
PRAGMA secure_delete = ON;
CREATE TABLE IF NOT EXISTS tenants (
  tenant_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  token_hash BLOB NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  disabled_at INTEGER,
  max_samples INTEGER NOT NULL,
  max_storage_bytes INTEGER NOT NULL,
  max_training_seconds INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS consents (
  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
  consent_id TEXT NOT NULL,
  version TEXT NOT NULL,
  purposes TEXT NOT NULL,
  granted_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  withdrawn_at INTEGER,
  PRIMARY KEY (tenant_id, consent_id)
);
CREATE TABLE IF NOT EXISTS receipts (
  tenant_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  receipt_id TEXT NOT NULL UNIQUE,
  accepted INTEGER NOT NULL,
  reason TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  received_at INTEGER NOT NULL,
  PRIMARY KEY (tenant_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  consent_id TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  clip_count INTEGER NOT NULL,
  received_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_tenant_idx ON samples(tenant_id, received_at);
CREATE TABLE IF NOT EXISTS tombstones (
  sample_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  deleted_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  attributes TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  deleted_at INTEGER
);
CREATE TABLE IF NOT EXISTS edges (
  parent TEXT NOT NULL,
  child TEXT NOT NULL,
  PRIMARY KEY (parent, child)
);
CREATE TABLE IF NOT EXISTS training_jobs (
  job_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  state TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  finished_at INTEGER
);
CREATE TABLE IF NOT EXISTS deliveries (
  model_version TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  signature_hex TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  revoked_at INTEGER
);
CREATE TABLE IF NOT EXISTS deletion_log (
  tenant_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  completed_at INTEGER NOT NULL,
  deleted_samples INTEGER NOT NULL,
  deleted_artifacts INTEGER NOT NULL,
  PRIMARY KEY (tenant_id, request_id)
);
";

/// Server-side sample retention (plan §6): up to 30 days.
pub const SAMPLE_RETENTION_SECONDS: i64 = 30 * 24 * 3600;

#[derive(Debug)]
pub enum ServerError {
    Database(rusqlite::Error),
    Io(std::io::Error),
    Crypto,
    Conflict(&'static str),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Database(error) => write!(formatter, "database: {error}"),
            Self::Io(error) => write!(formatter, "io: {error}"),
            Self::Crypto => write!(formatter, "object encryption failed"),
            Self::Conflict(code) => write!(formatter, "conflict: {code}"),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<rusqlite::Error> for ServerError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

impl From<std::io::Error> for ServerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tenant {
    pub tenant_id: String,
    pub display_name: String,
    pub max_samples: u64,
    pub max_storage_bytes: u64,
    pub max_training_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredReceipt {
    pub receipt_id: String,
    pub accepted: bool,
    pub reason: String,
    pub sample_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleRow {
    pub sample_id: String,
    pub consent_id: String,
    pub received_at: i64,
    pub duration_ms: u64,
}

#[must_use]
pub fn token_hash(token: &str) -> Vec<u8> {
    let mut digest = Sha256::new();
    digest.update(b"local-dictation-tenant-token-v1\0");
    digest.update(token.as_bytes());
    digest.finalize().to_vec()
}

#[must_use]
pub fn random_hex(bytes: usize) -> String {
    let mut buffer = vec![0_u8; bytes];
    OsRng.fill_bytes(&mut buffer);
    hex::encode(buffer)
}

fn to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

pub struct ServerStore {
    pub(crate) connection: Connection,
    root: PathBuf,
    master_key: Zeroizing<[u8; 32]>,
}

impl ServerStore {
    /// Opens or initializes the server state under `root`.
    ///
    /// # Errors
    ///
    /// Fails when the directory, key file, or database cannot be prepared.
    pub fn open(root: &Path) -> Result<Self, ServerError> {
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("deliveries"))?;
        fs::create_dir_all(root.join("jobs"))?;
        restrict(root)?;
        let key_path = root.join("master.key");
        let master_key = if key_path.exists() {
            let bytes = Zeroizing::new(fs::read(&key_path)?);
            let key: [u8; 32] = bytes.as_slice().try_into().map_err(|_| ServerError::Crypto)?;
            key
        } else {
            let mut key = [0_u8; 32];
            OsRng.fill_bytes(&mut key);
            write_private(&key_path, &key)?;
            key
        };
        let connection = Connection::open(root.join("server.sqlite3"))?;
        connection.execute_batch(SCHEMA)?;
        Ok(Self {
            connection,
            root: root.to_path_buf(),
            master_key: Zeroizing::new(master_key),
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn tenant_key(&self, tenant_id: &str) -> Result<Zeroizing<[u8; 32]>, ServerError> {
        let hkdf = Hkdf::<Sha256>::new(Some(b"local-dictation-object-v1"), self.master_key.as_ref());
        let mut key = Zeroizing::new([0_u8; 32]);
        hkdf.expand(tenant_id.as_bytes(), key.as_mut())
            .map_err(|_| ServerError::Crypto)?;
        Ok(key)
    }

    fn object_path(&self, tenant_id: &str, sample_id: &str) -> PathBuf {
        self.root.join("objects").join(tenant_id).join(format!("{sample_id}.bin"))
    }

    fn seal_object(&self, tenant_id: &str, object_id: &str, plaintext: &[u8]) -> Result<Vec<u8>, ServerError> {
        let key = self.tenant_key(tenant_id)?;
        let cipher = XChaCha20Poly1305::new(key.as_ref().into());
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let aad = format!("{tenant_id}\0{object_id}");
        let mut output = nonce.to_vec();
        output.extend(
            cipher
                .encrypt(&nonce, Payload { msg: plaintext, aad: aad.as_bytes() })
                .map_err(|_| ServerError::Crypto)?,
        );
        Ok(output)
    }

    fn open_object(&self, tenant_id: &str, object_id: &str, envelope: &[u8]) -> Result<Vec<u8>, ServerError> {
        if envelope.len() <= 24 {
            return Err(ServerError::Crypto);
        }
        let key = self.tenant_key(tenant_id)?;
        let (nonce, ciphertext) = envelope.split_at(24);
        let aad = format!("{tenant_id}\0{object_id}");
        XChaCha20Poly1305::new(key.as_ref().into())
            .decrypt(XNonce::from_slice(nonce), Payload { msg: ciphertext, aad: aad.as_bytes() })
            .map_err(|_| ServerError::Crypto)
    }

    // -- tenants --------------------------------------------------------------

    /// Creates a tenant and returns its bearer token, shown once.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn create_tenant(&self, display_name: &str, now: i64) -> Result<(Tenant, String), ServerError> {
        let tenant = Tenant {
            tenant_id: format!("tenant-{}", random_hex(12)),
            display_name: display_name.chars().filter(|c| !c.is_control()).take(64).collect(),
            max_samples: 20_000,
            max_storage_bytes: 5 * 1024 * 1024 * 1024,
            max_training_seconds: 4 * 3600,
        };
        let token = format!("ldt_{}", random_hex(32));
        self.connection.execute(
            "INSERT INTO tenants(tenant_id,display_name,token_hash,created_at,max_samples,max_storage_bytes,max_training_seconds)
             VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                tenant.tenant_id,
                tenant.display_name,
                token_hash(&token),
                now,
                to_i64(tenant.max_samples),
                to_i64(tenant.max_storage_bytes),
                to_i64(tenant.max_training_seconds)
            ],
        )?;
        fs::create_dir_all(self.root.join("objects").join(&tenant.tenant_id))?;
        Ok((tenant, token))
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn authenticate(&self, token: &str) -> Result<Option<Tenant>, ServerError> {
        Ok(self
            .connection
            .query_row(
                "SELECT tenant_id,display_name,max_samples,max_storage_bytes,max_training_seconds
                 FROM tenants WHERE token_hash=?1 AND disabled_at IS NULL",
                params![token_hash(token)],
                |row| {
                    Ok(Tenant {
                        tenant_id: row.get(0)?,
                        display_name: row.get(1)?,
                        max_samples: to_u64(row.get(2)?),
                        max_storage_bytes: to_u64(row.get(3)?),
                        max_training_seconds: to_u64(row.get(4)?),
                    })
                },
            )
            .optional()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn set_limits(&self, tenant_id: &str, max_samples: u64, max_training_seconds: u64) -> Result<(), ServerError> {
        self.connection.execute(
            "UPDATE tenants SET max_samples=?2, max_training_seconds=?3 WHERE tenant_id=?1",
            params![tenant_id, to_i64(max_samples), to_i64(max_training_seconds)],
        )?;
        Ok(())
    }

    // -- consent --------------------------------------------------------------

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn record_consent(
        &self,
        tenant_id: &str,
        grant: &dictation_protocol::ConsentGrant,
    ) -> Result<(), ServerError> {
        self.connection.execute(
            "INSERT INTO consents(tenant_id,consent_id,version,purposes,granted_at,expires_at)
             VALUES(?1,?2,?3,?4,?5,?6)
             ON CONFLICT(tenant_id,consent_id) DO NOTHING",
            params![
                tenant_id,
                grant.consent_id,
                grant.version,
                grant.purposes.join(","),
                grant.granted_at,
                grant.expires_at
            ],
        )?;
        Ok(())
    }

    /// Whether `consent_id` currently permits personalization for the tenant.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn consent_active(&self, tenant_id: &str, consent_id: &str, now: i64) -> Result<bool, ServerError> {
        let row: Option<(String, String, i64, Option<i64>)> = self
            .connection
            .query_row(
                "SELECT version,purposes,expires_at,withdrawn_at FROM consents WHERE tenant_id=?1 AND consent_id=?2",
                params![tenant_id, consent_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        Ok(row.is_some_and(|(version, purposes, expires_at, withdrawn)| {
            version == dictation_core::contribution::CONSENT_VERSION
                && purposes.split(',').any(|p| p == "customer_personalization")
                && expires_at > now
                && withdrawn.is_none()
        }))
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn withdraw_consent(&self, tenant_id: &str, consent_id: &str, now: i64) -> Result<bool, ServerError> {
        Ok(self.connection.execute(
            "UPDATE consents SET withdrawn_at=?3 WHERE tenant_id=?1 AND consent_id=?2 AND withdrawn_at IS NULL",
            params![tenant_id, consent_id, now],
        )? > 0)
    }

    // -- admission -------------------------------------------------------------

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn receipt_for(&self, tenant_id: &str, idempotency_key: &str) -> Result<Option<StoredReceipt>, ServerError> {
        Ok(self
            .connection
            .query_row(
                "SELECT receipt_id,accepted,reason,sample_id FROM receipts WHERE tenant_id=?1 AND idempotency_key=?2",
                params![tenant_id, idempotency_key],
                |row| {
                    Ok(StoredReceipt {
                        receipt_id: row.get(0)?,
                        accepted: row.get::<_, i64>(1)? != 0,
                        reason: row.get(2)?,
                        sample_id: row.get(3)?,
                    })
                },
            )
            .optional()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn record_refusal(&self, tenant_id: &str, idempotency_key: &str, reason: &str, now: i64) -> Result<StoredReceipt, ServerError> {
        let receipt = StoredReceipt {
            receipt_id: format!("receipt-{}", random_hex(12)),
            accepted: false,
            reason: reason.to_owned(),
            sample_id: String::new(),
        };
        self.connection.execute(
            "INSERT OR IGNORE INTO receipts(tenant_id,idempotency_key,receipt_id,accepted,reason,sample_id,received_at)
             VALUES(?1,?2,?3,0,?4,'',?5)",
            params![tenant_id, idempotency_key, receipt.receipt_id, reason, now],
        )?;
        Ok(self.receipt_for(tenant_id, idempotency_key)?.unwrap_or(receipt))
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn is_tombstoned(&self, sample_id: &str) -> Result<bool, ServerError> {
        Ok(self
            .connection
            .query_row("SELECT 1 FROM tombstones WHERE sample_id=?1", params![sample_id], |_| Ok(()))
            .optional()?
            .is_some())
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn sample_exists(&self, sample_id: &str) -> Result<bool, ServerError> {
        Ok(self
            .connection
            .query_row("SELECT 1 FROM samples WHERE sample_id=?1", params![sample_id], |_| Ok(()))
            .optional()?
            .is_some())
    }

    /// (sample count, stored bytes) for a tenant.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn usage(&self, tenant_id: &str) -> Result<(u64, u64), ServerError> {
        let (count, bytes): (i64, i64) = self.connection.query_row(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM samples WHERE tenant_id=?1",
            params![tenant_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok((to_u64(count), to_u64(bytes)))
    }

    /// Stores an admitted sample and its receipt atomically with its lineage node.
    ///
    /// # Errors
    ///
    /// Fails on database, filesystem, or encryption errors.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_sample(
        &mut self,
        tenant_id: &str,
        idempotency_key: &str,
        sample_id: &str,
        consent_id: &str,
        archive: &[u8],
        duration_ms: u64,
        clip_count: usize,
        now: i64,
    ) -> Result<StoredReceipt, ServerError> {
        let digest = hex::encode(Sha256::digest(archive));
        let envelope = self.seal_object(tenant_id, sample_id, archive)?;
        let path = self.object_path(tenant_id, sample_id);
        fs::create_dir_all(path.parent().unwrap_or(&self.root))?;
        let partial = path.with_extension("part");
        write_private(&partial, &envelope)?;
        let receipt = StoredReceipt {
            receipt_id: format!("receipt-{}", random_hex(12)),
            accepted: true,
            reason: String::new(),
            sample_id: sample_id.to_owned(),
        };
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO samples(sample_id,tenant_id,consent_id,sha256,size_bytes,duration_ms,clip_count,received_at,expires_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                sample_id,
                tenant_id,
                consent_id,
                digest,
                to_i64(archive.len() as u64),
                to_i64(duration_ms),
                to_i64(clip_count as u64),
                now,
                now + SAMPLE_RETENTION_SECONDS
            ],
        )?;
        transaction.execute(
            "INSERT INTO receipts(tenant_id,idempotency_key,receipt_id,accepted,reason,sample_id,received_at)
             VALUES(?1,?2,?3,1,'',?4,?5)",
            params![tenant_id, idempotency_key, receipt.receipt_id, sample_id, now],
        )?;
        transaction.execute(
            "INSERT INTO nodes(node_id,kind,tenant_id,attributes,created_at) VALUES(?1,'sample',?2,?3,?4)",
            params![
                sample_id,
                tenant_id,
                serde_json::json!({ "duration_ms": duration_ms, "clip_count": clip_count, "sha256": digest }).to_string(),
                now
            ],
        )?;
        fs::rename(&partial, &path)?;
        transaction.commit()?;
        Ok(receipt)
    }

    /// Decrypts a stored sample archive.
    ///
    /// # Errors
    ///
    /// Fails when the object is missing or was tampered with.
    pub fn read_sample(&self, tenant_id: &str, sample_id: &str) -> Result<Vec<u8>, ServerError> {
        let envelope = fs::read(self.object_path(tenant_id, sample_id))?;
        self.open_object(tenant_id, sample_id, &envelope)
    }

    /// Samples usable for training right now: unexpired, with active consent.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn trainable_samples(&self, tenant_id: &str, now: i64) -> Result<Vec<SampleRow>, ServerError> {
        let mut statement = self.connection.prepare(
            "SELECT sample_id,consent_id,received_at,duration_ms FROM samples
             WHERE tenant_id=?1 AND expires_at>?2 ORDER BY received_at, sample_id",
        )?;
        let rows = statement.query_map(params![tenant_id, now], |row| {
            Ok(SampleRow {
                sample_id: row.get(0)?,
                consent_id: row.get(1)?,
                received_at: row.get(2)?,
                duration_ms: to_u64(row.get(3)?),
            })
        })?;
        let mut samples = Vec::new();
        for row in rows {
            let row = row?;
            if self.consent_active(tenant_id, &row.consent_id, now)? {
                samples.push(row);
            }
        }
        Ok(samples)
    }

    // -- lineage ----------------------------------------------------------------

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn add_node(
        &self,
        node_id: &str,
        kind: &str,
        tenant_id: &str,
        parents: &[String],
        attributes: &serde_json::Value,
        now: i64,
    ) -> Result<(), ServerError> {
        self.connection.execute(
            "INSERT INTO nodes(node_id,kind,tenant_id,attributes,created_at) VALUES(?1,?2,?3,?4,?5)",
            params![node_id, kind, tenant_id, attributes.to_string(), now],
        )?;
        for parent in parents {
            self.connection.execute(
                "INSERT OR IGNORE INTO edges(parent,child) VALUES(?1,?2)",
                params![parent, node_id],
            )?;
        }
        Ok(())
    }

    /// Every node derived, directly or transitively, from `roots`.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn descendants(&self, roots: &[String]) -> Result<BTreeSet<String>, ServerError> {
        let mut found = BTreeSet::new();
        let mut frontier: Vec<String> = roots.to_vec();
        let mut statement = self.connection.prepare("SELECT child FROM edges WHERE parent=?1")?;
        while let Some(node) = frontier.pop() {
            let children: Vec<String> = statement
                .query_map(params![node], |row| row.get(0))?
                .collect::<Result<_, _>>()?;
            for child in children {
                if found.insert(child.clone()) {
                    frontier.push(child);
                }
            }
        }
        Ok(found)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn node_kinds(&self, nodes: &BTreeSet<String>) -> Result<BTreeMap<String, String>, ServerError> {
        let mut kinds = BTreeMap::new();
        for node in nodes {
            if let Some(kind) = self
                .connection
                .query_row("SELECT kind FROM nodes WHERE node_id=?1", params![node], |row| row.get::<_, String>(0))
                .optional()?
            {
                kinds.insert(node.clone(), kind);
            }
        }
        Ok(kinds)
    }

    // -- deletion ---------------------------------------------------------------

    /// Deletes samples and everything derived from them. Affected personalized
    /// models are revoked and deleted whole; no unlearning is claimed.
    /// Tombstones prevent a replayed upload or restored backup from bringing
    /// a sample back.
    ///
    /// Returns `(deleted_samples, deleted_artifacts)`.
    ///
    /// # Errors
    ///
    /// Fails on database or filesystem errors.
    pub fn delete_samples(&mut self, tenant_id: &str, sample_ids: &[String], now: i64) -> Result<(u32, u32), ServerError> {
        let derived = self.descendants(sample_ids)?;
        let kinds = self.node_kinds(&derived)?;
        let mut artifacts = 0_u32;
        for (node, kind) in &kinds {
            if kind == "delivery" {
                let path = self.delivery_path(tenant_id, node);
                if path.exists() {
                    fs::remove_file(path)?;
                }
                artifacts += 1;
            }
        }
        let mut samples = 0_u32;
        let transaction = self.connection.transaction()?;
        for sample_id in sample_ids {
            let removed = transaction.execute(
                "DELETE FROM samples WHERE sample_id=?1 AND tenant_id=?2",
                params![sample_id, tenant_id],
            )?;
            transaction.execute(
                "INSERT OR IGNORE INTO tombstones(sample_id,tenant_id,deleted_at) VALUES(?1,?2,?3)",
                params![sample_id, tenant_id, now],
            )?;
            transaction.execute(
                "UPDATE nodes SET deleted_at=?2 WHERE node_id=?1 AND deleted_at IS NULL",
                params![sample_id, now],
            )?;
            if removed > 0 {
                samples += 1;
            }
        }
        for node in derived {
            transaction.execute(
                "UPDATE nodes SET deleted_at=?2 WHERE node_id=?1 AND deleted_at IS NULL",
                params![node, now],
            )?;
            transaction.execute(
                "UPDATE deliveries SET revoked_at=?2 WHERE model_version=?1 AND revoked_at IS NULL",
                params![node, now],
            )?;
        }
        transaction.commit()?;
        for sample_id in sample_ids {
            let path = self.object_path(tenant_id, sample_id);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        Ok((samples, artifacts))
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn tenant_sample_ids(&self, tenant_id: &str) -> Result<Vec<String>, ServerError> {
        let mut statement = self
            .connection
            .prepare("SELECT sample_id FROM samples WHERE tenant_id=?1")?;
        let rows = statement.query_map(params![tenant_id], |row| row.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn sample_for_receipt(&self, tenant_id: &str, receipt_id: &str) -> Result<Option<String>, ServerError> {
        Ok(self
            .connection
            .query_row(
                "SELECT sample_id FROM receipts WHERE tenant_id=?1 AND receipt_id=?2 AND accepted=1",
                params![tenant_id, receipt_id],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// Everything for one tenant: samples, derived nodes, every delivery, and
    /// running training jobs. Consents are marked withdrawn.
    ///
    /// # Errors
    ///
    /// Fails on database or filesystem errors.
    pub fn delete_everything(&mut self, tenant_id: &str, now: i64) -> Result<(u32, u32), ServerError> {
        let samples = self.tenant_sample_ids(tenant_id)?;
        let (deleted, mut artifacts) = self.delete_samples(tenant_id, &samples, now)?;
        let versions: Vec<String> = {
            let mut statement = self
                .connection
                .prepare("SELECT model_version FROM deliveries WHERE tenant_id=?1")?;
            statement
                .query_map(params![tenant_id], |row| row.get(0))?
                .collect::<Result<_, _>>()?
        };
        for version in versions {
            let path = self.delivery_path(tenant_id, &version);
            if path.exists() {
                fs::remove_file(path)?;
                artifacts += 1;
            }
        }
        self.connection.execute(
            "UPDATE deliveries SET revoked_at=?2 WHERE tenant_id=?1 AND revoked_at IS NULL",
            params![tenant_id, now],
        )?;
        self.connection.execute(
            "UPDATE training_jobs SET state='cancelled', reason='deletion_requested', finished_at=?2
             WHERE tenant_id=?1 AND finished_at IS NULL",
            params![tenant_id, now],
        )?;
        self.connection.execute(
            "UPDATE consents SET withdrawn_at=?2 WHERE tenant_id=?1 AND withdrawn_at IS NULL",
            params![tenant_id, now],
        )?;
        Ok((deleted, artifacts))
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn log_deletion(&self, tenant_id: &str, request_id: &str, scope: &str, samples: u32, artifacts: u32, now: i64) -> Result<(), ServerError> {
        self.connection.execute(
            "INSERT OR REPLACE INTO deletion_log(tenant_id,request_id,scope,completed_at,deleted_samples,deleted_artifacts)
             VALUES(?1,?2,?3,?4,?5,?6)",
            params![tenant_id, request_id, scope, now, samples, artifacts],
        )?;
        Ok(())
    }

    /// Deletes expired samples and any object restored from a backup whose
    /// sample was tombstoned.
    ///
    /// # Errors
    ///
    /// Fails on database or filesystem errors.
    pub fn sweep(&mut self, now: i64) -> Result<u32, ServerError> {
        let expired: Vec<(String, String)> = {
            let mut statement = self
                .connection
                .prepare("SELECT tenant_id,sample_id FROM samples WHERE expires_at<=?1")?;
            statement
                .query_map(params![now], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<_, _>>()?
        };
        let mut removed = 0;
        for (tenant, sample) in expired {
            removed += self.delete_samples(&tenant, &[sample], now)?.0;
        }
        // Objects on disk without a live sample row (e.g. a restored backup).
        let objects = self.root.join("objects");
        if let Ok(tenants) = fs::read_dir(&objects) {
            for tenant in tenants.flatten() {
                let Ok(files) = fs::read_dir(tenant.path()) else { continue };
                for file in files.flatten() {
                    let name = file.file_name().to_string_lossy().into_owned();
                    let Some(sample_id) = name.strip_suffix(".bin") else { continue };
                    if !self.sample_exists(sample_id)? {
                        fs::remove_file(file.path())?;
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    // -- deliveries -----------------------------------------------------------

    #[must_use]
    pub fn delivery_path(&self, tenant_id: &str, version: &str) -> PathBuf {
        self.root.join("deliveries").join(tenant_id).join(format!("{version}.bin"))
    }

    /// # Errors
    ///
    /// Fails on database or filesystem errors.
    pub fn store_delivery(
        &self,
        tenant_id: &str,
        version: &str,
        signed: &dictation_models::delivery::SignedManifest,
        artifact: &[u8],
        now: i64,
    ) -> Result<(), ServerError> {
        let path = self.delivery_path(tenant_id, version);
        fs::create_dir_all(path.parent().unwrap_or(&self.root))?;
        write_private(&path, &self.seal_object(tenant_id, version, artifact)?)?;
        self.connection.execute(
            "INSERT INTO deliveries(model_version,tenant_id,manifest_json,signature_hex,created_at) VALUES(?1,?2,?3,?4,?5)",
            params![version, tenant_id, signed.manifest_json, signed.signature_hex, now],
        )?;
        Ok(())
    }

    /// Newest unrevoked delivery for the tenant.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn latest_delivery(&self, tenant_id: &str) -> Result<Option<(String, dictation_models::delivery::SignedManifest)>, ServerError> {
        Ok(self
            .connection
            .query_row(
                "SELECT model_version,manifest_json,signature_hex FROM deliveries
                 WHERE tenant_id=?1 AND revoked_at IS NULL ORDER BY created_at DESC LIMIT 1",
                params![tenant_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        dictation_models::delivery::SignedManifest {
                            manifest_json: row.get(1)?,
                            signature_hex: row.get(2)?,
                        },
                    ))
                },
            )
            .optional()?)
    }

    /// # Errors
    ///
    /// Fails when the artifact is missing, revoked, or another tenant's.
    pub fn delivery_artifact(&self, tenant_id: &str, version: &str) -> Result<Option<Vec<u8>>, ServerError> {
        let live = self
            .connection
            .query_row(
                "SELECT 1 FROM deliveries WHERE tenant_id=?1 AND model_version=?2 AND revoked_at IS NULL",
                params![tenant_id, version],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !live {
            return Ok(None);
        }
        let envelope = fs::read(self.delivery_path(tenant_id, version))?;
        self.open_object(tenant_id, version, &envelope).map(Some)
    }

    // -- training jobs -------------------------------------------------------

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn insert_training_job(&self, job_id: &str, tenant_id: &str, now: i64) -> Result<(), ServerError> {
        self.connection.execute(
            "INSERT INTO training_jobs(job_id,tenant_id,state,reason,created_at) VALUES(?1,?2,'running','',?3)",
            params![job_id, tenant_id, now],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn finish_training_job(&self, job_id: &str, state: &str, reason: &str, now: i64) -> Result<bool, ServerError> {
        Ok(self.connection.execute(
            "UPDATE training_jobs SET state=?2, reason=?3, finished_at=?4 WHERE job_id=?1 AND finished_at IS NULL",
            params![job_id, state, reason, now],
        )? > 0)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn training_job_state(&self, job_id: &str) -> Result<Option<(String, String)>, ServerError> {
        Ok(self
            .connection
            .query_row(
                "SELECT state,reason FROM training_jobs WHERE job_id=?1",
                params![job_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?)
    }
}

#[cfg(unix)]
fn restrict(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn restrict(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::{io::Write, os::unix::fs::OpenOptionsExt};
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(bytes)?;
        file.sync_all()
    }
    #[cfg(not(unix))]
    {
        fs::write(path, bytes)
    }
}
