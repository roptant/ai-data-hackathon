//! Consent records and the training-job queue (plan §9).
//!
//! Every state change is a guarded update that names the states it may leave,
//! so two workers cannot both claim a job and a withdrawal always wins over a
//! decision made earlier. Job rows hold identifiers and counters only; package
//! bytes live in an encrypted `eligible_package` scope.

use dictation_core::contribution::{ConsentRecord, JobState, Purpose};
use rusqlite::{OptionalExtension, Row, params};

use crate::{EncryptedStore, StoreError};

pub(crate) const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS consents (
  consent_id TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  purposes TEXT NOT NULL,
  granted_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  paused INTEGER NOT NULL DEFAULT 0,
  policy_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  consent_id TEXT NOT NULL,
  sample_id TEXT NOT NULL DEFAULT '',
  idempotency_key TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL DEFAULT '',
  duration_ms INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at INTEGER NOT NULL DEFAULT 0,
  session_started_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, next_attempt_at);
CREATE TABLE IF NOT EXISTS receipts (
  job_id TEXT NOT NULL,
  receipt_id TEXT NOT NULL,
  accepted INTEGER NOT NULL,
  received_at INTEGER NOT NULL,
  PRIMARY KEY (job_id, receipt_id)
);
CREATE TABLE IF NOT EXISTS deletion_requests (
  request_id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  sent_at INTEGER,
  completed_at INTEGER
);
CREATE TABLE IF NOT EXISTS contributed_hashes (
  content_hash TEXT PRIMARY KEY,
  contributed_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_volume (
  day INTEGER PRIMARY KEY,
  packages INTEGER NOT NULL,
  seconds INTEGER NOT NULL
);
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub job_id: String,
    pub session_id: String,
    pub state: JobState,
    pub reason: String,
    pub consent_id: String,
    pub sample_id: String,
    pub idempotency_key: String,
    pub content_hash: String,
    pub duration_ms: u64,
    pub attempts: u32,
    pub next_attempt_at: i64,
    pub session_started_at: i64,
    pub created_at: i64,
    pub updated_at: i64,
    pub expires_at: i64,
}

/// Optional columns written together with a transition.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JobUpdate<'a> {
    pub reason: Option<&'a str>,
    pub sample_id: Option<&'a str>,
    pub idempotency_key: Option<&'a str>,
    pub content_hash: Option<&'a str>,
    pub duration_ms: Option<u64>,
    pub attempts: Option<u32>,
    pub next_attempt_at: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeletionRequest<'a> {
    pub request_id: &'a str,
    pub scope: &'a str,
}

fn purposes_to_text(purposes: &[Purpose]) -> String {
    purposes
        .iter()
        .map(|purpose| purpose.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

fn purposes_from_text(text: &str) -> Vec<Purpose> {
    text.split(',')
        .filter_map(|value| match value {
            "customer_personalization" => Some(Purpose::CustomerPersonalization),
            "shared_model_training" => Some(Purpose::SharedModelTraining),
            _ => None,
        })
        .collect()
}

fn consent_from_row(row: &Row<'_>) -> rusqlite::Result<ConsentRecord> {
    Ok(ConsentRecord {
        consent_id: row.get(0)?,
        version: row.get(1)?,
        purposes: purposes_from_text(&row.get::<_, String>(2)?),
        granted_at: row.get(3)?,
        expires_at: row.get(4)?,
        revoked_at: row.get(5)?,
        paused: row.get::<_, i64>(6)? != 0,
        policy_version: row.get(7)?,
    })
}

const CONSENT_COLUMNS: &str =
    "consent_id,version,purposes,granted_at,expires_at,revoked_at,paused,policy_version";
const JOB_COLUMNS: &str = "job_id,session_id,state,reason,consent_id,sample_id,idempotency_key,content_hash,duration_ms,attempts,next_attempt_at,session_started_at,created_at,updated_at,expires_at";

fn job_from_row(row: &Row<'_>) -> rusqlite::Result<Job> {
    let state: String = row.get(2)?;
    Ok(Job {
        job_id: row.get(0)?,
        session_id: row.get(1)?,
        state: JobState::parse(&state).unwrap_or(JobState::Rejected),
        reason: row.get(3)?,
        consent_id: row.get(4)?,
        sample_id: row.get(5)?,
        idempotency_key: row.get(6)?,
        content_hash: row.get(7)?,
        duration_ms: u64::try_from(row.get::<_, i64>(8)?).unwrap_or(0),
        attempts: u32::try_from(row.get::<_, i64>(9)?).unwrap_or(u32::MAX),
        next_attempt_at: row.get(10)?,
        session_started_at: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
        expires_at: row.get(14)?,
    })
}

fn to_i64(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::InvalidMetadata)
}

/// UTC day number used for daily volume caps.
#[must_use]
pub const fn day_of(timestamp: i64) -> i64 {
    timestamp.div_euclid(86_400)
}

impl EncryptedStore {
    // -- consent ---------------------------------------------------------

    /// Records an explicit opt-in and revokes any earlier active grant.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn insert_consent(&mut self, record: &ConsentRecord) -> Result<(), StoreError> {
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "UPDATE consents SET revoked_at=?1 WHERE revoked_at IS NULL",
            params![record.granted_at],
        )?;
        transaction.execute(
            &format!("INSERT INTO consents({CONSENT_COLUMNS}) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)"),
            params![
                record.consent_id,
                record.version,
                purposes_to_text(&record.purposes),
                record.granted_at,
                record.expires_at,
                record.revoked_at,
                i64::from(record.paused),
                record.policy_version,
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// The most recent unrevoked consent, if any (it may be expired or paused).
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn current_consent(&self) -> Result<Option<ConsentRecord>, StoreError> {
        Ok(self
            .connection
            .query_row(
                &format!(
                    "SELECT {CONSENT_COLUMNS} FROM consents WHERE revoked_at IS NULL
                     ORDER BY granted_at DESC LIMIT 1"
                ),
                [],
                consent_from_row,
            )
            .optional()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn consent(&self, consent_id: &str) -> Result<Option<ConsentRecord>, StoreError> {
        Ok(self
            .connection
            .query_row(
                &format!("SELECT {CONSENT_COLUMNS} FROM consents WHERE consent_id=?1"),
                params![consent_id],
                consent_from_row,
            )
            .optional()?)
    }

    /// Contribution history without content.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn consent_history(&self) -> Result<Vec<ConsentRecord>, StoreError> {
        let mut statement = self.connection.prepare(&format!(
            "SELECT {CONSENT_COLUMNS} FROM consents ORDER BY granted_at DESC"
        ))?;
        let rows = statement.query_map([], consent_from_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn set_consent_paused(&self, consent_id: &str, paused: bool) -> Result<bool, StoreError> {
        Ok(self.connection.execute(
            "UPDATE consents SET paused=?2 WHERE consent_id=?1 AND revoked_at IS NULL",
            params![consent_id, i64::from(paused)],
        )? > 0)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn revoke_consent(&self, consent_id: &str, now: i64) -> Result<bool, StoreError> {
        Ok(self.connection.execute(
            "UPDATE consents SET revoked_at=?2 WHERE consent_id=?1 AND revoked_at IS NULL",
            params![consent_id, now],
        )? > 0)
    }

    // -- jobs ------------------------------------------------------------

    /// # Errors
    ///
    /// Fails on duplicate session jobs or database errors.
    pub fn insert_job(
        &self,
        job_id: &str,
        session_id: &str,
        consent_id: &str,
        session_started_at: i64,
        now: i64,
        expires_at: i64,
    ) -> Result<(), StoreError> {
        self.connection.execute(
            "INSERT INTO jobs(job_id,session_id,state,consent_id,session_started_at,created_at,updated_at,expires_at)
             VALUES(?1,?2,?3,?4,?5,?6,?6,?7)",
            params![
                job_id,
                session_id,
                JobState::LocalPending.as_str(),
                consent_id,
                session_started_at,
                now,
                expires_at
            ],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn job(&self, job_id: &str) -> Result<Option<Job>, StoreError> {
        Ok(self
            .connection
            .query_row(
                &format!("SELECT {JOB_COLUMNS} FROM jobs WHERE job_id=?1"),
                params![job_id],
                job_from_row,
            )
            .optional()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn jobs_in(&self, states: &[JobState]) -> Result<Vec<Job>, StoreError> {
        let mut statement = self.connection.prepare(&format!(
            "SELECT {JOB_COLUMNS} FROM jobs ORDER BY created_at"
        ))?;
        let rows = statement.query_map([], job_from_row)?;
        let mut jobs = Vec::new();
        for row in rows {
            let job = row?;
            if states.contains(&job.state) {
                jobs.push(job);
            }
        }
        Ok(jobs)
    }

    /// Moves a job to `target` only from a state the transition table allows.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::TransitionConflict`] when another actor moved the
    /// job first or the transition is not allowed.
    pub fn transition_job(
        &self,
        job_id: &str,
        target: JobState,
        update: &JobUpdate<'_>,
        now: i64,
    ) -> Result<Job, StoreError> {
        let allowed: Vec<&str> = JobState::predecessors(target)
            .into_iter()
            .map(JobState::as_str)
            .collect();
        if allowed.is_empty() {
            return Err(StoreError::TransitionConflict);
        }
        let placeholders = (0..allowed.len())
            .map(|index| format!("?{}", index + 11))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "UPDATE jobs SET state=?2, updated_at=?3,
               reason=COALESCE(?4,reason), sample_id=COALESCE(?5,sample_id),
               idempotency_key=COALESCE(?6,idempotency_key), content_hash=COALESCE(?7,content_hash),
               duration_ms=COALESCE(?8,duration_ms), attempts=COALESCE(?9,attempts),
               next_attempt_at=COALESCE(?10,next_attempt_at)
             WHERE job_id=?1 AND state IN ({placeholders})"
        );
        let duration = update.duration_ms.map(to_i64).transpose()?;
        let attempts = update.attempts.map(i64::from);
        let mut values: Vec<Box<dyn rusqlite::ToSql>> = vec![
            Box::new(job_id.to_owned()),
            Box::new(target.as_str()),
            Box::new(now),
            Box::new(update.reason.map(str::to_owned)),
            Box::new(update.sample_id.map(str::to_owned)),
            Box::new(update.idempotency_key.map(str::to_owned)),
            Box::new(update.content_hash.map(str::to_owned)),
            Box::new(duration),
            Box::new(attempts),
            Box::new(update.next_attempt_at),
        ];
        values.extend(allowed.iter().map(|state| Box::new(*state) as Box<dyn rusqlite::ToSql>));
        let changed = self.connection.execute(
            &sql,
            rusqlite::params_from_iter(values.iter().map(AsRef::as_ref)),
        )?;
        if changed == 0 {
            return Err(StoreError::TransitionConflict);
        }
        self.job(job_id)?.ok_or(StoreError::TransitionConflict)
    }

    /// Defers an eligible job without changing its state, e.g. when the daily
    /// volume cap is reached.
    ///
    /// # Errors
    ///
    /// Returns a conflict when the job is no longer eligible.
    pub fn reschedule_job(
        &self,
        job_id: &str,
        next_attempt_at: i64,
        reason: &str,
        now: i64,
    ) -> Result<(), StoreError> {
        let changed = self.connection.execute(
            "UPDATE jobs SET next_attempt_at=?2, reason=?3, updated_at=?4 WHERE job_id=?1 AND state=?5",
            params![job_id, next_attempt_at, reason, now, JobState::Eligible.as_str()],
        )?;
        if changed == 0 {
            return Err(StoreError::TransitionConflict);
        }
        Ok(())
    }

    /// Removes a terminal job row once its payloads are gone.
    ///
    /// # Errors
    ///
    /// Returns a conflict for an active job.
    pub fn purge_job(&self, job_id: &str) -> Result<(), StoreError> {
        let Some(job) = self.job(job_id)? else {
            return Ok(());
        };
        if !job.state.is_terminal() {
            return Err(StoreError::TransitionConflict);
        }
        self.connection
            .execute("DELETE FROM jobs WHERE job_id=?1", params![job_id])?;
        Ok(())
    }

    // -- receipts, deletion, volume, duplicates ---------------------------

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn insert_receipt(
        &self,
        job_id: &str,
        receipt_id: &str,
        accepted: bool,
        now: i64,
    ) -> Result<(), StoreError> {
        self.connection.execute(
            "INSERT OR IGNORE INTO receipts(job_id,receipt_id,accepted,received_at) VALUES(?1,?2,?3,?4)",
            params![job_id, receipt_id, i64::from(accepted), now],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn insert_deletion_request(
        &self,
        request: DeletionRequest<'_>,
        now: i64,
    ) -> Result<(), StoreError> {
        self.connection.execute(
            "INSERT OR IGNORE INTO deletion_requests(request_id,scope,created_at) VALUES(?1,?2,?3)",
            params![request.request_id, request.scope, now],
        )?;
        Ok(())
    }

    /// Deletion requests not yet confirmed by the server, oldest first.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn pending_deletion_requests(&self) -> Result<Vec<(String, String)>, StoreError> {
        let mut statement = self.connection.prepare(
            "SELECT request_id,scope FROM deletion_requests WHERE completed_at IS NULL ORDER BY created_at",
        )?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn complete_deletion_request(&self, request_id: &str, now: i64) -> Result<(), StoreError> {
        self.connection.execute(
            "UPDATE deletion_requests SET completed_at=?2, sent_at=COALESCE(sent_at,?2) WHERE request_id=?1",
            params![request_id, now],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn record_contributed_hash(&self, content_hash: &str, now: i64) -> Result<(), StoreError> {
        self.connection.execute(
            "INSERT OR IGNORE INTO contributed_hashes(content_hash,contributed_at) VALUES(?1,?2)",
            params![content_hash, now],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn contributed_hashes(&self) -> Result<std::collections::BTreeSet<String>, StoreError> {
        let mut statement = self
            .connection
            .prepare("SELECT content_hash FROM contributed_hashes")?;
        let rows = statement.query_map([], |row| row.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// # Errors
    ///
    /// Fails on database errors.
    pub fn add_volume(&self, now: i64, packages: u32, seconds: u64) -> Result<(), StoreError> {
        self.connection.execute(
            "INSERT INTO daily_volume(day,packages,seconds) VALUES(?1,?2,?3)
             ON CONFLICT(day) DO UPDATE SET packages=packages+excluded.packages, seconds=seconds+excluded.seconds",
            params![day_of(now), i64::from(packages), to_i64(seconds)?],
        )?;
        Ok(())
    }

    /// Packages and seconds acknowledged on the day containing `now`.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn volume_today(&self, now: i64) -> Result<(u32, u64), StoreError> {
        let row: Option<(i64, i64)> = self
            .connection
            .query_row(
                "SELECT packages,seconds FROM daily_volume WHERE day=?1",
                params![day_of(now)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        Ok(row.map_or((0, 0), |(packages, seconds)| {
            (
                u32::try_from(packages).unwrap_or(u32::MAX),
                u64::try_from(seconds).unwrap_or(0),
            )
        }))
    }

    /// Marks cancellable jobs deleted and deletes their package scopes in one
    /// transaction. Returns the cancelled jobs so the caller can delete their
    /// raw-session scopes, which live in the separate sessions database.
    ///
    /// # Errors
    ///
    /// Fails on database errors.
    pub fn cancel_all_jobs(&mut self, reason: &str, now: i64) -> Result<Vec<Job>, StoreError> {
        let cancellable: Vec<Job> = self
            .jobs_in(&JobState::ALL)?
            .into_iter()
            .filter(|job| job.state.is_cancellable())
            .collect();
        let transaction = self.connection.transaction()?;
        for job in &cancellable {
            transaction.execute(
                "UPDATE jobs SET state=?2, reason=?3, updated_at=?4 WHERE job_id=?1",
                params![job.job_id, JobState::Deleted.as_str(), reason, now],
            )?;
            transaction.execute(
                "DELETE FROM encryption_scopes WHERE scope_id=?1",
                params![crate::package_scope_id(&job.job_id)],
            )?;
        }
        transaction.commit()?;
        Ok(cancellable)
    }
}

#[cfg(test)]
mod tests {
    use dictation_core::contribution::CONSENT_VERSION;

    use super::*;
    use crate::test_support::store;

    fn record(id: &str, granted_at: i64) -> ConsentRecord {
        ConsentRecord {
            consent_id: id.to_owned(),
            version: CONSENT_VERSION.to_owned(),
            purposes: vec![Purpose::CustomerPersonalization],
            granted_at,
            expires_at: granted_at + 1_000,
            revoked_at: None,
            paused: false,
            policy_version: "p".to_owned(),
        }
    }

    #[test]
    fn a_new_grant_revokes_the_previous_one() {
        let mut store = store();
        store.insert_consent(&record("c1", 10)).unwrap();
        store.insert_consent(&record("c2", 20)).unwrap();
        assert_eq!(store.current_consent().unwrap().unwrap().consent_id, "c2");
        assert_eq!(store.consent("c1").unwrap().unwrap().revoked_at, Some(20));
        assert_eq!(store.consent_history().unwrap().len(), 2);
    }

    #[test]
    fn guarded_transitions_reject_conflicts() {
        let store = store();
        store.insert_job("j1", "s1", "c1", 5, 10, 100).unwrap();
        let job = store
            .transition_job("j1", JobState::Analyzing, &JobUpdate::default(), 11)
            .unwrap();
        assert_eq!(job.state, JobState::Analyzing);
        assert!(matches!(
            store.transition_job("j1", JobState::Acknowledged, &JobUpdate::default(), 12),
            Err(StoreError::TransitionConflict)
        ));
        let job = store
            .transition_job(
                "j1",
                JobState::Rejected,
                &JobUpdate {
                    reason: Some("schema_violation"),
                    ..JobUpdate::default()
                },
                13,
            )
            .unwrap();
        assert_eq!(job.reason, "schema_violation");
        assert!(matches!(
            store.transition_job("j1", JobState::Analyzing, &JobUpdate::default(), 14),
            Err(StoreError::TransitionConflict)
        ));
    }

    #[test]
    fn withdrawal_cancels_active_jobs_and_their_payloads() {
        let mut store = store();
        store.insert_job("j1", "s1", "c1", 5, 10, 100).unwrap();
        store.insert_job("j2", "s2", "c1", 5, 10, 100).unwrap();
        store
            .transition_job("j2", JobState::Rejected, &JobUpdate::default(), 11)
            .unwrap();
        store
            .create_scope(
                &crate::package_scope_id("j1"),
                crate::DataClass::EligiblePackage,
                100,
                1,
            )
            .unwrap();
        let cancelled = store.cancel_all_jobs("consent_withdrawn", 20).unwrap();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0].session_id, "s1");
        assert_eq!(store.job("j1").unwrap().unwrap().state, JobState::Deleted);
        assert_eq!(store.job("j2").unwrap().unwrap().state, JobState::Rejected);
        assert!(!store.delete_scope(&crate::package_scope_id("j1")).unwrap());
    }

    #[test]
    fn volume_and_duplicates_are_tracked() {
        let store = store();
        store.add_volume(86_400 * 3 + 5, 1, 12).unwrap();
        store.add_volume(86_400 * 3 + 50, 1, 8).unwrap();
        assert_eq!(store.volume_today(86_400 * 3 + 100).unwrap(), (2, 20));
        assert_eq!(store.volume_today(86_400 * 4).unwrap(), (0, 0));
        store.record_contributed_hash("abc", 1).unwrap();
        assert!(store.contributed_hashes().unwrap().contains("abc"));
    }
}
