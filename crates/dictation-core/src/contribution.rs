//! Consent, eligibility, and training-job policy (plan §9, §11, §12).
//!
//! Contribution is off by default. [`check_session`] is the only function that
//! can say a session may be contributed, and it answers no unless every
//! condition holds. The job state table is data so the store and its tests
//! read the same rules and an unlisted transition is impossible.

use std::fmt;

/// Version of the consent text shown to the user; recorded with the grant.
pub const CONSENT_VERSION: &str = "consent-2026.09-en-1";

/// Provisional: consent is re-confirmed yearly rather than assumed permanent.
pub const DEFAULT_CONSENT_DAYS: i64 = 365;

/// Recorded evaluation status of the automatic-upload gates (plan §12). No
/// recall benchmark with the required sample sizes has been run, so this is
/// false and production upload is refused rather than merely discouraged.
pub const UPLOAD_GATES_MET: bool = false;

/// Eligibility versions a server admits. Bumped when detection or removal
/// policy changes so stale clients cannot upload under an old policy.
pub const ELIGIBILITY_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    CustomerPersonalization,
    /// Listed to make the separation explicit. Not grantable in this release.
    SharedModelTraining,
}

impl Purpose {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CustomerPersonalization => "customer_personalization",
            Self::SharedModelTraining => "shared_model_training",
        }
    }

    #[must_use]
    pub const fn grantable(self) -> bool {
        matches!(self, Self::CustomerPersonalization)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentRecord {
    pub consent_id: String,
    pub version: String,
    pub purposes: Vec<Purpose>,
    pub granted_at: i64,
    pub expires_at: i64,
    pub revoked_at: Option<i64>,
    pub paused: bool,
    pub policy_version: String,
}

impl ConsentRecord {
    #[must_use]
    pub const fn is_active(&self, now: i64) -> bool {
        self.revoked_at.is_none() && self.expires_at > now
    }

    #[must_use]
    pub const fn is_usable(&self, now: i64) -> bool {
        self.is_active(now) && !self.paused
    }

    #[must_use]
    pub fn covers(&self, purpose: Purpose) -> bool {
        self.purposes.contains(&purpose)
    }
}

/// Where upload credentials point. Only a nonproduction server may receive
/// data while the evaluation gates are unmet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadTarget {
    Disabled,
    NonProduction,
    Production,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    NoConsent,
    ConsentPaused,
    ConsentExpiredOrRevoked,
    ConsentVersionOutdated,
    PurposeNotGranted,
    SessionBeforeConsent,
    SessionOptedOut,
    LanguageNotValidated,
    UploadGatesUnmet,
    UploadDisabled,
}

impl Refusal {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::NoConsent => "no_consent",
            Self::ConsentPaused => "consent_paused",
            Self::ConsentExpiredOrRevoked => "consent_expired_or_revoked",
            Self::ConsentVersionOutdated => "consent_version_outdated",
            Self::PurposeNotGranted => "purpose_not_granted",
            Self::SessionBeforeConsent => "session_before_consent",
            Self::SessionOptedOut => "session_opted_out",
            Self::LanguageNotValidated => "language_not_validated",
            Self::UploadGatesUnmet => "upload_gates_unmet",
            Self::UploadDisabled => "upload_disabled",
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Facts about one session relevant to contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionFacts<'a> {
    pub started_at: i64,
    pub opted_out: bool,
    pub language: &'a str,
}

/// Validated languages for automatic upload (plan §1: English initially).
pub const VALIDATED_LANGUAGES: [&str; 1] = ["en"];

/// Whether consent currently permits contributing one session.
///
/// # Errors
///
/// Returns the first unmet condition.
pub fn check_session(
    consent: Option<&ConsentRecord>,
    session: SessionFacts<'_>,
    target: UploadTarget,
    now: i64,
) -> Result<String, Refusal> {
    let consent = recheck(consent, target, now)?;
    if session.started_at < consent.granted_at {
        // Enabling contribution never sweeps up earlier sessions.
        return Err(Refusal::SessionBeforeConsent);
    }
    if session.opted_out {
        return Err(Refusal::SessionOptedOut);
    }
    if !VALIDATED_LANGUAGES.contains(&session.language) {
        return Err(Refusal::LanguageNotValidated);
    }
    Ok(consent.consent_id.clone())
}

/// Recheck at every boundary: enqueue, eligibility, transfer.
///
/// # Errors
///
/// Returns the first unmet condition.
pub fn recheck(
    consent: Option<&ConsentRecord>,
    target: UploadTarget,
    now: i64,
) -> Result<&ConsentRecord, Refusal> {
    match target {
        UploadTarget::Disabled => return Err(Refusal::UploadDisabled),
        UploadTarget::Production if !UPLOAD_GATES_MET => return Err(Refusal::UploadGatesUnmet),
        _ => {}
    }
    let consent = consent.ok_or(Refusal::NoConsent)?;
    if !consent.is_active(now) {
        return Err(Refusal::ConsentExpiredOrRevoked);
    }
    if consent.paused {
        return Err(Refusal::ConsentPaused);
    }
    if consent.version != CONSENT_VERSION {
        return Err(Refusal::ConsentVersionOutdated);
    }
    if !consent.covers(Purpose::CustomerPersonalization) {
        return Err(Refusal::PurposeNotGranted);
    }
    Ok(consent)
}

/// The disclosure shown before opt-in. Returned as data so the UI cannot
/// quietly present a shorter version than the one recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disclosure {
    pub version: &'static str,
    pub sections: &'static [(&'static str, &'static str)],
    pub gates_met: bool,
}

#[must_use]
pub const fn disclosure() -> Disclosure {
    Disclosure {
        version: CONSENT_VERSION,
        sections: &[
            (
                "Destination",
                "A customer-isolated training service hosted in the EU/EEA. Your examples are never mixed with other customers' data or used for a shared model.",
            ),
            (
                "What is sent",
                "Retained audio clips and exactly the words spoken in them, sample rate, language, duration and quality metrics, model and policy versions, and a consent reference. No filenames, device names, application titles, prompts, or the removed passages.",
            ),
            (
                "Purpose",
                "Improving recognition of your own speech, accent, and vocabulary. Not voice synthesis, voice cloning, speaker identification, or emotion inference.",
            ),
            (
                "Automatic uploads",
                "After you opt in, eligible clips upload automatically without per-recording review. You can mark any session \"do not contribute\".",
            ),
            (
                "What is removed first",
                "Sentences that local detectors or the local privacy model flag as sensitive, the utterances around them, and anything the detectors are unsure about. Detection can miss things, especially words the recognizer got wrong.",
            ),
            (
                "Your voice is still personal data",
                "A voice can remain identifiable from its sound, and context can identify someone without naming them. Removing personal words reduces risk; it is not anonymization.",
            ),
            (
                "Retention",
                "Local raw recordings: deleted after processing, at most 24 hours. Local upload packages: deleted after upload, at most 7 days. Server training examples: up to 30 days. Your personalized model: while personalization is enabled.",
            ),
            (
                "Pause, withdraw, delete",
                "Pause or withdraw at any time. Withdrawal cancels queued and in-flight uploads, stops new training, and starts deletion of received examples and your personalized model. Deleting examples does not provably remove their influence from a model already trained, so the personalized model itself is deleted.",
            ),
            (
                "Saying no",
                "Refusing leaves local dictation fully functional, including local vocabulary. Sessions recorded before you opt in are never uploaded.",
            ),
        ],
        gates_met: UPLOAD_GATES_MET,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobState {
    LocalPending,
    Analyzing,
    Building,
    Eligible,
    Uploading,
    Acknowledged,
    Rejected,
    Deleted,
}

impl JobState {
    pub const ALL: [Self; 8] = [
        Self::LocalPending,
        Self::Analyzing,
        Self::Building,
        Self::Eligible,
        Self::Uploading,
        Self::Acknowledged,
        Self::Rejected,
        Self::Deleted,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalPending => "local_pending",
            Self::Analyzing => "analyzing",
            Self::Building => "building",
            Self::Eligible => "eligible",
            Self::Uploading => "uploading",
            Self::Acknowledged => "acknowledged",
            Self::Rejected => "rejected",
            Self::Deleted => "deleted",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|state| state.as_str() == value)
    }

    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Acknowledged | Self::Rejected | Self::Deleted)
    }

    /// States withdrawal must cancel.
    #[must_use]
    pub const fn is_cancellable(self) -> bool {
        !self.is_terminal()
    }

    /// `Uploading -> Eligible` is the retry edge; `Analyzing`/`Building ->
    /// LocalPending` is the restart edge used when new dictation preempts a
    /// privacy job (plan §4); every state may be deleted.
    #[must_use]
    pub const fn may_transition(self, target: Self) -> bool {
        use JobState::{
            Acknowledged, Analyzing, Building, Deleted, Eligible, LocalPending, Rejected,
            Uploading,
        };
        matches!(
            (self, target),
            (LocalPending, Analyzing | Rejected | Deleted)
                | (Analyzing, Building | LocalPending | Rejected | Deleted)
                | (Building, Eligible | LocalPending | Rejected | Deleted)
                | (Eligible, Uploading | Rejected | Deleted)
                | (Uploading, Acknowledged | Eligible | Rejected | Deleted)
                | (Acknowledged | Rejected, Deleted)
        )
    }

    /// States a guarded update accepts when moving to `target`.
    #[must_use]
    pub fn predecessors(target: Self) -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|state| state.may_transition(target))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrySettings {
    pub max_attempts: u32,
    pub initial_backoff_seconds: i64,
    pub max_backoff_seconds: i64,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_attempts: 8,
            initial_backoff_seconds: 30,
            max_backoff_seconds: 6 * 3600,
        }
    }
}

/// Exponential backoff for attempt `attempts` (1-based), or `None` when the
/// job should be rejected instead of retried.
#[must_use]
pub fn retry_delay(attempts: u32, retriable: bool, settings: RetrySettings) -> Option<i64> {
    if !retriable || attempts >= settings.max_attempts {
        return None;
    }
    let exponent = attempts.saturating_sub(1).min(30);
    Some(
        settings
            .initial_backoff_seconds
            .saturating_mul(1_i64 << exponent)
            .min(settings.max_backoff_seconds),
    )
}

/// Collection caps (plan §2: cap collection volume). Provisional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolumeCaps {
    pub max_packages_per_day: u32,
    pub max_seconds_per_day: u64,
}

impl Default for VolumeCaps {
    fn default() -> Self {
        Self {
            max_packages_per_day: 200,
            max_seconds_per_day: 3_600,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn consent(granted_at: i64) -> ConsentRecord {
        ConsentRecord {
            consent_id: "consent-1".to_owned(),
            version: CONSENT_VERSION.to_owned(),
            purposes: vec![Purpose::CustomerPersonalization],
            granted_at,
            expires_at: granted_at + 1_000,
            revoked_at: None,
            paused: false,
            policy_version: "p".to_owned(),
        }
    }

    fn facts(started_at: i64) -> SessionFacts<'static> {
        SessionFacts {
            started_at,
            opted_out: false,
            language: "en",
        }
    }

    #[test]
    fn contribution_is_refused_without_every_condition() {
        let record = consent(100);
        assert_eq!(
            check_session(None, facts(150), UploadTarget::NonProduction, 150),
            Err(Refusal::NoConsent)
        );
        assert_eq!(
            check_session(Some(&record), facts(50), UploadTarget::NonProduction, 150),
            Err(Refusal::SessionBeforeConsent)
        );
        assert_eq!(
            check_session(Some(&record), facts(150), UploadTarget::Disabled, 150),
            Err(Refusal::UploadDisabled)
        );
        assert_eq!(
            check_session(
                Some(&record),
                SessionFacts {
                    language: "fi",
                    ..facts(150)
                },
                UploadTarget::NonProduction,
                150
            ),
            Err(Refusal::LanguageNotValidated)
        );
        assert_eq!(
            check_session(Some(&record), facts(150), UploadTarget::NonProduction, 150),
            Ok("consent-1".to_owned())
        );
    }

    #[test]
    fn production_upload_is_refused_until_gates_are_met() {
        assert!(!UPLOAD_GATES_MET);
        assert_eq!(
            recheck(Some(&consent(0)), UploadTarget::Production, 10),
            Err(Refusal::UploadGatesUnmet)
        );
    }

    #[test]
    fn pause_revocation_expiry_and_version_change_refuse() {
        let mut record = consent(0);
        record.paused = true;
        assert_eq!(
            recheck(Some(&record), UploadTarget::NonProduction, 10),
            Err(Refusal::ConsentPaused)
        );
        let mut record = consent(0);
        record.revoked_at = Some(5);
        assert_eq!(
            recheck(Some(&record), UploadTarget::NonProduction, 10),
            Err(Refusal::ConsentExpiredOrRevoked)
        );
        assert_eq!(
            recheck(Some(&consent(0)), UploadTarget::NonProduction, 5_000),
            Err(Refusal::ConsentExpiredOrRevoked)
        );
        let mut record = consent(0);
        record.version = "old".to_owned();
        assert_eq!(
            recheck(Some(&record), UploadTarget::NonProduction, 10),
            Err(Refusal::ConsentVersionOutdated)
        );
    }

    #[test]
    fn shared_model_training_is_not_grantable() {
        assert!(!Purpose::SharedModelTraining.grantable());
        let mut record = consent(0);
        record.purposes = vec![Purpose::SharedModelTraining];
        assert_eq!(
            recheck(Some(&record), UploadTarget::NonProduction, 10),
            Err(Refusal::PurposeNotGranted)
        );
    }

    #[test]
    fn job_transitions_follow_the_plan_diagram() {
        use JobState::*;
        assert!(LocalPending.may_transition(Analyzing));
        assert!(Uploading.may_transition(Eligible));
        assert!(!Acknowledged.may_transition(Eligible));
        assert!(!Deleted.may_transition(Eligible));
        assert!(!Rejected.may_transition(Uploading));
        for state in JobState::ALL {
            assert_eq!(state.may_transition(Deleted), state != Deleted);
            assert_eq!(JobState::parse(state.as_str()), Some(state));
        }
        assert_eq!(JobState::predecessors(Acknowledged), vec![Uploading]);
    }

    #[test]
    fn backoff_grows_and_is_bounded() {
        let settings = RetrySettings::default();
        assert_eq!(retry_delay(1, true, settings), Some(30));
        assert_eq!(retry_delay(2, true, settings), Some(60));
        assert_eq!(retry_delay(7, true, settings), Some(1_920));
        assert_eq!(retry_delay(8, true, settings), None);
        assert_eq!(retry_delay(1, false, settings), None);
    }

    #[test]
    fn disclosure_states_identifiability_and_gates() {
        let text = disclosure();
        assert!(text.sections.iter().any(|(_, body)| body.contains("not anonymization")));
        assert!(!text.gates_met);
    }
}
