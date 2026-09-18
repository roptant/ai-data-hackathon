"""Consent, pause, withdrawal and deletion requests (plan sections 9 and 11).

Contribution is off by default.  This module is the only place that can say
"yes, this session may be uploaded", and it answers no unless every condition
holds:

* an active, unexpired, unpaused consent record exists,
* its version matches the consent text the user was actually shown,
* the session started **after** consent was granted - enabling contribution
  later never sweeps up earlier sessions,
* the session was not marked "do not contribute",
* the evaluation gates of plan section 12 are recorded as met, or the operator
  explicitly opted into a non-production server.

The last condition is why :data:`dictation.version.UPLOAD_GATES_MET` is
``False`` in this implementation: no recall benchmark has been run, so
production upload is refused rather than merely discouraged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from dictation.errors import ConsentError
from dictation.logging_ import events
from dictation.store.db import Database
from dictation.types import new_id
from dictation.version import CONSENT_VERSION, POLICY_VERSION, UPLOAD_GATES_MET

#: Provisional: consent is re-confirmed yearly rather than assumed permanent.
DEFAULT_CONSENT_DAYS = 365


class Purpose(StrEnum):
    """Purposes a user can consent to, separately.

    Customer-specific personalization is kept apart from any future
    shared-model contribution (plan section 9); there is no code path that
    treats consent to one as consent to the other.
    """

    CUSTOMER_PERSONALIZATION = "customer_personalization"
    #: Not implemented, and not grantable in this release.  Listed so the
    #: separation is explicit rather than implied.
    SHARED_MODEL_TRAINING = "shared_model_training"


GRANTABLE_PURPOSES = frozenset({Purpose.CUSTOMER_PERSONALIZATION})


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    consent_id: str
    version: str
    purposes: tuple[str, ...]
    granted_at: float
    expires_at: float
    revoked_at: float | None = None
    paused: bool = False
    policy_version: str = POLICY_VERSION

    def is_active(self, now: float) -> bool:
        return self.revoked_at is None and self.expires_at > now

    def is_usable(self, now: float) -> bool:
        return self.is_active(now) and not self.paused

    def covers(self, purpose: Purpose) -> bool:
        return str(purpose) in self.purposes

    def summary(self) -> dict[str, object]:
        """Contribution history without any raw content (plan section 9)."""
        return {
            "consent_id": self.consent_id,
            "version": self.version,
            "purposes": list(self.purposes),
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
            "paused": self.paused,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """Why a session may or may not be contributed."""

    allowed: bool
    reason: str = ""
    consent_id: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class ConsentManager:
    """Reads and writes consent state, and answers eligibility questions."""

    def __init__(
        self,
        database: Database,
        *,
        consent_version: str = CONSENT_VERSION,
        allow_unvalidated_upload: bool = False,
    ) -> None:
        self.db = database
        self.consent_version = consent_version
        #: Set only for an isolated non-production server (plan section 12,
        #: milestone 4).  Never for production credentials.
        self.allow_unvalidated_upload = allow_unvalidated_upload

    # -- disclosure ----------------------------------------------------------

    def disclosure(self) -> dict[str, object]:
        """What must be shown before opt-in (plan section 9).

        Returned as data so the UI cannot quietly present a shorter version:
        the same text is what gets recorded with the consent version.
        """
        return {
            "version": self.consent_version,
            "destination": "customer-isolated training service, EU/EEA hosting",
            "data_types": [
                "retained audio clips with the words spoken in them",
                "sample rate, language, duration and quality metrics",
                "model and policy versions, and a consent reference",
            ],
            "purpose": "improving recognition of your own speech, accent and vocabulary",
            "retention": "training examples up to 30 days; personalized model while enabled",
            "automatic": "after opt-in, eligible clips upload without per-recording review",
            "identifiability": (
                "a voice can remain identifiable from its acoustic characteristics, and "
                "context can identify someone without naming them. Removing personal words "
                "reduces risk; it is not anonymisation."
            ),
            "not_included": [
                "sentences detected as sensitive, and the utterances around them",
                "anything the detectors were unsure about",
                "voice synthesis, voice cloning, speaker identification or emotion inference",
            ],
            "withdrawal": (
                "pause or withdraw at any time; withdrawal cancels queued and in-flight "
                "uploads, stops new training and starts deletion of received artifacts"
            ),
            "local_only_alternative": (
                "refusing contribution leaves local dictation fully functional, including "
                "local vocabulary corrections"
            ),
            "gates_met": UPLOAD_GATES_MET,
        }

    # -- state ---------------------------------------------------------------

    def active(self, *, now: float | None = None) -> ConsentRecord | None:
        stamp = time.time() if now is None else now
        row = self.db.active_consent(stamp)
        return _record(row) if row else None

    def history(self) -> tuple[ConsentRecord, ...]:
        return tuple(_record(row) for row in self.db.consent_history())

    def grant(
        self,
        purposes: tuple[Purpose, ...] = (Purpose.CUSTOMER_PERSONALIZATION,),
        *,
        now: float | None = None,
        days: int = DEFAULT_CONSENT_DAYS,
    ) -> ConsentRecord:
        """Record an explicit opt-in."""
        stamp = time.time() if now is None else now
        if not purposes:
            raise ConsentError("at least one purpose must be granted")
        for purpose in purposes:
            if purpose not in GRANTABLE_PURPOSES:
                raise ConsentError(
                    f"{purpose} needs a separate product decision and is not grantable here"
                )
        consent_id = new_id("consent")
        self.db.insert_consent(
            consent_id,
            version=self.consent_version,
            purposes=",".join(str(purpose) for purpose in purposes),
            granted_at=stamp,
            expires_at=stamp + days * 86400,
            policy_version=POLICY_VERSION,
        )
        events.emit("consent.granted", consent=consent_id, version=self.consent_version)
        record = self.active(now=stamp)
        assert record is not None
        return record

    def pause(self, *, now: float | None = None) -> ConsentRecord:
        """One-click pause: queued work stops, consent stays on record."""
        record = self._require_active(now)
        self.db.set_consent_paused(record.consent_id, True)
        events.emit("consent.paused", consent=record.consent_id)
        return _record(self.db.consent(record.consent_id))

    def resume(self, *, now: float | None = None) -> ConsentRecord:
        record = self._require_active(now)
        self.db.set_consent_paused(record.consent_id, False)
        events.emit("consent.resumed", consent=record.consent_id)
        return _record(self.db.consent(record.consent_id))

    def withdraw(self, *, now: float | None = None) -> str:
        """Withdraw consent and open a deletion request.

        Returns the deletion request ID.  Cancelling queued and in-flight work
        is the queue's job; see :meth:`dictation.upload.queue.UploadQueue.on_withdrawal`.
        """
        stamp = time.time() if now is None else now
        record = self.active(now=stamp)
        if record is None:
            raise ConsentError("there is no active consent to withdraw")
        self.db.revoke_consent(record.consent_id, now=stamp)
        request_id = new_id("deletion")
        self.db.insert_deletion(request_id, scope="withdrawal", now=stamp)
        events.emit("consent.withdrawn", consent=record.consent_id, request=request_id)
        return request_id

    def request_deletion(self, scope: str = "all_training_data", *, now: float | None = None) -> str:
        """Record a "delete my training data and personalized model" request."""
        stamp = time.time() if now is None else now
        request_id = new_id("deletion")
        self.db.insert_deletion(request_id, scope=scope, now=stamp)
        events.emit("consent.deletion_requested", request=request_id, scope=scope)
        return request_id

    # -- eligibility ---------------------------------------------------------

    def check_session(
        self,
        *,
        session_started_at: float,
        session_opted_out: bool = False,
        language: str = "en",
        now: float | None = None,
    ) -> EligibilityDecision:
        """Decide whether one session may be contributed."""
        stamp = time.time() if now is None else now
        if not UPLOAD_GATES_MET and not self.allow_unvalidated_upload:
            return EligibilityDecision(False, "upload_gates_not_met")
        if session_opted_out:
            return EligibilityDecision(False, "session_opted_out")
        record = self.active(now=stamp)
        if record is None:
            return EligibilityDecision(False, "no_active_consent")
        if record.version != self.consent_version:
            return EligibilityDecision(False, "consent_version_superseded", record.consent_id)
        if record.paused:
            return EligibilityDecision(False, "consent_paused", record.consent_id)
        if not record.covers(Purpose.CUSTOMER_PERSONALIZATION):
            return EligibilityDecision(False, "purpose_not_granted", record.consent_id)
        if session_started_at < record.granted_at:
            # No retroactive uploads when consent is enabled later.
            return EligibilityDecision(False, "session_predates_consent", record.consent_id)
        return EligibilityDecision(True, "", record.consent_id)

    def recheck(self, consent_id: str, *, now: float | None = None) -> EligibilityDecision:
        """Re-verify a specific consent record before transfer.

        Called at enqueue, before transfer and at server admission.  A receipt
        arriving after withdrawal never makes a sample eligible again, because
        this returns the current state, not the state at enqueue.
        """
        stamp = time.time() if now is None else now
        row = self.db.consent(consent_id)
        if row is None:
            return EligibilityDecision(False, "consent_missing")
        record = _record(row)
        if record.revoked_at is not None:
            return EligibilityDecision(False, "consent_withdrawn", consent_id)
        if record.expires_at <= stamp:
            return EligibilityDecision(False, "consent_expired", consent_id)
        if record.paused:
            return EligibilityDecision(False, "consent_paused", consent_id)
        if record.version != self.consent_version:
            return EligibilityDecision(False, "consent_version_superseded", consent_id)
        return EligibilityDecision(True, "", consent_id)

    def _require_active(self, now: float | None) -> ConsentRecord:
        record = self.active(now=now)
        if record is None:
            raise ConsentError("there is no active consent")
        return record


def _record(row: object) -> ConsentRecord:
    assert row is not None
    return ConsentRecord(
        consent_id=row["consent_id"],  # type: ignore[index]
        version=row["version"],  # type: ignore[index]
        purposes=tuple(part for part in str(row["purposes"]).split(",") if part),  # type: ignore[index]
        granted_at=row["granted_at"],  # type: ignore[index]
        expires_at=row["expires_at"],  # type: ignore[index]
        revoked_at=row["revoked_at"],  # type: ignore[index]
        paused=bool(row["paused"]),  # type: ignore[index]
        policy_version=row["policy_version"],  # type: ignore[index]
    )
