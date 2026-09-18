"""Error taxonomy.

Every failure below is *fail-closed* with respect to training data: raising any
of them must leave the session ineligible for upload while local dictation keeps
working (plan section 2).
"""

from __future__ import annotations


class DictationError(Exception):
    """Base class for all errors raised by this package."""


# -- capture -----------------------------------------------------------------

class CaptureError(DictationError):
    """Microphone or session lifecycle failure."""


class InvalidTransition(CaptureError):
    """A recording command is not valid in the current state (plan section 5)."""

    def __init__(self, state: object, event: object) -> None:
        super().__init__(f"event {event!r} is not valid in state {state!r}")
        self.state = state
        self.event = event


class BufferOverflow(CaptureError):
    """Bounded audio buffer filled up; explicit error, never silent drop."""


class SessionLimitExceeded(CaptureError):
    """Maximum session duration reached (plan section 5)."""


# -- models and workers ------------------------------------------------------

class ModelError(DictationError):
    """Model registry, download or verification failure."""


class ModelNotConfigured(ModelError):
    """No model artifact has been chosen for a role yet (placeholder registry)."""


class ModelNotInstalled(ModelError):
    """A configured model has not been fetched to local storage."""


class ModelVerificationError(ModelError):
    """Checksum, size or manifest verification failed."""


class WorkerError(DictationError):
    """An inference worker failed, timed out or crashed."""


class AsrUnavailable(WorkerError):
    """Speech recognition cannot run; dictation must report, not fake, this."""


class PrivacyWorkerUnavailable(WorkerError):
    """Privacy classification cannot run; training data must be discarded."""


class SchemaViolation(WorkerError):
    """Classifier output failed independent validation (plan section 7.4)."""


# -- dataset -----------------------------------------------------------------

class DatasetRejected(DictationError):
    """The training copy was rejected; the reason code is machine readable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class AlignmentError(DatasetRejected):
    """Word timings are unusable, so no cut can be proven correct."""

    def __init__(self, detail: str = "") -> None:
        super().__init__("alignment_unusable", detail)


# -- storage -----------------------------------------------------------------

class StoreError(DictationError):
    """Local encrypted store failure."""


class SecureStorageUnavailable(StoreError):
    """OS credential store unusable; never fall back to plaintext (plan 6)."""


class DecryptionError(StoreError):
    """Authenticated decryption failed: wrong key, wrong context or tampering."""


class StateTransitionConflict(StoreError):
    """A guarded queue transition lost its race; the caller must re-read state."""


# -- consent, upload, api ----------------------------------------------------

class ConsentError(DictationError):
    """Consent is missing, withdrawn, expired or for a different version."""


class UploadError(DictationError):
    """Upload transport or server admission failure."""


class ServerRejected(UploadError):
    """The server refused the package; it is deleted locally, never retried raw."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ApiError(DictationError):
    """Local API failure."""


class Unauthorized(ApiError):
    """Missing, unknown or revoked token, or a rejected Host/Origin header."""


class ScopeDenied(ApiError):
    """Token authenticated but lacks the scope for this operation."""


class SlowConsumer(ApiError):
    """Client event queue overflowed; it must resynchronise (plan section 8)."""


# -- platform ----------------------------------------------------------------

class PlatformError(DictationError):
    """Platform adapter failure."""


class CapabilityUnavailable(PlatformError):
    """The platform cannot do this; callers must degrade visibly, not silently."""


class PermissionDenied(PlatformError):
    """Microphone, accessibility or portal permission denied or revoked."""
