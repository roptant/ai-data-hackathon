"""Settings and the provisional constants the plan asks to keep labelled.

Every value marked *provisional* in the plan appears here once, so that
measuring it later changes one place.  Nothing in this module is a statutory
deadline or a safety guarantee (plan sections 4, 5, 6, 7).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from dictation.types import CANONICAL_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    sample_rate: int = CANONICAL_SAMPLE_RATE
    #: Bounded capture buffer.  Overflow is an explicit error, not a drop.
    max_session_seconds: int = 600  # provisional 10-minute cap (plan 5)
    warn_before_limit_seconds: int = 30
    #: Audio captured while a cold model is still loading.  Bounded; on overflow
    #: the session reports failure instead of silently losing early speech.
    warmup_buffer_seconds: int = 20
    #: Recording feedback target (plan 4).  Used by the benchmark harness.
    feedback_target_ms: int = 150
    #: Sessions whose speech ratio is below this are treated as silence and
    #: never produce transcripts or training examples (plan 6).
    min_speech_ratio: float = 0.08


@dataclass(frozen=True, slots=True)
class PrivacySettings:
    #: Padding added to each side of a removal interval before merging.
    #: Provisional: tune by listening tests; 250 ms is not a safety guarantee.
    padding_ms: int = 250
    #: Default removal granularity is the whole sentence (plan 7.5).
    expand_to_sentence: bool = True
    #: Word-level cuts require validated alignment; off until measured.
    allow_word_level_cuts: bool = False
    #: Words per classifier window and the overlap between windows.  Long
    #: transcripts are processed in overlapping sentence windows (plan 4).
    window_words: int = 220
    window_overlap_words: int = 40
    #: Per-window classifier timeout.  A timeout rejects the session.
    classifier_timeout_s: float = 30.0
    #: Minimum word confidence for a word to be usable in a training example.
    min_word_confidence: float = 0.45
    #: Mean confidence required across a retained clip.
    min_clip_mean_confidence: float = 0.6


@dataclass(frozen=True, slots=True)
class QualitySettings:
    min_clip_ms: int = 1_000
    max_clip_ms: int = 30_000
    min_clip_words: int = 3
    min_total_ms: int = 1_500
    max_total_ms: int = 600_000
    #: Retained clip must be mostly speech, not room noise.
    min_clip_speech_ratio: float = 0.35


@dataclass(frozen=True, slots=True)
class RetentionSettings:
    """Plan section 6.  Engineering defaults, not statutory deadlines."""

    raw_session_hours: int = 24
    upload_package_days: int = 7
    rejected_example_seconds: int = 0  # deleted at the decision
    metadata_days: int = 90


@dataclass(frozen=True, slots=True)
class ApiSettings:
    #: Disabled until the user enables integration access (plan 8).
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    #: Events buffered per client before it is disconnected as a slow consumer.
    client_queue_limit: int = 256
    #: Pairing and authentication attempts allowed per window.
    auth_attempts_per_minute: int = 10
    pairing_window_seconds: int = 120


@dataclass(frozen=True, slots=True)
class UploadSettings:
    #: Contribution is off by default (plan 9).
    endpoint: str = ""
    max_attempts: int = 6
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 600.0
    max_package_bytes: int = 32 * 1024 * 1024
    #: Cap on collection volume: retain fewer, higher-quality examples (plan 2).
    max_packages_per_day: int = 40
    max_seconds_per_day: int = 900


@dataclass(frozen=True, slots=True)
class InsertionSettings:
    #: Clipboard fallback needs explicit disclosure before use (plan 5).
    clipboard_fallback_allowed: bool = False
    restore_clipboard: bool = True
    excluded_apps: tuple[str, ...] = ()
    #: Never synthesised, regardless of settings.  Present for documentation.
    submit_with_enter: bool = False


@dataclass(frozen=True, slots=True)
class Settings:
    language: str = "en"
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    retention: RetentionSettings = field(default_factory=RetentionSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    upload: UploadSettings = field(default_factory=UploadSettings)
    insertion: InsertionSettings = field(default_factory=InsertionSettings)
    #: User-defined private terms stay local and are never uploaded (plan 7).
    private_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        nested = {
            "capture": CaptureSettings,
            "privacy": PrivacySettings,
            "quality": QualitySettings,
            "retention": RetentionSettings,
            "api": ApiSettings,
            "upload": UploadSettings,
            "insertion": InsertionSettings,
        }
        kwargs: dict[str, Any] = {}
        for spec in fields(cls):
            if spec.name not in data:
                continue
            value = data[spec.name]
            if spec.name in nested and isinstance(value, dict):
                section = nested[spec.name]
                allowed = {f.name for f in fields(section)}
                kwargs[spec.name] = section(
                    **{k: _coerce(v) for k, v in value.items() if k in allowed}
                )
            else:
                kwargs[spec.name] = _coerce(value)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> Settings:
        """Read settings, falling back to defaults when absent or unreadable.

        A corrupt settings file must not enable anything: defaults keep the API
        disabled and contribution off.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        return cls.from_dict(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


def _coerce(value: Any) -> Any:
    """JSON gives lists where the dataclasses declare tuples."""
    if isinstance(value, list):
        return tuple(value)
    return value
