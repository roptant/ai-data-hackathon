"""Quality and privacy gates (plan sections 7.9-7.10 and 9).

Each gate returns a reason code instead of a boolean, because the reason is
what the local audit trail and the status UI need, and because "rejected" with
no reason is impossible to evaluate later.

The bar is deliberately high: the plan's preferred improvement over
indiscriminate collection is to retain fewer, better examples.
"""

from __future__ import annotations

from dataclasses import dataclass

from dictation.capture.vad import EnergyVad
from dictation.config import PrivacySettings, QualitySettings
from dictation.types import RetainedClip, Transcript
from dictation.version import VALIDATED_UPLOAD_LANGUAGES


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


OK = GateResult(True)


def check_language(language: str) -> GateResult:
    """Only validated languages may become eligible for automatic upload."""
    if language not in VALIDATED_UPLOAD_LANGUAGES:
        return GateResult(False, f"language_not_validated:{language}")
    return OK


def check_clip(
    clip: RetainedClip,
    *,
    quality: QualitySettings | None = None,
    privacy: PrivacySettings | None = None,
) -> GateResult:
    """Gate one retained clip."""
    quality = quality or QualitySettings()
    privacy = privacy or PrivacySettings()
    if clip.duration_ms < quality.min_clip_ms:
        return GateResult(False, "clip_too_short")
    if clip.duration_ms > quality.max_clip_ms:
        return GateResult(False, "clip_too_long")
    if len(clip.words) < quality.min_clip_words:
        return GateResult(False, "clip_too_few_words")
    if not clip.text.strip():
        return GateResult(False, "clip_empty_text")
    if clip.mean_confidence < privacy.min_clip_mean_confidence:
        return GateResult(False, "clip_low_confidence")
    if clip.speech_ratio < quality.min_clip_speech_ratio:
        return GateResult(False, "clip_mostly_silence")
    return OK


def check_session(
    clips: tuple[RetainedClip, ...],
    *,
    quality: QualitySettings | None = None,
) -> GateResult:
    """Gate the session as a whole once per-clip gates have run."""
    quality = quality or QualitySettings()
    if not clips:
        return GateResult(False, "no_retained_clips")
    total = sum(clip.duration_ms for clip in clips)
    if total < quality.min_total_ms:
        return GateResult(False, "retained_audio_too_short")
    if total > quality.max_total_ms:
        return GateResult(False, "retained_audio_too_long")
    return OK


def check_audio_text_consistency(
    clip: RetainedClip,
    pcm: bytes,
    *,
    vad: EnergyVad | None = None,
    min_speech_ratio: float = 0.2,
) -> GateResult:
    """Look for speech in a clip that the transcript does not account for.

    The check is coarse on purpose.  It catches a clip that is mostly noise, and
    a clip whose words cover far less time than its speech does - the signature
    of unexplained or background speech.  Single-speaker detection is imperfect
    and this is not it: the product is personal dictation, not meeting capture
    (plan section 7.9).
    """
    vad = vad or EnergyVad(sample_rate=clip.sample_rate)
    ratio = vad.speech_ratio(pcm)
    if ratio < min_speech_ratio:
        return GateResult(False, "clip_no_speech_detected")
    word_samples = sum(word.end_sample - word.start_sample for word in clip.words)
    speech_samples = sum(interval.length for interval in vad.speech_intervals(pcm))
    if speech_samples and word_samples < 0.45 * speech_samples:
        return GateResult(False, "unexplained_speech_in_clip")
    return OK


def check_retained_text_clean(clean: bool, reason: str) -> GateResult:
    """Adapt the post-removal recheck into a gate result."""
    if clean:
        return OK
    return GateResult(False, reason or "sensitive_after_removal")


def check_duplicate(content_hash: str, seen: set[str]) -> GateResult:
    """Reject an example whose retained text was already contributed.

    Duplicate dictation - a phrase repeated in every session - would be
    over-represented in training and adds no new speech (plan section 10).
    """
    if content_hash in seen:
        return GateResult(False, "duplicate_example")
    return OK


def check_volume_cap(
    packages_today: int,
    seconds_today: int,
    *,
    max_packages: int,
    max_seconds: int,
) -> GateResult:
    """Cap collection volume (plan section 2)."""
    if packages_today >= max_packages:
        return GateResult(False, "daily_package_cap_reached")
    if seconds_today >= max_seconds:
        return GateResult(False, "daily_duration_cap_reached")
    return OK


def check_transcript_usable(transcript: Transcript) -> GateResult:
    if transcript.is_empty:
        return GateResult(False, "empty_transcript")
    if transcript.span.length <= 0:
        return GateResult(False, "no_audio")
    return OK
