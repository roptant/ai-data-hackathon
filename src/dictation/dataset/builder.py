"""Dataset construction (plan section 7, steps 1-10).

This is the privacy-critical module.  It takes a frozen transcript, its audio
and a completed privacy analysis, and produces either eligible training
examples or a rejection with a reason code.  It never produces "probably fine".

Invariants it enforces, each covered by a test:

* No sample inside a removal interval appears in any output clip or in any
  metadata that leaves the machine.
* Every retained clip's words lie wholly inside that clip's audio, so the
  audio/text pair is exact.
* Clips are contiguous source intervals.  The spliced export exists for
  playback and carries a local-only boundary manifest; the canonical dataset is
  the contiguous clips (plan section 7, last paragraph).
* Alignment problems discard the utterance or the session; they never cause a
  cut that cannot be justified, and they never introduce a third model.
* Failure of any stage - worker crash, timeout, schema violation, missing
  timings, silence, uncertainty - yields ``Decision.REJECTED``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from dictation.capture.vad import EnergyVad
from dictation.config import PrivacySettings, QualitySettings
from dictation.dataset import quality
from dictation.dataset.intervals import RemovalPlan, plan_removals, words_fully_inside
from dictation.errors import DatasetRejected
from dictation.logging_ import events
from dictation.privacy.pipeline import PrivacyAnalysis
from dictation.time_map import DestinationMap, complement, validate_alignment
from dictation.types import (
    CANONICAL_SAMPLE_WIDTH,
    Decision,
    QualityMetrics,
    RetainedClip,
    SampleInterval,
    Transcript,
    Word,
)
from dictation.version import POLICY_VERSION

#: Alignment problems that make the whole session unusable: the timings
#: contradict themselves, so no cut anywhere can be trusted.
SESSION_FATAL_ALIGNMENT = frozenset(
    {"overlapping_or_nonmonotonic", "word_outside_audio", "zero_length_word", "empty_word_text"}
)
#: Problems that are local to a word: drop the sentence around it instead.
SENTENCE_LOCAL_ALIGNMENT = frozenset({"low_confidence_word", "unreliable_timing"})


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Outcome of building the training copy for one session."""

    session_id: str
    decision: Decision
    reason: str = ""
    clips: tuple[RetainedClip, ...] = ()
    clip_audio: tuple[bytes, ...] = ()
    removed: tuple[SampleInterval, ...] = ()
    retained: tuple[SampleInterval, ...] = ()
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    #: Local-only: maps retained source intervals to the spliced timeline.
    boundary_manifest: tuple[dict[str, int], ...] = ()
    content_hash: str = ""
    policy_version: str = POLICY_VERSION
    asr_model: str = ""
    privacy_model: str = ""
    language: str = "en"
    #: Per-clip rejections, for the local audit trail.
    clip_rejections: tuple[tuple[int, str], ...] = ()

    @property
    def eligible(self) -> bool:
        return self.decision is Decision.ELIGIBLE

    @property
    def total_retained_ms(self) -> int:
        return sum(clip.duration_ms for clip in self.clips)

    def spliced_audio(self) -> bytes:
        """Concatenated retained audio, for local playback and export.

        Its timestamps are the destination timeline in
        :attr:`boundary_manifest`.  Nothing is inserted between clips: no
        beeps, no silence, no synthetic bridge audio.
        """
        return b"".join(self.clip_audio)

    def spliced_text(self) -> str:
        """Retained text, one line per clip.

        Separate lines rather than a single paragraph: joining unrelated
        sentences as though they were continuous speech would create a
        grammatical claim nobody made (plan section 7.8).
        """
        return "\n".join(clip.text for clip in self.clips)


def _reject(session_id: str, reason: str, **fields: object) -> BuildResult:
    events.emit("dataset.rejected", session=session_id, reason=reason, **fields)
    return BuildResult(session_id=session_id, decision=Decision.REJECTED, reason=reason)


def content_hash(clips: tuple[RetainedClip, ...]) -> str:
    """Stable hash of retained text, used only for duplicate detection."""
    digest = hashlib.sha256()
    for clip in clips:
        digest.update(clip.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_dataset(
    analysis: PrivacyAnalysis,
    pcm: bytes,
    *,
    quality_settings: QualitySettings | None = None,
    privacy_settings: PrivacySettings | None = None,
    seen_hashes: set[str] | None = None,
    vad: EnergyVad | None = None,
    recheck: "RecheckFunction | None" = None,
) -> BuildResult:
    """Build the training copy for one session.

    ``recheck`` runs detection again over the retained text (plan section 7.9).
    It is injected rather than called directly so that the same builder works
    with the model loaded or released, and so a recheck failure is a rejection
    rather than an exception.
    """
    transcript = analysis.transcript
    session_id = transcript.session_id
    quality_settings = quality_settings or QualitySettings()
    privacy_settings = privacy_settings or PrivacySettings()
    vad = vad or EnergyVad(sample_rate=transcript.sample_rate)

    # Step 10 first, as a precondition: nothing proceeds on a failed analysis.
    if analysis.rejected_reason:
        return _reject(session_id, analysis.rejected_reason)
    if analysis.uncertain:
        return _reject(session_id, "uncertain_analysis")

    usable = quality.check_transcript_usable(transcript)
    if not usable:
        return _reject(session_id, usable.reason)

    language_gate = quality.check_language(transcript.language)
    if not language_gate:
        return _reject(session_id, language_gate.reason)

    available_samples = len(pcm) // CANONICAL_SAMPLE_WIDTH
    if available_samples <= 0:
        return _reject(session_id, "no_audio")
    bound = SampleInterval(0, available_samples)

    # Step 7: alignment validation before any cut is computed.
    problems = validate_alignment(
        transcript.words, bound, min_confidence=privacy_settings.min_word_confidence
    )
    fatal = SESSION_FATAL_ALIGNMENT.intersection(problems)
    if fatal:
        return _reject(session_id, "alignment_unusable", detail=sorted(fatal)[0])

    local = SENTENCE_LOCAL_ALIGNMENT.intersection(problems)
    extra_sentences: set[int] = set()
    if local:
        for word in transcript.words:
            if word.confidence < privacy_settings.min_word_confidence or word.timing_unreliable:
                for sentence in analysis.sentences:
                    if sentence.contains_word(word.id):
                        extra_sentences.add(sentence.index)

    # Steps 5-7: expand, map to intervals, pad, merge, pull in touched words.
    removal: RemovalPlan = plan_removals(
        transcript,
        analysis.spans,
        analysis.sentences,
        settings=privacy_settings,
        extra_sentence_indices=frozenset(extra_sentences),
        bound=bound,
    )

    # Step 8: the complement is what may be retained.
    retained = complement(removal.intervals, bound)
    if not retained:
        return _reject(session_id, "fully_redacted")

    destination = DestinationMap.build(retained)
    clips: list[RetainedClip] = []
    audio: list[bytes] = []
    rejections: list[tuple[int, str]] = []

    for index, interval in enumerate(retained):
        # Defensive: the complement must not intersect any removal interval.
        if any(interval.intersects(removed) for removed in removal.intervals):
            return _reject(session_id, "retained_interval_overlaps_removal")

        word_ids = words_fully_inside(transcript, interval, excluded=removal.removed_word_ids)
        words = tuple(transcript.word_by_id(word_id) for word_id in word_ids)
        clip_pcm = pcm[
            interval.start * CANONICAL_SAMPLE_WIDTH : interval.end * CANONICAL_SAMPLE_WIDTH
        ]
        clip = _make_clip(
            index=len(clips),
            interval=interval,
            destination=destination.destination_of(index),
            words=words,
            transcript=transcript,
            clip_pcm=clip_pcm,
            vad=vad,
        )
        gate = quality.check_clip(clip, quality=quality_settings, privacy=privacy_settings)
        if not gate:
            rejections.append((index, gate.reason))
            continue
        consistency = quality.check_audio_text_consistency(clip, clip_pcm, vad=vad)
        if not consistency:
            rejections.append((index, consistency.reason))
            continue
        clips.append(clip)
        audio.append(clip_pcm)

    session_gate = quality.check_session(tuple(clips), quality=quality_settings)
    if not session_gate:
        return _reject(session_id, session_gate.reason)

    kept = tuple(clips)
    # Rebuild the destination timeline over the clips that survived, so the
    # spliced export's timestamps match the audio it actually contains.
    kept_map = DestinationMap.build(tuple(clip.source for clip in kept))
    kept = tuple(
        RetainedClip(
            clip_index=index,
            source=clip.source,
            destination=kept_map.destination_of(index),
            words=clip.words,
            text=clip.text,
            sample_rate=clip.sample_rate,
            mean_confidence=clip.mean_confidence,
            speech_ratio=clip.speech_ratio,
        )
        for index, clip in enumerate(kept)
    )

    # Step 9: recheck what survived.
    if recheck is not None:
        clean, reason = recheck(kept, transcript)
        gate = quality.check_retained_text_clean(clean, reason)
        if not gate:
            return _reject(session_id, gate.reason)

    digest = content_hash(kept)
    if seen_hashes is not None:
        duplicate = quality.check_duplicate(digest, seen_hashes)
        if not duplicate:
            return _reject(session_id, duplicate.reason)

    metrics = QualityMetrics(
        duration_ms=sum(clip.duration_ms for clip in kept),
        word_count=sum(len(clip.words) for clip in kept),
        mean_confidence=(
            sum(clip.mean_confidence * len(clip.words) for clip in kept)
            / max(1, sum(len(clip.words) for clip in kept))
        ),
        speech_ratio=(
            sum(clip.speech_ratio * clip.duration_ms for clip in kept)
            / max(1, sum(clip.duration_ms for clip in kept))
        ),
        removed_interval_count=len(removal.intervals),
        removed_duration_ms=int(round(removal.removed_samples * 1000 / transcript.sample_rate)),
        clip_count=len(kept),
    )

    events.emit(
        "dataset.built",
        session=session_id,
        clips=len(kept),
        retained_ms=metrics.duration_ms,
        removed_ms=metrics.removed_duration_ms,
        rule_span_count=len(analysis.mandatory_spans),
        span_count=len(analysis.spans),
        policy=POLICY_VERSION,
    )
    return BuildResult(
        session_id=session_id,
        decision=Decision.ELIGIBLE,
        clips=kept,
        clip_audio=tuple(audio),
        removed=removal.intervals,
        retained=tuple(clip.source for clip in kept),
        metrics=metrics,
        boundary_manifest=kept_map.boundary_manifest(),
        content_hash=digest,
        asr_model=transcript.asr_model,
        privacy_model=analysis.classifier_model,
        language=transcript.language,
        clip_rejections=tuple(rejections),
    )


def _make_clip(
    *,
    index: int,
    interval: SampleInterval,
    destination: SampleInterval,
    words: tuple[Word, ...],
    transcript: Transcript,
    clip_pcm: bytes,
    vad: EnergyVad,
) -> RetainedClip:
    mean_confidence = (
        sum(word.confidence for word in words) / len(words) if words else 0.0
    )
    return RetainedClip(
        clip_index=index,
        source=interval,
        destination=destination,
        words=words,
        text=transcript.spoken_text(words),
        sample_rate=transcript.sample_rate,
        mean_confidence=mean_confidence,
        speech_ratio=vad.speech_ratio(clip_pcm),
    )


class RecheckFunction:
    """Callable protocol for the post-removal recheck.

    Implementations take the retained clips and the original transcript, and
    return ``(clean, reason)``.  See :func:`dictation.dataset.builder.make_recheck`.
    """

    def __call__(
        self, clips: tuple[RetainedClip, ...], transcript: Transcript
    ) -> tuple[bool, str]:  # pragma: no cover - protocol
        raise NotImplementedError


def make_recheck(analyze_clean: "CleanChecker") -> RecheckFunction:
    """Adapt a transcript-level checker into a clip-level recheck.

    The retained clips are re-assembled into a transcript whose word IDs are
    renumbered from zero, because the retained text is all the recheck may see:
    passing the original IDs would leak the positions of what was removed.
    """

    class _Recheck(RecheckFunction):
        def __call__(
            self, clips: tuple[RetainedClip, ...], transcript: Transcript
        ) -> tuple[bool, str]:
            words: list[Word] = []
            cursor = 0
            for clip in clips:
                for word in clip.words:
                    offset = clip.destination.start - clip.source.start
                    words.append(
                        Word(
                            id=len(words),
                            text=word.text,
                            display=word.display,
                            start_sample=word.start_sample + offset,
                            end_sample=word.end_sample + offset,
                            confidence=word.confidence,
                            segment_id=f"clip-{clip.clip_index}",
                            timing_unreliable=word.timing_unreliable,
                        )
                    )
                cursor += clip.source.length
            retained_transcript = Transcript(
                session_id=f"{transcript.session_id}-retained",
                revision=1,
                words=tuple(words),
                sample_rate=transcript.sample_rate,
                total_samples=cursor,
                language=transcript.language,
                asr_model=transcript.asr_model,
                asr_model_revision=transcript.asr_model_revision,
                word_timings_experimental=transcript.word_timings_experimental,
            )
            return analyze_clean(retained_transcript)

    return _Recheck()


class CleanChecker:
    """Callable that reports whether a transcript is free of sensitive spans."""

    def __call__(self, transcript: Transcript) -> tuple[bool, str]:  # pragma: no cover
        raise NotImplementedError


def assert_no_removed_audio(result: BuildResult) -> None:
    """Raise unless every output clip avoids every removal interval.

    Called by the store before an artifact is written, so that the invariant is
    checked against the object that is about to be persisted, not only against
    the one the builder returned.
    """
    for clip in result.clips:
        for removed in result.removed:
            if clip.source.intersects(removed):
                raise DatasetRejected(
                    "removed_audio_in_output",
                    f"clip {clip.clip_index} intersects a removal interval",
                )
    for clip, pcm in zip(result.clips, result.clip_audio, strict=False):
        expected = clip.source.length * CANONICAL_SAMPLE_WIDTH
        if len(pcm) != expected:
            raise DatasetRejected(
                "clip_length_mismatch",
                f"clip {clip.clip_index} has {len(pcm)} bytes, expected {expected}",
            )
