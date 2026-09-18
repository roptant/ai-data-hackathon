"""Dataset construction: the invariants of plan section 7."""

from __future__ import annotations

import pytest

from dictation.capture.audio_buffer import read_wav
from dictation.config import PrivacySettings, QualitySettings
from dictation.dataset.builder import assert_no_removed_audio, build_dataset, make_recheck
from dictation.dataset.package import build_package, read_package, verify_package
from dictation.errors import DatasetRejected
from dictation.privacy import rules
from dictation.privacy.classifier.mock import (
    EmptyClassifier,
    FailingClassifier,
    MalformedClassifier,
    RuleEchoClassifier,
    ScriptedClassifier,
    TimeoutClassifier,
)
from dictation.privacy.pipeline import analyze
from dictation.types import CANONICAL_SAMPLE_WIDTH, Decision, Transcript, Word

from tests.conftest import Session, make_session


def _build(session: Session, classifier=None, **kwargs):
    classifier = classifier or RuleEchoClassifier()
    analysis = analyze(session.transcript, classifier)
    return analysis, build_dataset(analysis, session.pcm, **kwargs)


def _clean_checker(private_terms: tuple[str, ...] = ()):
    def check(transcript: Transcript) -> tuple[bool, str]:
        findings = rules.detect(transcript, private_terms=private_terms)
        if findings.injection_suspected:
            return False, "injection_after_removal"
        if findings.spans:
            return False, "sensitive_after_removal"
        return True, ""

    return check


# -- the plan's worked example ------------------------------------------------


def test_sensitive_sentence_is_removed_and_the_rest_retained(parcel_session: Session) -> None:
    analysis, result = _build(parcel_session)
    assert analysis.ok
    assert result.decision is Decision.ELIGIBLE, result.reason

    retained_text = result.spliced_text().lower()
    assert "jane" not in retained_text
    assert "oak" not in retained_text
    assert "14" not in retained_text
    assert "parcel" not in retained_text  # whole sentence goes, not just the name
    assert "meeting starts tomorrow" in retained_text
    assert "water the plants" in retained_text


def test_no_retained_sample_comes_from_a_removed_interval(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    assert result.eligible
    assert_no_removed_audio(result)
    for clip in result.clips:
        for removed in result.removed:
            assert not clip.source.intersects(removed)


def test_every_retained_word_lies_inside_its_clip_audio(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    for clip in result.clips:
        for word in clip.words:
            assert clip.source.start <= word.start_sample
            assert word.end_sample <= clip.source.end


def test_clip_audio_length_matches_its_interval(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    for clip, pcm in zip(result.clips, result.clip_audio, strict=True):
        assert len(pcm) == clip.source.length * CANONICAL_SAMPLE_WIDTH


def test_destination_timeline_is_contiguous_from_zero(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    cursor = 0
    for clip in result.clips:
        assert clip.destination.start == cursor
        assert clip.destination.length == clip.source.length
        cursor = clip.destination.end
    assert cursor == len(result.spliced_audio()) // CANONICAL_SAMPLE_WIDTH


def test_spliced_text_keeps_clips_on_separate_lines(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    if len(result.clips) > 1:
        assert "\n" in result.spliced_text()


# -- fail-closed behaviour ---------------------------------------------------


def test_worker_crash_rejects_the_training_copy(parcel_session: Session) -> None:
    analysis = analyze(parcel_session.transcript, FailingClassifier())
    assert not analysis.ok
    assert analysis.rejected_reason == "privacy_worker_unavailable"
    result = build_dataset(analysis, parcel_session.pcm)
    assert result.decision is Decision.REJECTED
    assert result.clips == ()


def test_timeout_rejects_the_training_copy(parcel_session: Session) -> None:
    analysis = analyze(parcel_session.transcript, TimeoutClassifier())
    assert analysis.rejected_reason == "classifier_timeout"
    assert build_dataset(analysis, parcel_session.pcm).decision is Decision.REJECTED


@pytest.mark.parametrize(
    "mode",
    ["junk", "truncated", "out_of_range", "missing_uncertain", "unknown_category"],
)
def test_malformed_output_rejects_the_training_copy(parcel_session: Session, mode: str) -> None:
    analysis = analyze(parcel_session.transcript, MalformedClassifier(mode=mode))
    assert analysis.rejected_reason == "schema_violation"
    assert build_dataset(analysis, parcel_session.pcm).decision is Decision.REJECTED


def test_uncertain_answer_rejects_the_training_copy(clean_session: Session) -> None:
    analysis = analyze(clean_session.transcript, EmptyClassifier(uncertain=True))
    assert analysis.uncertain
    assert not analysis.ok
    assert build_dataset(analysis, clean_session.pcm).decision is Decision.REJECTED


def test_missing_audio_rejects_the_training_copy(clean_session: Session) -> None:
    analysis = analyze(clean_session.transcript, EmptyClassifier())
    assert build_dataset(analysis, b"").decision is Decision.REJECTED


def test_non_monotonic_timings_reject_the_session(clean_session: Session) -> None:
    words = list(clean_session.transcript.words)
    words[3] = Word(
        id=words[3].id,
        text=words[3].text,
        display=words[3].display,
        start_sample=words[1].start_sample,  # overlaps an earlier word
        end_sample=words[3].end_sample,
        confidence=words[3].confidence,
        segment_id=words[3].segment_id,
    )
    transcript = Transcript(
        session_id="bad-alignment",
        revision=1,
        words=tuple(words),
        sample_rate=clean_session.sample_rate,
        total_samples=clean_session.transcript.total_samples,
    )
    analysis = analyze(transcript, EmptyClassifier())
    result = build_dataset(analysis, clean_session.pcm)
    assert result.decision is Decision.REJECTED
    assert result.reason == "alignment_unusable"


def test_unreliable_timing_drops_the_sentence_not_the_session(clean_session: Session) -> None:
    words = list(clean_session.transcript.words)
    words[2] = Word(
        id=words[2].id,
        text=words[2].text,
        display=words[2].display,
        start_sample=words[2].start_sample,
        end_sample=words[2].end_sample,
        confidence=words[2].confidence,
        segment_id=words[2].segment_id,
        timing_unreliable=True,
    )
    transcript = Transcript(
        session_id="soft-alignment",
        revision=1,
        words=tuple(words),
        sample_rate=clean_session.sample_rate,
        total_samples=clean_session.transcript.total_samples,
    )
    analysis = analyze(transcript, EmptyClassifier())
    result = build_dataset(analysis, clean_session.pcm)
    assert result.decision is Decision.ELIGIBLE, result.reason
    assert "printed agenda" not in result.spliced_text()
    assert "water the plants" in result.spliced_text()


def test_fully_sensitive_session_is_rejected_not_emptied() -> None:
    session = make_session("My password is hunter seven seven seven seven.")
    analysis, result = _build(session)
    assert result.decision is Decision.REJECTED
    assert result.reason in {"fully_redacted", "no_retained_clips", "retained_audio_too_short"}


def test_silence_does_not_become_a_training_example() -> None:
    session = make_session("The meeting starts tomorrow and I will bring the agenda.")
    silent = bytes(len(session.pcm))
    analysis = analyze(session.transcript, EmptyClassifier())
    result = build_dataset(analysis, silent)
    assert result.decision is Decision.REJECTED


# -- recheck, duplicates, gates ---------------------------------------------


def test_recheck_rejects_when_sensitive_text_survives(parcel_session: Session) -> None:
    """A classifier that reports nothing cannot smuggle text past the recheck.

    Here the rules would have caught the address, so we force the analysis to
    be empty and confirm the recheck still refuses the result.
    """
    analysis = analyze(parcel_session.transcript, EmptyClassifier())
    empty_spans = type(analysis)(
        transcript=analysis.transcript,
        sentences=analysis.sentences,
        spans=(),
        uncertain=False,
        injection_suspected=False,
        rule_detectors=(),
        window_count=analysis.window_count,
        classifier_model="forced-empty",
        classifier_revision="",
    )
    result = build_dataset(
        empty_spans,
        parcel_session.pcm,
        recheck=make_recheck(_clean_checker()),
    )
    assert result.decision is Decision.REJECTED
    assert result.reason == "sensitive_after_removal"


def test_recheck_passes_for_a_properly_filtered_session(parcel_session: Session) -> None:
    analysis = analyze(parcel_session.transcript, RuleEchoClassifier())
    result = build_dataset(analysis, parcel_session.pcm, recheck=make_recheck(_clean_checker()))
    assert result.decision is Decision.ELIGIBLE, result.reason


def test_duplicate_example_is_rejected(clean_session: Session) -> None:
    analysis = analyze(clean_session.transcript, EmptyClassifier())
    first = build_dataset(analysis, clean_session.pcm, seen_hashes=set())
    assert first.eligible
    seen = {first.content_hash}
    second = build_dataset(analysis, clean_session.pcm, seen_hashes=seen)
    assert second.decision is Decision.REJECTED
    assert second.reason == "duplicate_example"


def test_unvalidated_language_is_not_eligible(clean_session: Session) -> None:
    transcript = Transcript(
        session_id="finnish",
        revision=1,
        words=clean_session.transcript.words,
        sample_rate=clean_session.sample_rate,
        total_samples=clean_session.transcript.total_samples,
        language="fi",
    )
    analysis = analyze(transcript, EmptyClassifier())
    result = build_dataset(analysis, clean_session.pcm)
    assert result.decision is Decision.REJECTED
    assert result.reason.startswith("language_not_validated")


def test_short_clips_are_dropped_by_the_quality_gate(clean_session: Session) -> None:
    analysis = analyze(clean_session.transcript, EmptyClassifier())
    result = build_dataset(
        analysis,
        clean_session.pcm,
        quality_settings=QualitySettings(min_clip_ms=60_000),
    )
    assert result.decision is Decision.REJECTED
    assert result.reason == "no_retained_clips"


def test_padding_widens_the_cut(parcel_session: Session) -> None:
    analysis = analyze(parcel_session.transcript, RuleEchoClassifier())
    narrow = build_dataset(
        analysis, parcel_session.pcm, privacy_settings=PrivacySettings(padding_ms=0)
    )
    wide = build_dataset(
        analysis, parcel_session.pcm, privacy_settings=PrivacySettings(padding_ms=250)
    )
    assert wide.metrics.removed_duration_ms > narrow.metrics.removed_duration_ms


# -- packaging ---------------------------------------------------------------


def test_package_round_trips_and_verifies(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    package = build_package(result, consent_version="consent-test", consent_reference="ref-1")
    manifest = verify_package(package.archive_bytes)
    assert manifest["sample_id"] == package.sample_id
    assert manifest["clip_count"] == len(result.clips)
    assert manifest["language"] == "en"


def test_package_contains_no_session_or_tenant_identity(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    package = build_package(result, consent_version="consent-test", consent_reference="ref-1")
    blob = package.archive_bytes
    assert parcel_session.transcript.session_id.encode() not in blob
    manifest, _ = read_package(blob)
    assert "session_id" not in manifest
    assert "tenant" not in manifest
    assert "boundary_manifest" not in manifest


def test_package_never_contains_removed_words(parcel_session: Session) -> None:
    _, result = _build(parcel_session)
    package = build_package(result, consent_version="consent-test", consent_reference="ref-1")
    _, payloads = read_package(package.archive_bytes)
    text = b" ".join(value for name, value in payloads.items() if name.endswith(".txt")).lower()
    for token in (b"jane", b"oak", b"street", b"parcel"):
        assert token not in text


def test_package_audio_matches_clip_duration(parcel_session: Session, tmp_path) -> None:
    _, result = _build(parcel_session)
    package = build_package(result, consent_version="consent-test", consent_reference="ref-1")
    _, payloads = read_package(package.archive_bytes)
    for clip in result.clips:
        name = f"clip-{clip.clip_index:03d}.wav"
        path = tmp_path / name
        path.write_bytes(payloads[name])
        pcm, rate = read_wav(path)
        assert rate == clip.sample_rate
        assert len(pcm) == clip.source.length * CANONICAL_SAMPLE_WIDTH


def test_rejected_result_cannot_be_packaged(parcel_session: Session) -> None:
    analysis = analyze(parcel_session.transcript, FailingClassifier())
    result = build_dataset(analysis, parcel_session.pcm)
    with pytest.raises(DatasetRejected):
        build_package(result, consent_version="consent-test", consent_reference="ref-1")


def test_identical_results_produce_identical_archive_digests(clean_session: Session) -> None:
    analysis = analyze(clean_session.transcript, EmptyClassifier())
    result = build_dataset(analysis, clean_session.pcm)
    first = build_package(
        result, consent_version="c", consent_reference="r", sample_id="sample-fixed"
    )
    second = build_package(
        result, consent_version="c", consent_reference="r", sample_id="sample-fixed"
    )
    assert first.archive_sha256 == second.archive_sha256


# -- scripted classifier spans ----------------------------------------------


def test_model_span_removes_a_sentence_the_rules_would_miss(clean_session: Session) -> None:
    """A contextual span is honoured even with no rule hit."""
    first_words = clean_session.transcript.words[:3]
    answers = {
        0: {
            "spans": [
                {
                    "start_word_id": first_words[0].id,
                    "end_word_id_exclusive": first_words[2].id + 1,
                    "category": "sensitive_narrative",
                    "action": "drop_sentence",
                }
            ],
            "uncertain": False,
        }
    }
    analysis = analyze(clean_session.transcript, ScriptedClassifier(answers=answers))
    result = build_dataset(analysis, clean_session.pcm)
    assert result.decision is Decision.ELIGIBLE, result.reason
    assert "meeting starts tomorrow" not in result.spliced_text()
    assert "water the plants" in result.spliced_text()


def test_rule_hit_is_not_overridable_by_the_model(secret_session: Session) -> None:
    """The model answering "nothing here" cannot retain a rule-flagged span."""
    analysis = analyze(secret_session.transcript, EmptyClassifier())
    assert analysis.mandatory_spans, "rules should have fired on spoken credentials"
    result = build_dataset(analysis, secret_session.pcm)
    if result.eligible:
        text = result.spliced_text().lower()
        assert "password" not in text
        assert "hunter" not in text
        assert "account number" not in text
