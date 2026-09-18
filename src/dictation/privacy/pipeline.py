"""Union of deterministic rules and the local privacy model (plan section 7).

Steps 1-4 of the plan's pipeline live here: freeze, detect, validate, and take
the union.  Everything about this function is fail-closed.  If a window times
out, the worker crashes, the answer fails validation, or any word is left
uncovered by a window, the result carries a rejection reason and the caller
discards the training copy.  Local dictation is unaffected: this runs on a
separate copy, after the transcript has already been delivered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from dictation.config import PrivacySettings
from dictation.errors import PrivacyWorkerUnavailable, SchemaViolation
from dictation.logging_ import events
from dictation.privacy import rules
from dictation.privacy.classifier.base import (
    ClassifierRun,
    PrivacyClassifier,
    build_windows,
    coverage_gap,
)
from dictation.privacy.schema import validate_output
from dictation.types import Sentence, Span, Transcript, sentences_from_words
from dictation.version import POLICY_VERSION


@dataclass(frozen=True, slots=True)
class PrivacyAnalysis:
    """Everything the dataset builder needs, plus why it may not proceed."""

    transcript: Transcript
    sentences: tuple[Sentence, ...]
    spans: tuple[Span, ...]
    uncertain: bool
    injection_suspected: bool
    rule_detectors: tuple[str, ...]
    window_count: int
    classifier_model: str
    classifier_revision: str
    policy_version: str = POLICY_VERSION
    rejected_reason: str = ""
    run: ClassifierRun = field(default_factory=ClassifierRun)

    @property
    def ok(self) -> bool:
        """True only when every required stage succeeded and nothing is unsure."""
        return not self.rejected_reason and not self.uncertain

    @property
    def mandatory_spans(self) -> tuple[Span, ...]:
        """Rule and user-term hits.  The model cannot override these."""
        return tuple(span for span in self.spans if span.is_mandatory)


def analyze(
    transcript: Transcript,
    classifier: PrivacyClassifier,
    *,
    settings: PrivacySettings | None = None,
    private_terms: tuple[str, ...] = (),
) -> PrivacyAnalysis:
    """Detect sensitive spans in a frozen transcript.

    The transcript must already be frozen: word IDs and timings do not change
    after this point, so a span always refers to the same audio.
    """
    settings = settings or PrivacySettings()
    sentences = sentences_from_words(transcript.words, sample_rate=transcript.sample_rate)
    findings = rules.detect(transcript, private_terms=private_terms)
    run = ClassifierRun()

    def result(
        spans: tuple[Span, ...],
        *,
        uncertain: bool,
        reason: str = "",
        windows: int = 0,
    ) -> PrivacyAnalysis:
        capabilities = classifier.capabilities
        analysis = PrivacyAnalysis(
            transcript=transcript,
            sentences=sentences,
            spans=tuple(sorted(spans)),
            uncertain=uncertain,
            injection_suspected=findings.injection_suspected,
            rule_detectors=findings.detectors,
            window_count=windows,
            classifier_model=capabilities.model_id,
            classifier_revision=capabilities.model_revision,
            rejected_reason=reason,
            run=run,
        )
        events.emit(
            "privacy.analyzed",
            session=transcript.session_id,
            span_count=len(analysis.spans),
            rule_span_count=len(analysis.mandatory_spans),
            windows=windows,
            uncertain=uncertain,
            injection=findings.injection_suspected,
            reason=reason or "ok",
            policy=POLICY_VERSION,
        )
        return analysis

    if transcript.is_empty:
        # Nothing was said.  Not an error, but nothing to contribute either.
        return result((), uncertain=False, reason="empty_transcript")

    windows = build_windows(
        transcript,
        sentences,
        window_words=settings.window_words,
        overlap_words=settings.window_overlap_words,
    )
    gap = coverage_gap(transcript, windows)
    if gap:
        # Text that exceeded a context limit is never treated as clean.
        run.reasons.append("window_coverage_gap")
        return result(findings.spans, uncertain=True, reason="window_coverage_gap", windows=len(windows))

    run.window_count = len(windows)
    model_spans: list[Span] = []
    uncertain = findings.injection_suspected

    for window in windows:
        began = time.monotonic()
        try:
            raw = classifier.classify(window, timeout_s=settings.classifier_timeout_s)
        except TimeoutError:
            run.failures += 1
            run.reasons.append("classifier_timeout")
            return result(findings.spans, uncertain=True, reason="classifier_timeout", windows=len(windows))
        except PrivacyWorkerUnavailable:
            run.failures += 1
            run.reasons.append("privacy_worker_unavailable")
            return result(
                findings.spans, uncertain=True, reason="privacy_worker_unavailable", windows=len(windows)
            )
        except Exception:  # noqa: BLE001 - any worker fault rejects the copy
            run.failures += 1
            run.reasons.append("privacy_worker_error")
            return result(
                findings.spans, uncertain=True, reason="privacy_worker_error", windows=len(windows)
            )

        elapsed = time.monotonic() - began
        if elapsed > settings.classifier_timeout_s:
            run.failures += 1
            run.reasons.append("classifier_timeout")
            return result(findings.spans, uncertain=True, reason="classifier_timeout", windows=len(windows))

        try:
            spans, window_uncertain = validate_output(
                raw,
                allowed_word_ids=window.word_ids,
                allow_word_level_cuts=settings.allow_word_level_cuts,
            )
        except SchemaViolation:
            run.failures += 1
            run.reasons.append("schema_violation")
            return result(findings.spans, uncertain=True, reason="schema_violation", windows=len(windows))

        run.answered += 1
        run.span_count += len(spans)
        if window_uncertain:
            run.uncertain_windows += 1
            uncertain = True
        model_spans.extend(spans)

    # Union.  A rule-based exclusion is never removed by a model answer, and
    # duplicate spans from overlapping windows are harmless: the interval stage
    # merges them.
    combined = tuple(sorted({*findings.spans, *model_spans}))
    return result(combined, uncertain=uncertain, windows=len(windows))


def recheck_clean(
    transcript: Transcript,
    classifier: PrivacyClassifier,
    *,
    settings: PrivacySettings | None = None,
    private_terms: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Re-run detection over retained text only (plan section 7.9).

    Returns ``(clean, reason)``.  This is a second look at what survived, not
    independent proof: the same detectors and the same model see it again, and
    both can miss the same thing twice.
    """
    analysis = analyze(transcript, classifier, settings=settings, private_terms=private_terms)
    if analysis.rejected_reason == "empty_transcript":
        return False, "empty_after_removal"
    if analysis.rejected_reason:
        return False, analysis.rejected_reason
    if analysis.uncertain:
        return False, "uncertain_after_removal"
    if analysis.spans:
        return False, "sensitive_after_removal"
    return True, ""
