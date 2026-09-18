"""Deterministic privacy-classifier stand-ins.

These are how the fail-closed behaviour of plan section 7.10 gets tested: no
timeout, parse error, worker crash or memory shortage may default to a clean
result.  Each class here reproduces one of those failures on demand.

:class:`RuleEchoClassifier` is the useful development default.  It re-reports
the deterministic rule hits as if they were model spans, so the pipeline
produces realistic output without a model, while making no pretence of
contextual judgement - it cannot find what the rules cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dictation.errors import PrivacyWorkerUnavailable
from dictation.privacy import rules
from dictation.privacy.classifier.base import ClassifierCapabilities, ClassifierWindow
from dictation.privacy.schema import spans_to_payload
from dictation.types import Transcript


@dataclass
class EmptyClassifier:
    """Finds nothing.  Rule hits still apply, since the union is taken."""

    uncertain: bool = False
    calls: int = 0

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(model_id="empty-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> dict[str, Any]:
        self.calls += 1
        return {"spans": [], "uncertain": self.uncertain}


@dataclass
class ScriptedClassifier:
    """Returns a prepared answer per window index."""

    answers: dict[int, Any] = field(default_factory=dict)
    default: Any = field(default_factory=lambda: {"spans": [], "uncertain": False})
    calls: list[int] = field(default_factory=list)

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(model_id="scripted-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> Any:
        self.calls.append(window.index)
        return self.answers.get(window.index, self.default)


@dataclass
class RuleEchoClassifier:
    """Reports the deterministic rule hits inside the window as model spans."""

    private_terms: tuple[str, ...] = ()
    uncertain_on_injection: bool = True
    calls: int = 0

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(model_id="rule-echo-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> dict[str, Any]:
        self.calls += 1
        fragment = Transcript(
            session_id="window",
            revision=1,
            words=window.words,
            total_samples=window.words[-1].end_sample if window.words else 0,
        )
        findings = rules.detect(fragment, private_terms=self.private_terms)
        payload = spans_to_payload(
            sorted(findings.spans),
            uncertain=self.uncertain_on_injection and findings.injection_suspected,
        )
        return payload


@dataclass
class FailingClassifier:
    """Crashes.  The session must be rejected, not passed through."""

    message: str = "synthetic privacy worker crash"

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(model_id="failing-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> Any:
        raise PrivacyWorkerUnavailable(self.message)


@dataclass
class TimeoutClassifier:
    """Reports a timeout for the chosen windows."""

    fail_on: frozenset[int] = frozenset({0})

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(model_id="timeout-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> Any:
        if window.index in self.fail_on:
            raise TimeoutError(f"window {window.index} exceeded {timeout_s}s")
        return {"spans": [], "uncertain": False}


@dataclass
class MalformedClassifier:
    """Returns output that fails validation: junk, or IDs it was never shown."""

    mode: str = "junk"

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(model_id="malformed-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> Any:
        match self.mode:
            case "junk":
                return "I could not find anything sensitive."
            case "truncated":
                return '{"spans": [{"start_word_id": 0,'
            case "out_of_range":
                far = window.end_word_id_exclusive + 500
                return {
                    "spans": [
                        {
                            "start_word_id": far,
                            "end_word_id_exclusive": far + 2,
                            "category": "contact",
                            "action": "drop_sentence",
                        }
                    ],
                    "uncertain": False,
                }
            case "missing_uncertain":
                return {"spans": []}
            case "unknown_category":
                return {
                    "spans": [
                        {
                            "start_word_id": window.start_word_id,
                            "end_word_id_exclusive": window.start_word_id + 1,
                            "category": "vibes",
                            "action": "drop_sentence",
                        }
                    ],
                    "uncertain": False,
                }
        raise ValueError(f"unknown malformed mode {self.mode!r}")


def callable_classifier(
    function: Callable[[ClassifierWindow], Any],
    model_id: str = "callable-mock",
) -> Any:
    """Wrap a plain function as a classifier, for one-off tests."""

    @dataclass
    class _Wrapper:
        @property
        def capabilities(self) -> ClassifierCapabilities:
            return ClassifierCapabilities(model_id=model_id)

        def warm_up(self) -> None:
            return None

        def release(self) -> None:
            return None

        def classify(self, window: ClassifierWindow, *, timeout_s: float) -> Any:
            return function(window)

    return _Wrapper()
