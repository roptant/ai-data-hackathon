"""Core data model shared by every module.

Terminology follows the plan: *samples* are indices into the canonical 16 kHz
mono PCM stream, intervals are half-open ``[start, end)`` integer sample ranges
(plan section 7), and word IDs are immutable within a frozen transcript
revision so the privacy worker can only ever refer to text it was shown.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Iterable, Iterator, Sequence

CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH = 2  # int16 little endian


def new_id(prefix: str) -> str:
    """Random, non-sequential identifier.  Nothing derived from user content."""
    return f"{prefix}-{secrets.token_hex(8)}"


class SensitiveCategory(StrEnum):
    """Categories the detectors may assign (plan section 7, first paragraph)."""

    DIRECT_IDENTIFIER = "direct_identifier"
    CONTACT = "contact"
    ADDRESS = "address"
    ACCOUNT_IDENTIFIER = "account_identifier"
    GOVERNMENT_ID = "government_id"
    FINANCIAL = "financial"
    CREDENTIAL_SECRET = "credential_secret"
    CUSTOMER_CONFIDENTIAL = "customer_confidential"
    HEALTH = "health"
    POLITICAL_RELIGIOUS = "political_religious"
    SEXUALITY = "sexuality"
    SENSITIVE_NARRATIVE = "sensitive_narrative"
    CONTEXTUAL_COMBINATION = "contextual_combination"
    USER_DEFINED = "user_defined"


class RemovalAction(StrEnum):
    """What to do with a detected span.

    ``DROP_SENTENCE`` is the default: it removes surrounding clues and avoids
    training on unnatural fragments.  ``DROP_WORDS`` is permitted only where
    alignment has been validated (plan section 7.5).
    """

    DROP_SENTENCE = "drop_sentence"
    DROP_WORDS = "drop_words"


class DetectionSource(StrEnum):
    """Where a span came from.  RULES and USER_TERMS are never overridable by
    the model (plan section 7.2)."""

    RULES = "rules"
    MODEL = "model"
    USER_TERMS = "user_terms"


class Decision(StrEnum):
    """Outcome of building a training copy."""

    ELIGIBLE = "eligible"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True, order=True)
class SampleInterval:
    """Half-open sample range ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("sample interval bounds must be integers")
        if self.start < 0:
            raise ValueError(f"negative interval start: {self.start}")
        if self.end < self.start:
            raise ValueError(f"inverted interval: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        return self.end == self.start

    def contains(self, sample: int) -> bool:
        return self.start <= sample < self.end

    def intersects(self, other: SampleInterval) -> bool:
        """True when the two ranges share at least one sample."""
        return self.start < other.end and other.start < self.end

    def touches(self, other: SampleInterval) -> bool:
        """True when they intersect or abut, so merging them loses nothing."""
        return self.start <= other.end and other.start <= self.end

    def clamp(self, bound: SampleInterval) -> SampleInterval:
        start = min(max(self.start, bound.start), bound.end)
        end = min(max(self.end, bound.start), bound.end)
        return SampleInterval(start, end)

    def shift(self, delta: int) -> SampleInterval:
        return SampleInterval(self.start + delta, self.end + delta)


@dataclass(frozen=True, slots=True)
class Word:
    """One spoken word with its alignment to the canonical sample stream.

    ``text`` is the spoken form used as a training label.  Display punctuation
    and casing are kept out of it (plan section 6) and rebuilt separately.
    """

    id: int
    text: str
    start_sample: int
    end_sample: int
    confidence: float = 1.0
    segment_id: str = ""
    #: Display form including punctuation, used for insertion into the focused
    #: application; never used as a spoken-word training label.
    display: str = ""
    #: True when the recogniser reported the timing as low quality or derived
    #: rather than measured.  Any such word blocks fine-grained word cuts.
    timing_unreliable: bool = False

    def __post_init__(self) -> None:
        if self.end_sample < self.start_sample:
            raise ValueError(f"word {self.id} has inverted timing")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"word {self.id} confidence out of range")

    @property
    def interval(self) -> SampleInterval:
        return SampleInterval(self.start_sample, self.end_sample)

    @property
    def display_text(self) -> str:
        return self.display or self.text


@dataclass(frozen=True, slots=True)
class Segment:
    """A revisable unit of streaming output (plan section 6).

    A later ``revision`` for the same ``id`` replaces the earlier text; clients
    must not append partials blindly (plan section 8).
    """

    id: str
    revision: int
    text: str
    start_sample: int
    end_sample: int
    is_final: bool = False

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("segment revision must be non-negative")
        if self.end_sample < self.start_sample:
            raise ValueError(f"segment {self.id} has inverted timing")


@dataclass(frozen=True, slots=True)
class Transcript:
    """A frozen transcript revision.

    Frozen means: word IDs, texts and timings do not change afterwards.  The
    dataset builder and the privacy worker both operate on one frozen revision
    so that a span always refers to the same audio (plan section 7.1).
    """

    session_id: str
    revision: int
    words: tuple[Word, ...]
    sample_rate: int = CANONICAL_SAMPLE_RATE
    total_samples: int = 0
    language: str = "en"
    asr_model: str = ""
    asr_model_revision: str = ""
    word_timings_experimental: bool = True

    def __post_init__(self) -> None:
        previous_id = -1
        for word in self.words:
            if word.id <= previous_id:
                raise ValueError("word ids must be unique and strictly increasing")
            previous_id = word.id

    def __iter__(self) -> Iterator[Word]:
        return iter(self.words)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def is_empty(self) -> bool:
        return not self.words

    @property
    def span(self) -> SampleInterval:
        """Sample range the transcript covers, including trailing audio."""
        end = self.total_samples
        if self.words:
            end = max(end, self.words[-1].end_sample)
        return SampleInterval(0, end)

    def word_by_id(self, word_id: int) -> Word:
        for word in self.words:
            if word.id == word_id:
                return word
        raise KeyError(f"unknown word id {word_id}")

    def index_of(self, word_id: int) -> int:
        for index, word in enumerate(self.words):
            if word.id == word_id:
                return index
        raise KeyError(f"unknown word id {word_id}")

    def word_ids(self) -> tuple[int, ...]:
        return tuple(word.id for word in self.words)

    def slice_by_ids(self, start_id: int, end_id_exclusive: int) -> tuple[Word, ...]:
        return tuple(w for w in self.words if start_id <= w.id < end_id_exclusive)

    def spoken_text(self, words: Iterable[Word] | None = None) -> str:
        """Space-joined spoken labels: the form used for training targets."""
        chosen = self.words if words is None else words
        return " ".join(w.text for w in chosen)

    def display_text(self) -> str:
        """Punctuated form delivered to the user's focused application."""
        out: list[str] = []
        for word in self.words:
            token = word.display_text
            if out and not token.startswith((",", ".", "!", "?", ";", ":")):
                out.append(" ")
            out.append(token)
        return "".join(out).strip()

    def bumped(self, **changes: object) -> Transcript:
        return replace(self, revision=self.revision + 1, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """A half-open word-ID range marked for removal.

    Word IDs, not character offsets: the model cannot invent a position that
    does not correspond to audio it was shown.
    """

    start_word_id: int
    end_word_id_exclusive: int
    category: SensitiveCategory = SensitiveCategory.CONTEXTUAL_COMBINATION
    action: RemovalAction = RemovalAction.DROP_SENTENCE
    source: DetectionSource = DetectionSource.MODEL
    detector: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.end_word_id_exclusive <= self.start_word_id:
            raise ValueError(
                f"empty or inverted span [{self.start_word_id}, {self.end_word_id_exclusive})"
            )

    @property
    def is_mandatory(self) -> bool:
        """Rules and user terms cannot be overridden by the model."""
        return self.source in (DetectionSource.RULES, DetectionSource.USER_TERMS)


@dataclass(frozen=True, slots=True)
class Sentence:
    """Sentence-or-utterance unit used as the default removal granularity."""

    index: int
    start_word_id: int
    end_word_id_exclusive: int
    interval: SampleInterval

    def contains_word(self, word_id: int) -> bool:
        return self.start_word_id <= word_id < self.end_word_id_exclusive

    def overlaps_span(self, span: Span) -> bool:
        return (
            self.start_word_id < span.end_word_id_exclusive
            and span.start_word_id < self.end_word_id_exclusive
        )


@dataclass(frozen=True, slots=True)
class RetainedClip:
    """One contiguous training example: audio plus the exact words spoken in it."""

    clip_index: int
    source: SampleInterval
    destination: SampleInterval
    words: tuple[Word, ...]
    text: str
    sample_rate: int = CANONICAL_SAMPLE_RATE
    mean_confidence: float = 1.0
    speech_ratio: float = 1.0

    @property
    def duration_ms(self) -> int:
        return int(round(self.source.length * 1000 / self.sample_rate))


@dataclass(slots=True)
class QualityMetrics:
    """Non-content metrics attached to an upload package (plan section 9)."""

    duration_ms: int = 0
    word_count: int = 0
    mean_confidence: float = 0.0
    speech_ratio: float = 0.0
    removed_interval_count: int = 0
    removed_duration_ms: int = 0
    clip_count: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "duration_ms": self.duration_ms,
            "word_count": self.word_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "speech_ratio": round(self.speech_ratio, 4),
            "removed_interval_count": self.removed_interval_count,
            "removed_duration_ms": self.removed_duration_ms,
            "clip_count": self.clip_count,
        }


@dataclass(slots=True)
class SessionMeta:
    """Operational metadata about a session.  Never transcript content."""

    session_id: str
    started_at: float
    sample_rate: int = CANONICAL_SAMPLE_RATE
    language: str = "en"
    contribute: bool = False
    finished_at: float | None = None
    duration_ms: int = 0
    word_count: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)


def sentences_from_words(
    words: Sequence[Word],
    *,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    pause_ms: int = 700,
) -> tuple[Sentence, ...]:
    """Split words into sentences on terminal punctuation or a long pause.

    Pauses matter because dictated speech often omits punctuation entirely; a
    sentence that never terminates would otherwise swallow the whole session and
    drop every example whenever one identifier is detected.
    """
    if not words:
        return ()
    pause_samples = int(pause_ms * sample_rate / 1000)
    sentences: list[Sentence] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        sentences.append(
            Sentence(
                index=len(sentences),
                start_word_id=current[0].id,
                end_word_id_exclusive=current[-1].id + 1,
                interval=SampleInterval(current[0].start_sample, current[-1].end_sample),
            )
        )
        current.clear()

    for position, word in enumerate(words):
        current.append(word)
        terminal = word.display_text.rstrip().endswith((".", "!", "?"))
        gap = False
        if position + 1 < len(words):
            gap = words[position + 1].start_sample - word.end_sample >= pause_samples
        if terminal or gap:
            flush()
    flush()
    return tuple(sentences)
