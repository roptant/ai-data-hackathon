"""Sample arithmetic and interval algebra (plan sections 6 and 7).

Two rules run through this module:

* Intervals are half-open integer sample ranges, so concatenation is exact and
  a sample belongs to exactly one side of a boundary.
* Every rounding on a *removal* interval rounds outward, and every rounding on
  a *retained* interval rounds inward.  A rounding error must never leave part
  of a removed word in the output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from dictation.types import CANONICAL_SAMPLE_RATE, SampleInterval, Word


def ms_to_samples(ms: float, sample_rate: int = CANONICAL_SAMPLE_RATE) -> int:
    """Round-half-up conversion for durations such as padding."""
    return int(round(ms * sample_rate / 1000.0))


def samples_to_ms(samples: int, sample_rate: int = CANONICAL_SAMPLE_RATE) -> float:
    return samples * 1000.0 / sample_rate


def normalize(intervals: Iterable[SampleInterval]) -> tuple[SampleInterval, ...]:
    """Sort, drop empties and merge overlapping or abutting intervals."""
    ordered = sorted((i for i in intervals if not i.is_empty), key=lambda i: (i.start, i.end))
    merged: list[SampleInterval] = []
    for interval in ordered:
        if merged and interval.start <= merged[-1].end:
            previous = merged[-1]
            if interval.end > previous.end:
                merged[-1] = SampleInterval(previous.start, interval.end)
        else:
            merged.append(interval)
    return tuple(merged)


def pad(
    interval: SampleInterval,
    padding_samples: int,
    bound: SampleInterval,
) -> SampleInterval:
    """Grow a removal interval outward, clipped to the available audio."""
    if padding_samples < 0:
        raise ValueError("padding must be non-negative")
    start = max(bound.start, interval.start - padding_samples)
    end = min(bound.end, interval.end + padding_samples)
    return SampleInterval(start, max(start, end))


def pad_all(
    intervals: Iterable[SampleInterval],
    padding_samples: int,
    bound: SampleInterval,
) -> tuple[SampleInterval, ...]:
    return normalize(pad(i, padding_samples, bound) for i in intervals)


def total_length(intervals: Iterable[SampleInterval]) -> int:
    return sum(i.length for i in intervals)


def complement(
    removed: Sequence[SampleInterval],
    bound: SampleInterval,
) -> tuple[SampleInterval, ...]:
    """Retained intervals: ``bound`` minus ``removed`` (plan section 7.8)."""
    retained: list[SampleInterval] = []
    cursor = bound.start
    for interval in normalize(removed):
        clipped = interval.clamp(bound)
        if clipped.is_empty:
            continue
        if clipped.start > cursor:
            retained.append(SampleInterval(cursor, clipped.start))
        cursor = max(cursor, clipped.end)
    if cursor < bound.end:
        retained.append(SampleInterval(cursor, bound.end))
    return tuple(retained)


def intersects_any(interval: SampleInterval, intervals: Iterable[SampleInterval]) -> bool:
    return any(interval.intersects(other) for other in intervals)


def contains_interval(outer: Sequence[SampleInterval], inner: SampleInterval) -> bool:
    """True when ``inner`` lies wholly inside one of ``outer``."""
    return any(o.start <= inner.start and inner.end <= o.end for o in normalize(outer))


@dataclass(frozen=True, slots=True)
class DestinationMap:
    """Maps retained source samples onto the concatenated export timeline.

    For retained source interval ``[a, b)`` following retained intervals of
    total length ``L``, the destination interval is ``[L, L + b - a)`` and a
    retained sample ``t`` maps to ``L + t - a`` (plan section 7).
    """

    retained: tuple[SampleInterval, ...]
    offsets: tuple[int, ...]

    @classmethod
    def build(cls, retained: Sequence[SampleInterval]) -> DestinationMap:
        normalized = normalize(retained)
        offsets: list[int] = []
        running = 0
        for interval in normalized:
            offsets.append(running)
            running += interval.length
        return cls(tuple(normalized), tuple(offsets))

    @property
    def total_samples(self) -> int:
        return total_length(self.retained)

    def destination_of(self, index: int) -> SampleInterval:
        """Destination interval of the ``index``-th retained interval."""
        offset = self.offsets[index]
        return SampleInterval(offset, offset + self.retained[index].length)

    def map_sample(self, sample: int) -> int:
        """Destination position of a retained source sample.

        Raises ``KeyError`` for a removed sample: a removed position has no
        destination, and silently clamping it would fabricate alignment.
        """
        for index, interval in enumerate(self.retained):
            if interval.contains(sample):
                return self.offsets[index] + sample - interval.start
        raise KeyError(f"sample {sample} is not in retained audio")

    def map_word(self, word: Word) -> SampleInterval:
        """Destination interval of a word that lies wholly inside one clip."""
        for index, interval in enumerate(self.retained):
            if interval.start <= word.start_sample and word.end_sample <= interval.end:
                offset = self.offsets[index] - interval.start
                return SampleInterval(word.start_sample + offset, word.end_sample + offset)
        raise KeyError(f"word {word.id} crosses or misses a retained interval")

    def boundary_manifest(self) -> tuple[dict[str, int], ...]:
        """Per-clip boundary record kept with a spliced export (plan 7.8).

        Source positions stay local: they are never uploaded, because a removal
        map would explain what was deleted (plan section 7, last paragraph).
        """
        return tuple(
            {
                "clip_index": index,
                "source_start": interval.start,
                "source_end": interval.end,
                "destination_start": self.offsets[index],
                "destination_end": self.offsets[index] + interval.length,
            }
            for index, interval in enumerate(self.retained)
        )


@dataclass(frozen=True, slots=True)
class ResampleMap:
    """Mapping between the stored signal and canonical 16 kHz sample indices.

    Capture may arrive at 44.1 or 48 kHz.  The canonical stream is what the ASR
    sees and what spans refer to, but cuts must apply to the exact stored
    signal, so the mapping is kept rather than recomputed from a ratio at use
    time (plan section 6).
    """

    source_rate: int
    canonical_rate: int = CANONICAL_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.source_rate <= 0 or self.canonical_rate <= 0:
            raise ValueError("sample rates must be positive")

    @property
    def ratio(self) -> float:
        return self.source_rate / self.canonical_rate

    def to_source_sample(self, canonical_sample: int) -> int:
        return int(math.floor(canonical_sample * self.ratio))

    def to_canonical_sample(self, source_sample: int) -> int:
        return int(math.floor(source_sample / self.ratio))

    def removal_to_source(self, interval: SampleInterval) -> SampleInterval:
        """Round a removal interval outward in the source timeline."""
        start = int(math.floor(interval.start * self.ratio))
        end = int(math.ceil(interval.end * self.ratio))
        return SampleInterval(max(0, start), max(0, end))

    def retained_to_source(self, interval: SampleInterval) -> SampleInterval:
        """Round a retained interval inward in the source timeline."""
        start = int(math.ceil(interval.start * self.ratio))
        end = int(math.floor(interval.end * self.ratio))
        return SampleInterval(start, max(start, end))


class AlignmentProblem(str):
    """Reason code describing why an alignment is unusable."""


def validate_alignment(
    words: Sequence[Word],
    bound: SampleInterval,
    *,
    min_confidence: float,
) -> tuple[str, ...]:
    """Return reason codes for anything that makes a cut unprovable.

    Plan section 7.7: missing, overlapping, non-monotonic, low-confidence or
    transcript-inconsistent timings discard the utterance or the session.  This
    function only reports; the caller decides the scope of the discard.
    """
    problems: list[str] = []
    previous_end = bound.start
    for word in words:
        if not word.text.strip():
            problems.append("empty_word_text")
        if word.start_sample == word.end_sample:
            problems.append("zero_length_word")
        if word.start_sample < previous_end:
            problems.append("overlapping_or_nonmonotonic")
        if word.start_sample < bound.start or word.end_sample > bound.end:
            problems.append("word_outside_audio")
        if word.confidence < min_confidence:
            problems.append("low_confidence_word")
        if word.timing_unreliable:
            problems.append("unreliable_timing")
        previous_end = word.end_sample
    return tuple(dict.fromkeys(problems))
