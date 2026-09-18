"""Turning word spans into audio cuts (plan sections 7.5-7.7).

The order of operations is the safety argument:

1. expand each span to its whole sentence, which removes the surrounding clues
   and avoids training on unnatural fragments,
2. convert word ranges to sample intervals through the ASR alignment,
3. pad each interval outward, merge what overlaps,
4. then pull in **every** neighbouring word whose audio intersects the padded
   cut, and iterate until that set stops growing.

Step 4 is what keeps text and audio consistent.  Padding can reach into the
word next door; if that word stayed in the transcript its audio would be gone,
and the training pair would claim a word that is not in the clip.
"""

from __future__ import annotations

from dataclasses import dataclass

from dictation.config import PrivacySettings
from dictation.time_map import ms_to_samples, normalize, pad_all
from dictation.types import SampleInterval, Sentence, Span, Transcript


@dataclass(frozen=True, slots=True)
class RemovalPlan:
    """What will be cut, in both domains."""

    intervals: tuple[SampleInterval, ...]
    removed_word_ids: frozenset[int]
    removed_sentence_indices: frozenset[int]
    padding_samples: int
    iterations: int = 0

    @property
    def removed_samples(self) -> int:
        return sum(interval.length for interval in self.intervals)


def expand_to_sentences(
    spans: tuple[Span, ...],
    sentences: tuple[Sentence, ...],
    *,
    expand_all: bool,
) -> tuple[frozenset[int], frozenset[int]]:
    """Return ``(word_ids, sentence_indices)`` selected by the spans.

    A span with ``drop_sentence`` - the default - takes every sentence it
    touches.  ``drop_words`` keeps only its own words, and is only reachable
    when word-level cuts are enabled and alignment validated.
    """
    word_ids: set[int] = set()
    sentence_indices: set[int] = set()
    for span in spans:
        sentence_scope = expand_all or span.action.value == "drop_sentence"
        if sentence_scope:
            touched = [sentence for sentence in sentences if sentence.overlaps_span(span)]
            for sentence in touched:
                sentence_indices.add(sentence.index)
                word_ids.update(range(sentence.start_word_id, sentence.end_word_id_exclusive))
            if touched:
                continue
            # No sentence covers it (an empty sentence list, for instance):
            # fall back to the span's own words rather than dropping nothing.
        word_ids.update(range(span.start_word_id, span.end_word_id_exclusive))
    return frozenset(word_ids), frozenset(sentence_indices)


def plan_removals(
    transcript: Transcript,
    spans: tuple[Span, ...],
    sentences: tuple[Sentence, ...],
    *,
    settings: PrivacySettings | None = None,
    extra_sentence_indices: frozenset[int] = frozenset(),
    bound: SampleInterval | None = None,
) -> RemovalPlan:
    """Build the removal plan for a frozen transcript.

    ``extra_sentence_indices`` lets the caller drop sentences for reasons other
    than sensitivity - unreliable timings, low confidence - through the same
    interval machinery, so those cuts get the same padding and word pull-in.
    """
    settings = settings or PrivacySettings()
    audio = bound or transcript.span
    padding = ms_to_samples(settings.padding_ms, transcript.sample_rate)

    word_ids, sentence_indices = expand_to_sentences(
        spans, sentences, expand_all=settings.expand_to_sentence
    )
    sentence_indices = set(sentence_indices) | set(extra_sentence_indices)
    for index in extra_sentence_indices:
        if 0 <= index < len(sentences):
            sentence = sentences[index]
            word_ids = word_ids | frozenset(
                range(sentence.start_word_id, sentence.end_word_id_exclusive)
            )

    if not word_ids:
        return RemovalPlan((), frozenset(), frozenset(sentence_indices), padding)

    known_ids = {word.id for word in transcript.words}
    selected = {word_id for word_id in word_ids if word_id in known_ids}
    intervals = normalize(
        transcript.word_by_id(word_id).interval for word_id in sorted(selected)
    )
    intervals = pad_all(intervals, padding, audio)

    # Iterate to a fixed point: a pulled-in word contributes its own audio,
    # which can reach a further word.
    iterations = 0
    while True:
        iterations += 1
        grown = set(selected)
        for word in transcript.words:
            if word.id in grown:
                continue
            if any(word.interval.intersects(interval) for interval in intervals):
                grown.add(word.id)
        if grown == selected:
            break
        selected = grown
        intervals = pad_all(
            normalize(transcript.word_by_id(word_id).interval for word_id in sorted(selected)),
            padding,
            audio,
        )
        if iterations > 32:  # pragma: no cover - defensive
            break

    # Any sentence that lost a word is recorded as removed, so the caller can
    # tell whether a retained clip still contains whole utterances.
    for sentence in sentences:
        if any(
            word_id in selected
            for word_id in range(sentence.start_word_id, sentence.end_word_id_exclusive)
        ):
            sentence_indices.add(sentence.index)

    return RemovalPlan(
        intervals=intervals,
        removed_word_ids=frozenset(selected),
        removed_sentence_indices=frozenset(sentence_indices),
        padding_samples=padding,
        iterations=iterations,
    )


def words_fully_inside(
    transcript: Transcript,
    interval: SampleInterval,
    *,
    excluded: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Word IDs whose audio lies wholly within ``interval``.

    A word that straddles a boundary is not returned: its audio is incomplete,
    so keeping its label would misstate what the clip contains.
    """
    return tuple(
        word.id
        for word in transcript.words
        if word.id not in excluded
        and interval.start <= word.start_sample
        and word.end_sample <= interval.end
    )
