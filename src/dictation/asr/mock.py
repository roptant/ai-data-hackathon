"""Deterministic ASR backends for development, tests and demos.

These exist so the rest of the pipeline - streaming reconciliation, insertion,
the privacy classifier, interval mathematics, quality gates, the queue - can be
exercised and tested end to end before a model has been chosen.  They never
pretend to be recognition: :class:`ScriptedAsr` returns exactly the words it was
given, with the timings it was given.

Production behaviour differs in one way that matters: a real recogniser can omit
or mistranscribe a sensitive word, which is the failure mode the plan warns
about (plan section 2).  :func:`script_with_mistranscription` makes that case
reproducible in evaluation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from dictation.asr.base import AsrCapabilities, PartialUpdate, WorkerStats
from dictation.errors import AsrUnavailable
from dictation.types import (
    CANONICAL_SAMPLE_RATE,
    CANONICAL_SAMPLE_WIDTH,
    Segment,
    Transcript,
    Word,
)

_PUNCTUATION = re.compile(r"[^\w'\-]+", re.UNICODE)


def spoken_form(display: str) -> str:
    """The training label for a display token: no punctuation, lower case."""
    return _PUNCTUATION.sub("", display).lower()


@dataclass(frozen=True, slots=True)
class ScriptedWord:
    display: str
    start_ms: int
    end_ms: int
    confidence: float = 0.95
    timing_unreliable: bool = False


def script_from_text(
    text: str,
    *,
    word_ms: int = 320,
    gap_ms: int = 60,
    sentence_gap_ms: int = 800,
    start_ms: int = 0,
    confidence: float = 0.95,
) -> tuple[ScriptedWord, ...]:
    """Lay out evenly spaced words, with a real pause between sentences.

    The sentence gap matters: dictated text often has no punctuation, and the
    sentence splitter uses pauses as well as full stops.
    """
    cursor = start_ms
    out: list[ScriptedWord] = []
    for token in text.split():
        out.append(ScriptedWord(token, cursor, cursor + word_ms, confidence))
        cursor += word_ms
        cursor += sentence_gap_ms if token.rstrip().endswith((".", "!", "?")) else gap_ms
    return tuple(out)


def script_with_mistranscription(
    text: str,
    replacements: dict[str, str],
    **kwargs: object,
) -> tuple[ScriptedWord, ...]:
    """Build a script where the recogniser got some words wrong.

    Used to evaluate the case the plan calls out: the privacy model only sees
    ASR output, so an identifier the recogniser mangled may never be classified.
    """
    script = script_from_text(text, **kwargs)  # type: ignore[arg-type]
    return tuple(
        ScriptedWord(
            replacements.get(word.display, word.display),
            word.start_ms,
            word.end_ms,
            word.confidence,
            word.timing_unreliable,
        )
        for word in script
    )


class ScriptedAsr:
    """Returns a fixed word script, chunked into revisable segments."""

    def __init__(
        self,
        script: Sequence[ScriptedWord],
        *,
        model_id: str = "scripted-mock",
        model_revision: str = "0",
        words_per_segment: int = 6,
        language: str = "en",
        word_timings_experimental: bool = True,
    ) -> None:
        self._script = tuple(script)
        self._words_per_segment = max(1, words_per_segment)
        self._loaded = False
        self.stats = WorkerStats()
        self._capabilities = AsrCapabilities(
            model_id=model_id,
            model_revision=model_revision,
            quantization="none",
            languages=(language,),
            streaming=True,
            word_timestamps=True,
            word_timings_experimental=word_timings_experimental,
        )

    @property
    def capabilities(self) -> AsrCapabilities:
        return self._capabilities

    @property
    def loaded(self) -> bool:
        return self._loaded

    def warm_up(self) -> None:
        self._loaded = True

    def release(self) -> None:
        self._loaded = False

    # -- transcription -------------------------------------------------------

    def _words(self, sample_rate: int) -> tuple[Word, ...]:
        words: list[Word] = []
        for index, scripted in enumerate(self._script):
            display = scripted.display
            text = spoken_form(display)
            if not text:
                continue
            words.append(
                Word(
                    id=index,
                    text=text,
                    display=display,
                    start_sample=int(scripted.start_ms * sample_rate / 1000),
                    end_sample=int(scripted.end_ms * sample_rate / 1000),
                    confidence=scripted.confidence,
                    segment_id=f"segment-{index // self._words_per_segment}",
                    timing_unreliable=scripted.timing_unreliable,
                )
            )
        return tuple(words)

    def transcribe(
        self,
        pcm: bytes,
        *,
        session_id: str,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        language: str = "en",
    ) -> Transcript:
        if not self._loaded:
            self.warm_up()
        began = time.monotonic()
        words = self._words(sample_rate)
        audio_samples = len(pcm) // CANONICAL_SAMPLE_WIDTH
        total = max(audio_samples, words[-1].end_sample if words else 0)
        transcript = Transcript(
            session_id=session_id,
            revision=1,
            words=words,
            sample_rate=sample_rate,
            total_samples=total,
            language=language,
            asr_model=self._capabilities.model_id,
            asr_model_revision=self._capabilities.model_revision,
            word_timings_experimental=self._capabilities.word_timings_experimental,
        )
        self.stats.record(
            audio_ms=int(audio_samples * 1000 / sample_rate),
            compute_ms=int((time.monotonic() - began) * 1000),
        )
        return transcript

    def stream(
        self,
        chunks: Iterable[bytes],
        *,
        session_id: str,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        language: str = "en",
    ) -> Iterator[PartialUpdate]:
        """Emit each segment twice: a rough hypothesis, then a revision.

        This reproduces the behaviour clients must tolerate - later revisions
        replacing earlier text for the same segment id.
        """
        if not self._loaded:
            self.warm_up()
        words = self._words(sample_rate)
        consumed = 0
        emitted: dict[str, int] = {}
        for chunk in chunks:
            consumed += len(chunk) // CANONICAL_SAMPLE_WIDTH
            for segment_id in dict.fromkeys(w.segment_id for w in words):
                in_segment = [w for w in words if w.segment_id == segment_id]
                visible = [w for w in in_segment if w.end_sample <= consumed]
                if not visible:
                    continue
                revision = emitted.get(segment_id, 0) + 1
                if revision > len(in_segment):
                    continue
                emitted[segment_id] = revision
                text = " ".join(w.display_text for w in visible)
                yield PartialUpdate(
                    segment=Segment(
                        id=segment_id,
                        revision=revision,
                        text=text,
                        start_sample=in_segment[0].start_sample,
                        end_sample=visible[-1].end_sample,
                        is_final=False,
                    ),
                    consumed_samples=consumed,
                )


@dataclass
class FailingAsr:
    """Always fails.  Proves that recognition failure is reported, never faked,
    and that no training artifact can result from it."""

    message: str = "synthetic ASR failure"
    stats: WorkerStats = field(default_factory=WorkerStats)

    @property
    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(model_id="failing-mock")

    def warm_up(self) -> None:
        return None

    def release(self) -> None:
        return None

    def transcribe(self, pcm: bytes, **kwargs: object) -> Transcript:
        self.stats.failures += 1
        raise AsrUnavailable(self.message)

    def stream(self, chunks: Iterable[bytes], **kwargs: object) -> Iterator[PartialUpdate]:
        self.stats.failures += 1
        raise AsrUnavailable(self.message)
