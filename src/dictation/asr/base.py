"""Speech recognition worker interface (plan sections 3, 4, 6).

The worker produces revisable partial hypotheses during recording and one
frozen canonical transcript after stop.  Word timings come from the recogniser
and are treated as *evidence, not proof*: whisper.cpp documents word timestamps
as experimental, so :attr:`AsrCapabilities.word_timings_experimental` propagates
into the transcript and the dataset builder refuses fine-grained word cuts
unless alignment has been validated (plan sections 4 and 7.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Protocol, runtime_checkable

from dictation.types import CANONICAL_SAMPLE_RATE, Segment, Transcript


@dataclass(frozen=True, slots=True)
class AsrCapabilities:
    """What a backend can actually do.  Reported, never assumed."""

    model_id: str = ""
    model_revision: str = ""
    quantization: str = ""
    languages: tuple[str, ...] = ("en",)
    streaming: bool = False
    word_timestamps: bool = False
    word_timings_experimental: bool = True
    #: Measured on this machine by the benchmark harness; 0.0 means unmeasured.
    realtime_factor: float = 0.0
    #: Peak resident memory measured for this backend, in MiB; 0 means unknown.
    peak_memory_mib: int = 0

    def supports(self, language: str) -> bool:
        return language in self.languages


@dataclass(slots=True)
class PartialUpdate:
    """One streaming hypothesis for a segment."""

    segment: Segment
    #: Sample count consumed when this hypothesis was produced.
    consumed_samples: int = 0


@runtime_checkable
class AsrWorker(Protocol):
    """Minimal contract the capture coordinator depends on."""

    @property
    def capabilities(self) -> AsrCapabilities: ...

    def warm_up(self) -> None:
        """Load weights.  Called off the recording path; may be slow."""

    def release(self) -> None:
        """Free weights.  Used on low-memory machines before the privacy model
        loads (plan section 4)."""

    def transcribe(
        self,
        pcm: bytes,
        *,
        session_id: str,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        language: str = "en",
    ) -> Transcript:
        """Produce the canonical, frozen transcript for a finished recording."""

    def stream(
        self,
        chunks: Iterable[bytes],
        *,
        session_id: str,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        language: str = "en",
    ) -> Iterator[PartialUpdate]:
        """Yield revisable partial hypotheses while recording."""


@dataclass(slots=True)
class WorkerStats:
    """Counters for the status UI and the benchmark harness."""

    sessions: int = 0
    audio_ms: int = 0
    compute_ms: int = 0
    failures: int = 0
    history: list[float] = field(default_factory=list)

    def record(self, audio_ms: int, compute_ms: int) -> None:
        self.sessions += 1
        self.audio_ms += audio_ms
        self.compute_ms += compute_ms
        if audio_ms:
            self.history.append(compute_ms / audio_ms)

    @property
    def realtime_factor(self) -> float:
        """Compute time divided by audio time.  Must stay below 1.0 under
        sustained operation, or the backlog grows without bound (plan 4)."""
        if not self.audio_ms:
            return 0.0
        return self.compute_ms / self.audio_ms
