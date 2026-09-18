"""Audio sources (plan section 6).

Only the microphone is captured, never system output, and only while the user
has explicitly activated recording.  There is no ambient-listening path in this
module by construction: a source is opened on start and closed on stop.

The real microphone source needs a platform capture binding that is not part of
this reference implementation; :class:`MicrophoneSource` therefore probes for a
backend and reports an actionable failure instead of pretending to record.  The
file and synthetic sources make the whole pipeline runnable and testable.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from dictation.capture.audio_buffer import read_wav
from dictation.errors import CaptureError, CapabilityUnavailable
from dictation.types import CANONICAL_SAMPLE_RATE, CANONICAL_SAMPLE_WIDTH


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    identifier: str
    name: str
    sample_rate: int
    channels: int = 1


@runtime_checkable
class AudioSource(Protocol):
    """A microphone-like source of canonical int16 mono PCM chunks."""

    @property
    def sample_rate(self) -> int: ...

    def open(self) -> DeviceInfo: ...

    def read_chunk(self, max_samples: int) -> bytes:
        """Return up to ``max_samples`` samples, or ``b""`` at end of stream."""

    def close(self) -> None: ...


class _BytesSource:
    """Shared chunking logic for in-memory sources."""

    def __init__(self, pcm: bytes, sample_rate: int = CANONICAL_SAMPLE_RATE, name: str = "bytes") -> None:
        self._pcm = pcm
        self._rate = sample_rate
        self._name = name
        self._cursor = 0
        self._open = False

    @property
    def sample_rate(self) -> int:
        return self._rate

    def open(self) -> DeviceInfo:
        self._cursor = 0
        self._open = True
        return DeviceInfo(identifier=self._name, name=self._name, sample_rate=self._rate)

    def read_chunk(self, max_samples: int) -> bytes:
        if not self._open:
            raise CaptureError("source is not open")
        start = self._cursor
        end = min(len(self._pcm), start + max_samples * CANONICAL_SAMPLE_WIDTH)
        self._cursor = end
        return self._pcm[start:end]

    def close(self) -> None:
        self._open = False


class FileSource(_BytesSource):
    """Canonical-format WAV file, used for fixtures and offline evaluation."""

    def __init__(self, path: Path) -> None:
        pcm, rate = read_wav(path)
        super().__init__(pcm, rate, name=f"file:{path.name}")


class MemorySource(_BytesSource):
    """Pre-rendered PCM, used by tests and the synthetic pipeline demo."""

    def __init__(self, pcm: bytes, sample_rate: int = CANONICAL_SAMPLE_RATE) -> None:
        super().__init__(pcm, sample_rate, name="memory")


class SilenceSource(_BytesSource):
    """Digital silence.  Exercises the empty-session guard."""

    def __init__(self, seconds: float, sample_rate: int = CANONICAL_SAMPLE_RATE) -> None:
        super().__init__(bytes(int(seconds * sample_rate) * CANONICAL_SAMPLE_WIDTH), sample_rate, "silence")


class MicrophoneSource:
    """Real capture device.

    Placeholder by design: the platform capture backend is a dependency the
    product owner has not selected yet, in the same way the models have not
    been.  Opening it raises :class:`CapabilityUnavailable` with the reason,
    which the capture coordinator surfaces as a microphone failure state
    instead of producing an empty recording.
    """

    BACKEND_CANDIDATES = ("sounddevice", "pyaudio", "soundcard")

    def __init__(self, device: str | None = None, sample_rate: int = CANONICAL_SAMPLE_RATE) -> None:
        self.device = device
        self._rate = sample_rate
        self._backend = self.detect_backend()

    @property
    def sample_rate(self) -> int:
        return self._rate

    @classmethod
    def detect_backend(cls) -> str | None:
        import importlib.util

        for candidate in cls.BACKEND_CANDIDATES:
            if importlib.util.find_spec(candidate) is not None:
                return candidate
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.detect_backend() is not None

    def open(self) -> DeviceInfo:
        raise CapabilityUnavailable(
            "no microphone capture backend is wired up. Install one of "
            f"{', '.join(self.BACKEND_CANDIDATES)} and implement the adapter in "
            "dictation.capture.devices.MicrophoneSource, or run with a file "
            "source. Dictation reports this as a microphone failure rather "
            "than recording silence."
        )

    def read_chunk(self, max_samples: int) -> bytes:  # pragma: no cover - unreachable
        raise CapabilityUnavailable("microphone source is not open")

    def close(self) -> None:
        return None


def synthetic_speech(
    segments: list[tuple[float, float]],
    *,
    total_seconds: float,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    amplitude: int = 9000,
    noise: int = 60,
) -> bytes:
    """Render tone bursts over low noise for deterministic pipeline tests.

    ``segments`` are ``(start_seconds, end_seconds)`` pairs treated as speech.
    The result is not speech a recogniser could transcribe; it exists so that
    energy-based gates, interval mathematics and clip extraction can be tested
    against known boundaries.
    """
    samples = array.array("h", bytes(int(total_seconds * sample_rate) * CANONICAL_SAMPLE_WIDTH))
    for index in range(len(samples)):
        samples[index] = int(noise * math.sin(index * 0.37)) if noise else 0
    for start_s, end_s in segments:
        start = max(0, int(start_s * sample_rate))
        end = min(len(samples), int(end_s * sample_rate))
        for index in range(start, end):
            phase = 2 * math.pi * 180 * (index - start) / sample_rate
            samples[index] = int(amplitude * math.sin(phase))
    return samples.tobytes()


def chunks_of(pcm: bytes, chunk_samples: int) -> Iterator[bytes]:
    step = chunk_samples * CANONICAL_SAMPLE_WIDTH
    for start in range(0, len(pcm), step):
        yield pcm[start : start + step]
