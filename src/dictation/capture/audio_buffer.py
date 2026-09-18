"""Bounded PCM buffers (plan section 6).

Buffers are bounded and overflow is an explicit error.  With contribution off,
audio lives in memory and is discarded after delivery; only an opted-in session
persists encrypted chunks.  ``clear`` overwrites the bytes before releasing
them, which removes the copy this process controls and says nothing about swap
or OS-level copies (plan section 6, last paragraph).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from dictation.errors import BufferOverflow
from dictation.types import (
    CANONICAL_CHANNELS,
    CANONICAL_SAMPLE_RATE,
    CANONICAL_SAMPLE_WIDTH,
    SampleInterval,
)


@dataclass(slots=True)
class BoundedPcmBuffer:
    """Append-only int16 mono PCM buffer with a hard sample ceiling."""

    max_samples: int
    sample_rate: int = CANONICAL_SAMPLE_RATE
    _data: bytearray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self._data is None:
            self._data = bytearray()

    @classmethod
    def for_seconds(cls, seconds: float, sample_rate: int = CANONICAL_SAMPLE_RATE) -> BoundedPcmBuffer:
        return cls(max_samples=int(seconds * sample_rate), sample_rate=sample_rate)

    # -- state ---------------------------------------------------------------

    @property
    def sample_count(self) -> int:
        return len(self._data) // CANONICAL_SAMPLE_WIDTH

    @property
    def duration_ms(self) -> int:
        return int(round(self.sample_count * 1000 / self.sample_rate))

    @property
    def remaining_samples(self) -> int:
        return self.max_samples - self.sample_count

    @property
    def interval(self) -> SampleInterval:
        return SampleInterval(0, self.sample_count)

    def __len__(self) -> int:
        return self.sample_count

    # -- writing -------------------------------------------------------------

    def append(self, pcm: bytes | bytearray | memoryview) -> int:
        """Append raw int16 PCM.  Returns the new sample count.

        Raises :class:`~dictation.errors.BufferOverflow` rather than dropping
        audio, because a silent drop would desynchronise every timestamp.
        """
        payload = bytes(pcm)
        if len(payload) % CANONICAL_SAMPLE_WIDTH:
            raise ValueError("PCM payload is not a whole number of int16 samples")
        incoming = len(payload) // CANONICAL_SAMPLE_WIDTH
        if incoming > self.remaining_samples:
            raise BufferOverflow(
                f"buffer holds {self.sample_count} of {self.max_samples} samples; "
                f"cannot append {incoming}"
            )
        self._data.extend(payload)
        return self.sample_count

    def append_silence(self, samples: int) -> int:
        return self.append(bytes(samples * CANONICAL_SAMPLE_WIDTH))

    # -- reading -------------------------------------------------------------

    def read(self, interval: SampleInterval | None = None) -> bytes:
        if interval is None:
            return bytes(self._data)
        if interval.end > self.sample_count:
            raise ValueError(
                f"interval [{interval.start}, {interval.end}) exceeds buffer of "
                f"{self.sample_count} samples"
            )
        return bytes(
            self._data[
                interval.start * CANONICAL_SAMPLE_WIDTH : interval.end * CANONICAL_SAMPLE_WIDTH
            ]
        )

    def read_intervals(self, intervals: list[SampleInterval]) -> bytes:
        """Concatenate the given intervals in order (the spliced export)."""
        return b"".join(self.read(interval) for interval in intervals)

    # -- lifecycle -----------------------------------------------------------

    def clear(self) -> None:
        """Overwrite and drop the samples this process holds."""
        for index in range(len(self._data)):
            self._data[index] = 0
        self._data = bytearray()

    def drain_into(self, other: BoundedPcmBuffer) -> int:
        """Move warm-up audio into the session buffer, then clear this one."""
        moved = other.append(self.read())
        self.clear()
        return moved


def write_wav(path: Path, pcm: bytes, sample_rate: int = CANONICAL_SAMPLE_RATE) -> None:
    """Write mono 16-bit PCM.  Used for spliced exports and fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CANONICAL_CHANNELS)
        handle.setsampwidth(CANONICAL_SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def read_wav(path: Path) -> tuple[bytes, int]:
    """Read mono 16-bit PCM, rejecting anything not in canonical form.

    Refusing to guess keeps the sample-index timing contract intact; conversion
    belongs in the capture pipeline where the resample map is recorded.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != CANONICAL_CHANNELS:
            raise ValueError("only mono audio is supported")
        if handle.getsampwidth() != CANONICAL_SAMPLE_WIDTH:
            raise ValueError("only 16-bit PCM is supported")
        return handle.readframes(handle.getnframes()), handle.getframerate()
