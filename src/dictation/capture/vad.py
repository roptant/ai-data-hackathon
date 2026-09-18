"""Non-neural voice activity detection (plan section 3).

The plan ships exactly two learned model roles, so activity detection is
deterministic signal processing: framed RMS energy against an adaptive noise
floor, with hysteresis so a short dip inside a word does not split it.  This is
the interface a WebRTC VAD binding would drop into unchanged.

It answers two questions and no others: did anyone speak at all (so an empty
session cannot become hallucinated text), and how much of a retained clip is
speech rather than room noise.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass

from dictation.types import CANONICAL_SAMPLE_RATE, SampleInterval
from dictation.time_map import normalize


@dataclass(frozen=True, slots=True)
class VadSettings:
    frame_ms: int = 20
    #: Speech must exceed the noise floor by this many decibels.
    threshold_db: float = 9.0
    #: Absolute floor so that digital silence never counts as speech.
    absolute_floor_db: float = -55.0
    #: Frames of speech required to open, and of silence required to close.
    open_frames: int = 2
    close_frames: int = 8


def _frame_rms_db(samples: array.array, start: int, end: int) -> float:
    if end <= start:
        return -120.0
    total = 0.0
    for index in range(start, end):
        value = samples[index] / 32768.0
        total += value * value
    mean = total / (end - start)
    if mean <= 1e-12:
        return -120.0
    return 10.0 * math.log10(mean)


class EnergyVad:
    """Framed-energy detector over canonical int16 PCM."""

    def __init__(self, settings: VadSettings | None = None, sample_rate: int = CANONICAL_SAMPLE_RATE) -> None:
        self.settings = settings or VadSettings()
        self.sample_rate = sample_rate

    @property
    def frame_samples(self) -> int:
        return max(1, int(self.sample_rate * self.settings.frame_ms / 1000))

    def frame_levels(self, pcm: bytes) -> list[float]:
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        frame = self.frame_samples
        return [
            _frame_rms_db(samples, start, min(start + frame, len(samples)))
            for start in range(0, len(samples), frame)
        ]

    def noise_floor_db(self, levels: list[float]) -> float:
        """Robust noise floor: the 20th percentile of frame energies."""
        if not levels:
            return -120.0
        ordered = sorted(levels)
        index = max(0, int(len(ordered) * 0.2) - 1)
        return ordered[index]

    def frame_flags(self, pcm: bytes) -> list[bool]:
        """Per-frame speech decision with hysteresis."""
        levels = self.frame_levels(pcm)
        if not levels:
            return []
        floor = self.noise_floor_db(levels)
        threshold = max(floor + self.settings.threshold_db, self.settings.absolute_floor_db)
        flags: list[bool] = []
        speaking = False
        run = 0
        for level in levels:
            loud = level >= threshold
            if speaking:
                run = 0 if loud else run + 1
                if run >= self.settings.close_frames:
                    speaking = False
                    run = 0
            else:
                run = run + 1 if loud else 0
                if run >= self.settings.open_frames:
                    speaking = True
                    run = 0
            flags.append(speaking)
        return flags

    def speech_ratio(self, pcm: bytes) -> float:
        flags = self.frame_flags(pcm)
        if not flags:
            return 0.0
        return sum(1 for flag in flags if flag) / len(flags)

    def is_effectively_silent(self, pcm: bytes, *, min_ratio: float = 0.08) -> bool:
        """True when the session should not produce text or a training example."""
        return self.speech_ratio(pcm) < min_ratio

    def speech_intervals(self, pcm: bytes, *, offset: int = 0) -> tuple[SampleInterval, ...]:
        """Sample intervals judged to contain speech."""
        frame = self.frame_samples
        intervals: list[SampleInterval] = []
        start: int | None = None
        flags = self.frame_flags(pcm)
        for index, flag in enumerate(flags):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                intervals.append(
                    SampleInterval(offset + start * frame, offset + index * frame)
                )
                start = None
        if start is not None:
            intervals.append(SampleInterval(offset + start * frame, offset + len(flags) * frame))
        return normalize(intervals)
