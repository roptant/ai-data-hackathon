"""Benchmark harness (plan section 4 and milestone 1).

Milestone 1 requires benchmarking both model roles on an 8 GB CPU-only machine
and a 16 GB reference machine, recording exact OS, CPU, memory, model,
quantization and timings, and deciding supported configurations *from evidence*.
This harness produces that record.

Two honesty rules are built in:

* peak memory is measured from the operating system where possible, never
  derived from weight-file size,
* every target the plan states is provisional is compared explicitly, and the
  result says which targets were not met rather than rounding them away.

With no model chosen the harness still runs against the scripted backends,
which measures the harness and the pipeline overhead - and reports plainly that
the model figures are absent.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from dictation.asr.base import AsrWorker
from dictation.capture.devices import synthetic_speech
from dictation.config import Settings
from dictation.privacy.classifier.base import (
    ClassifierWindow,
    PrivacyClassifier,
    build_windows,
)
from dictation.types import CANONICAL_SAMPLE_RATE, CANONICAL_SAMPLE_WIDTH, Transcript, sentences_from_words

#: Plan section 4 targets.  Provisional until confirmed on named machines.
TARGET_REALTIME_FACTOR = 1.0
TARGET_FINAL_INSERTION_MS = 2_000
TARGET_PARTIAL_UPDATE_MS = 1_000
TARGET_PEAK_MEMORY_MIB = 4_096


@dataclass(frozen=True, slots=True)
class MachineProfile:
    """Exactly what the plan asks to record about the machine."""

    os_name: str
    os_release: str
    os_version: str
    machine: str
    processor: str
    cpu_count: int
    total_memory_mib: int
    python_version: str

    @classmethod
    def detect(cls) -> MachineProfile:
        return cls(
            os_name=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor() or "unknown",
            cpu_count=os.cpu_count() or 0,
            total_memory_mib=total_memory_mib(),
            python_version=sys.version.split()[0],
        )


def total_memory_mib() -> int:
    """Total physical memory, or 0 when it cannot be determined."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys // (1024 * 1024))
        return 0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 0


def peak_memory_mib() -> int:
    """Peak resident set size of this process, or 0 when unavailable."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        if ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            handle, ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize // (1024 * 1024))
        return 0
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kibibytes, macOS reports bytes.
        return int(peak // 1024) if sys.platform == "linux" else int(peak // (1024 * 1024))
    except (ImportError, OSError):  # pragma: no cover
        return 0


@dataclass(slots=True)
class RoleMeasurement:
    role: str
    model_id: str
    model_revision: str = ""
    quantization: str = ""
    runs: int = 0
    audio_ms: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    failures: int = 0
    note: str = ""

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return ordered[index]

    @property
    def realtime_factor(self) -> float:
        if not self.audio_ms:
            return 0.0
        return sum(self.latencies_ms) / self.audio_ms

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.update(
            {
                "median_ms": round(self.median_ms, 2),
                "p95_ms": round(self.p95_ms, 2),
                "realtime_factor": round(self.realtime_factor, 4),
            }
        )
        return data


@dataclass(slots=True)
class BenchmarkReport:
    machine: MachineProfile
    measurements: list[RoleMeasurement] = field(default_factory=list)
    peak_memory_mib: int = 0
    unmet_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, object]:
        return {
            "machine": asdict(self.machine),
            "measurements": [measurement.as_dict() for measurement in self.measurements],
            "peak_memory_mib": self.peak_memory_mib,
            "targets": {
                "realtime_factor": TARGET_REALTIME_FACTOR,
                "final_insertion_ms": TARGET_FINAL_INSERTION_MS,
                "partial_update_ms": TARGET_PARTIAL_UPDATE_MS,
                "peak_memory_mib": TARGET_PEAK_MEMORY_MIB,
            },
            "unmet_targets": self.unmet_targets,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def summary(self) -> str:
        lines = [
            f"machine: {self.machine.os_name} {self.machine.os_release} "
            f"{self.machine.machine} / {self.machine.cpu_count} CPUs / "
            f"{self.machine.total_memory_mib} MiB RAM",
            f"peak memory: {self.peak_memory_mib} MiB (target below {TARGET_PEAK_MEMORY_MIB})",
            "",
        ]
        for measurement in self.measurements:
            lines.append(
                f"  {measurement.role:<8} {measurement.model_id or '(none)':<32} "
                f"runs={measurement.runs} median={measurement.median_ms:.0f}ms "
                f"p95={measurement.p95_ms:.0f}ms rtf={measurement.realtime_factor:.2f}"
                + (f"  [{measurement.note}]" if measurement.note else "")
            )
        if self.unmet_targets:
            lines.extend(["", "unmet targets:"])
            lines.extend(f"  - {target}" for target in self.unmet_targets)
        if self.notes:
            lines.extend(["", "notes:"])
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def benchmark_asr(
    asr: AsrWorker,
    *,
    durations_s: Sequence[float] = (2.0, 5.0, 10.0),
    repeats: int = 3,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
) -> RoleMeasurement:
    """Time final transcription over several utterance lengths."""
    capabilities = asr.capabilities
    measurement = RoleMeasurement(
        role="asr",
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
        quantization=capabilities.quantization,
    )
    asr.warm_up()
    for seconds in durations_s:
        pcm = synthetic_speech(
            [(0.2, seconds - 0.2)], total_seconds=seconds, sample_rate=sample_rate
        )
        for _ in range(repeats):
            began = time.perf_counter()
            try:
                asr.transcribe(pcm, session_id="bench", sample_rate=sample_rate)
            except Exception:  # noqa: BLE001 - a failure is a measurement
                measurement.failures += 1
                continue
            measurement.latencies_ms.append((time.perf_counter() - began) * 1000)
            measurement.audio_ms += int(len(pcm) / CANONICAL_SAMPLE_WIDTH / sample_rate * 1000)
            measurement.runs += 1
    return measurement


def benchmark_privacy(
    classifier: PrivacyClassifier,
    transcript: Transcript,
    *,
    settings: Settings | None = None,
    repeats: int = 1,
) -> RoleMeasurement:
    """Time classification per window over a real transcript shape."""
    settings = settings or Settings()
    capabilities = classifier.capabilities
    measurement = RoleMeasurement(
        role="privacy",
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
        quantization=capabilities.quantization,
    )
    sentences = sentences_from_words(transcript.words, sample_rate=transcript.sample_rate)
    windows: tuple[ClassifierWindow, ...] = build_windows(
        transcript,
        sentences,
        window_words=settings.privacy.window_words,
        overlap_words=settings.privacy.window_overlap_words,
    )
    classifier.warm_up()
    for _ in range(repeats):
        for window in windows:
            began = time.perf_counter()
            try:
                classifier.classify(window, timeout_s=settings.privacy.classifier_timeout_s)
            except Exception:  # noqa: BLE001
                measurement.failures += 1
                continue
            measurement.latencies_ms.append((time.perf_counter() - began) * 1000)
            measurement.runs += 1
    return measurement


def check_targets(report: BenchmarkReport) -> list[str]:
    """Which provisional targets were not met by these measurements."""
    unmet: list[str] = []
    for measurement in report.measurements:
        if measurement.role == "asr" and measurement.realtime_factor > TARGET_REALTIME_FACTOR:
            unmet.append(
                f"asr_realtime_factor={measurement.realtime_factor:.2f} exceeds "
                f"{TARGET_REALTIME_FACTOR}"
            )
        if measurement.role == "asr" and measurement.p95_ms > TARGET_FINAL_INSERTION_MS:
            unmet.append(
                f"asr_p95={measurement.p95_ms:.0f}ms exceeds {TARGET_FINAL_INSERTION_MS}ms"
            )
        if measurement.failures:
            unmet.append(f"{measurement.role}_failures={measurement.failures}")
    if report.peak_memory_mib > TARGET_PEAK_MEMORY_MIB:
        unmet.append(f"peak_memory={report.peak_memory_mib}MiB exceeds {TARGET_PEAK_MEMORY_MIB}MiB")
    return unmet


def run_benchmarks(
    asr: AsrWorker,
    classifier: PrivacyClassifier | None,
    transcript: Transcript | None = None,
    *,
    settings: Settings | None = None,
) -> BenchmarkReport:
    """Benchmark both roles and compare against the provisional targets."""
    report = BenchmarkReport(machine=MachineProfile.detect())
    report.measurements.append(benchmark_asr(asr))
    if classifier is not None and transcript is not None:
        report.measurements.append(benchmark_privacy(classifier, transcript, settings=settings))
    elif classifier is not None:
        report.notes.append("privacy role not benchmarked: no transcript supplied")
    else:
        report.notes.append("privacy role not benchmarked: no classifier configured")
    report.peak_memory_mib = peak_memory_mib()
    if any(m.model_id.endswith("mock") for m in report.measurements):
        report.notes.append(
            "figures include scripted stand-in backends; they measure the pipeline, "
            "not a model. Choose and fetch models before treating these as evidence."
        )
    report.unmet_targets = check_targets(report)
    return report
