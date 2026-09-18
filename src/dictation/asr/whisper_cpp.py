"""whisper.cpp ASR backend (plan sections 3 and 4).

The model itself is an open decision: the registry ships *unresolved candidates*
(quantized Whisper base and small) with no URL or checksum, and this backend
refuses to run until one is chosen and fetched.  That refusal is the intended
behaviour, not a gap - it keeps an unvetted artifact from being pulled in
silently, and it is where ``dictation models`` plugs in.

The JSON parser is separated from the subprocess so that timestamp handling,
low-probability tokens and malformed output are unit-testable without the
binary or the weights.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from dictation.asr.base import AsrCapabilities, PartialUpdate, WorkerStats
from dictation.asr.mock import spoken_form
from dictation.capture.audio_buffer import write_wav
from dictation.errors import AsrUnavailable, ModelNotInstalled, WorkerError
from dictation.models.registry import ModelRole, ModelSpec
from dictation.types import CANONICAL_SAMPLE_RATE, CANONICAL_SAMPLE_WIDTH, Transcript, Word

#: Probability below which whisper.cpp token timing is treated as unreliable.
#: Provisional: it gates fine-grained word cuts, not the transcript itself.
TOKEN_PROBABILITY_FLOOR = 0.35

BINARY_CANDIDATES = ("whisper-cli", "whisper-cpp", "main")


def find_binary(explicit: Path | None = None) -> Path | None:
    """Locate a bundled or system whisper.cpp binary.

    Release builds bundle the binary so users need no compiler or Python
    (plan section 3); development falls back to PATH.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    env = os.environ.get("DICTATION_WHISPER_BINARY")
    if env:
        candidate = Path(env)
        return candidate if candidate.exists() else None
    for name in BINARY_CANDIDATES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _offsets_to_samples(offset_ms: float, sample_rate: int) -> int:
    return max(0, int(round(offset_ms * sample_rate / 1000.0)))


def parse_whisper_json(
    payload: dict[str, Any],
    *,
    session_id: str,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    total_samples: int = 0,
    model_id: str = "",
    model_revision: str = "",
    language: str = "en",
    words_per_segment: int = 6,
) -> Transcript:
    """Convert whisper.cpp JSON into a frozen transcript.

    Accepts both the plain ``transcription`` array and the ``--output-json-full``
    form with per-token offsets.  Tokens without usable offsets are kept in the
    transcript - the user still gets their text - but marked
    ``timing_unreliable``, which blocks word-level cuts and, if the recogniser
    marks enough of them, makes the dataset builder reject the session rather
    than cut audio it cannot place (plan section 7.7).
    """
    entries = payload.get("transcription")
    if not isinstance(entries, list):
        raise WorkerError("whisper.cpp output has no transcription array")

    words: list[Word] = []

    def add(display: str, start_ms: float | None, end_ms: float | None, probability: float | None) -> None:
        text = spoken_form(display)
        if not text:
            return
        unreliable = start_ms is None or end_ms is None
        start = _offsets_to_samples(start_ms or 0.0, sample_rate)
        end = _offsets_to_samples(end_ms if end_ms is not None else (start_ms or 0.0), sample_rate)
        if end < start:
            start, end = end, start
            unreliable = True
        if probability is not None and probability < TOKEN_PROBABILITY_FLOOR:
            unreliable = True
        index = len(words)
        words.append(
            Word(
                id=index,
                text=text,
                display=display.strip(),
                start_sample=start,
                end_sample=end,
                confidence=1.0 if probability is None else max(0.0, min(1.0, probability)),
                segment_id=f"segment-{index // max(1, words_per_segment)}",
                timing_unreliable=unreliable,
            )
        )

    for entry in entries:
        if not isinstance(entry, dict):
            raise WorkerError("whisper.cpp transcription entry is not an object")
        tokens = entry.get("tokens")
        if isinstance(tokens, list) and tokens:
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                display = str(token.get("text", ""))
                if display.strip().startswith("[_") or not display.strip():
                    continue  # special tokens such as [_BEG_]
                offsets = token.get("offsets") if isinstance(token.get("offsets"), dict) else {}
                add(
                    display,
                    offsets.get("from"),
                    offsets.get("to"),
                    token.get("p"),
                )
            continue
        offsets = entry.get("offsets") if isinstance(entry.get("offsets"), dict) else {}
        text = str(entry.get("text", ""))
        pieces = text.split()
        if not pieces:
            continue
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")
        if start_ms is None or end_ms is None or len(pieces) == 1:
            add(text.strip(), start_ms, end_ms, None)
            continue
        # A multi-word segment without per-token offsets: distribute timings
        # evenly and flag every word, because an even split is a guess.
        step = (float(end_ms) - float(start_ms)) / len(pieces)
        for position, piece in enumerate(pieces):
            add(piece, float(start_ms) + position * step, float(start_ms) + (position + 1) * step, None)
            if words:
                words[-1] = Word(
                    id=words[-1].id,
                    text=words[-1].text,
                    display=words[-1].display,
                    start_sample=words[-1].start_sample,
                    end_sample=words[-1].end_sample,
                    confidence=words[-1].confidence,
                    segment_id=words[-1].segment_id,
                    timing_unreliable=True,
                )

    detected = payload.get("result", {})
    if isinstance(detected, dict):
        language = str(detected.get("language") or language)

    end_of_words = words[-1].end_sample if words else 0
    return Transcript(
        session_id=session_id,
        revision=1,
        words=tuple(words),
        sample_rate=sample_rate,
        total_samples=max(total_samples, end_of_words),
        language=language,
        asr_model=model_id,
        asr_model_revision=model_revision,
        word_timings_experimental=True,
    )


@dataclass(slots=True)
class WhisperCppSettings:
    threads: int = 4
    #: ``-ml 1`` asks whisper.cpp for token-level segments, which is what makes
    #: word timings available at all.  Still experimental upstream.
    max_len: int = 1
    extra_args: tuple[str, ...] = ()
    timeout_s: float = 300.0


class WhisperCppWorker:
    """Runs whisper.cpp as a subprocess over private files, never a server.

    Inference talks to nothing but the filesystem paths it is handed; there is
    no externally reachable inference endpoint (plan section 3).
    """

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        *,
        binary: Path | None = None,
        settings: WhisperCppSettings | None = None,
    ) -> None:
        if spec.role is not ModelRole.ASR:
            raise ValueError(f"{spec.identifier} is not an ASR model")
        self.spec = spec
        self.model_path = model_path
        self.binary = find_binary(binary)
        self.settings = settings or WhisperCppSettings()
        self.stats = WorkerStats()
        self._warm = False

    @property
    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(
            model_id=self.spec.identifier,
            model_revision=self.spec.revision,
            quantization=self.spec.quantization,
            languages=tuple(self.spec.languages),
            streaming=False,  # chunked re-decoding, not true streaming
            word_timestamps=True,
            word_timings_experimental=True,
        )

    def ensure_ready(self) -> None:
        """Fail with an actionable message instead of degrading silently."""
        if self.binary is None:
            raise ModelNotInstalled(
                "whisper.cpp binary not found. Bundle it with the release or set "
                "DICTATION_WHISPER_BINARY. Looked for: " + ", ".join(BINARY_CANDIDATES)
            )
        if self.model_path is None or not self.model_path.exists():
            raise ModelNotInstalled(
                f"ASR model {self.spec.identifier} is not installed. Choose a candidate "
                "and run: dictation models fetch --role asr"
            )

    def warm_up(self) -> None:
        self.ensure_ready()
        self._warm = True

    def release(self) -> None:
        self._warm = False

    def transcribe(
        self,
        pcm: bytes,
        *,
        session_id: str,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        language: str = "en",
    ) -> Transcript:
        self.ensure_ready()
        assert self.binary is not None and self.model_path is not None
        audio_samples = len(pcm) // CANONICAL_SAMPLE_WIDTH
        began = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="dictation-asr-") as scratch:
            work = Path(scratch)
            wav_path = work / "audio.wav"
            write_wav(wav_path, pcm, sample_rate)
            out_base = work / "out"
            command = [
                str(self.binary),
                "-m",
                str(self.model_path),
                "-f",
                str(wav_path),
                "-t",
                str(self.settings.threads),
                "-l",
                language,
                "-ml",
                str(self.settings.max_len),
                "-oj",
                "-of",
                str(out_base),
                *self.settings.extra_args,
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=self.settings.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                self.stats.failures += 1
                raise AsrUnavailable("whisper.cpp timed out") from error
            if completed.returncode != 0:
                self.stats.failures += 1
                # stderr is not logged: it can echo recognised text.
                raise AsrUnavailable(f"whisper.cpp exited with {completed.returncode}")
            json_path = out_base.with_suffix(".json")
            if not json_path.exists():
                self.stats.failures += 1
                raise AsrUnavailable("whisper.cpp produced no JSON output")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        transcript = parse_whisper_json(
            payload,
            session_id=session_id,
            sample_rate=sample_rate,
            total_samples=audio_samples,
            model_id=self.spec.identifier,
            model_revision=self.spec.revision,
            language=language,
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
        """Not implemented for this backend.

        Partial hypotheses need incremental re-decoding of overlapping windows;
        doing that through a one-shot subprocess would miss the ~1 second
        partial-update target (plan section 4).  The coordinator falls back to
        recording without live partials, which is visible in the API
        capabilities rather than silently degraded.
        """
        raise AsrUnavailable("whisper.cpp subprocess backend does not stream partials")
        yield  # pragma: no cover - makes this a generator for type checkers
