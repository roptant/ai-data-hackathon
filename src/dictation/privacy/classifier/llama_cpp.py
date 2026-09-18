"""llama.cpp privacy-classifier backend (plan sections 3 and 4).

Like the ASR backend, the artifact is an open decision: the four-bit
Qwen3-4B-Instruct-2507 candidate has no URL or checksum in the registry, so this
worker refuses to run until one is recorded and fetched.

Two properties are structural rather than prompt-level:

* the subprocess gets the prompt and the grammar file and nothing else - no
  tools, no network, no filesystem authority beyond its own input and output,
* grammar-constrained decoding shapes the answer, and the answer is then
  validated against the frozen transcript by :mod:`dictation.privacy.schema`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dictation.errors import ModelNotInstalled, PrivacyWorkerUnavailable
from dictation.models.registry import ModelRole, ModelSpec
from dictation.privacy.classifier.base import (
    SYSTEM_PROMPT,
    ClassifierCapabilities,
    ClassifierWindow,
)
from dictation.privacy.schema import GBNF_GRAMMAR

BINARY_CANDIDATES = ("llama-cli", "llama-cpp", "main")


def find_binary(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    env = os.environ.get("DICTATION_LLAMA_BINARY")
    if env:
        candidate = Path(env)
        return candidate if candidate.exists() else None
    for name in BINARY_CANDIDATES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


@dataclass(slots=True)
class LlamaCppSettings:
    threads: int = 4
    context_tokens: int = 4096
    #: Deterministic decoding: the same transcript must yield the same spans,
    #: otherwise a rejected session could pass on a retry.
    temperature: float = 0.0
    seed: int = 1
    max_predict: int = 512
    extra_args: tuple[str, ...] = ()


class LlamaCppPrivacyWorker:
    """Runs one classification per window through a llama.cpp subprocess."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        *,
        binary: Path | None = None,
        settings: LlamaCppSettings | None = None,
    ) -> None:
        if spec.role is not ModelRole.PRIVACY:
            raise ValueError(f"{spec.identifier} is not a privacy model")
        self.spec = spec
        self.model_path = model_path
        self.binary = find_binary(binary)
        self.settings = settings or LlamaCppSettings()
        self._grammar_path: Path | None = None
        self._scratch: tempfile.TemporaryDirectory[str] | None = None

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(
            model_id=self.spec.identifier,
            model_revision=self.spec.revision,
            quantization=self.spec.quantization,
            context_tokens=self.settings.context_tokens,
            grammar_constrained=True,
            measured_span_recall=0.0,  # unmeasured until the benchmark is run
        )

    def ensure_ready(self) -> None:
        if self.binary is None:
            raise ModelNotInstalled(
                "llama.cpp binary not found. Bundle it with the release or set "
                "DICTATION_LLAMA_BINARY. Looked for: " + ", ".join(BINARY_CANDIDATES)
            )
        if self.model_path is None or not self.model_path.exists():
            raise ModelNotInstalled(
                f"privacy model {self.spec.identifier} is not installed. Choose a "
                "candidate and run: dictation models fetch --role privacy"
            )

    def warm_up(self) -> None:
        self.ensure_ready()
        if self._grammar_path is None:
            self._scratch = tempfile.TemporaryDirectory(prefix="dictation-privacy-")
            path = Path(self._scratch.name) / "spans.gbnf"
            path.write_text(GBNF_GRAMMAR, encoding="utf-8")
            self._grammar_path = path

    def release(self) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
        self._grammar_path = None

    def classify(self, window: ClassifierWindow, *, timeout_s: float) -> Any:
        self.warm_up()
        assert self.binary is not None and self.model_path is not None
        assert self._grammar_path is not None
        prompt = f"{SYSTEM_PROMPT}\n\n{window.prompt()}"
        command = [
            str(self.binary),
            "-m",
            str(self.model_path),
            "-c",
            str(self.settings.context_tokens),
            "-t",
            str(self.settings.threads),
            "--temp",
            str(self.settings.temperature),
            "--seed",
            str(self.settings.seed),
            "-n",
            str(self.settings.max_predict),
            "--grammar-file",
            str(self._grammar_path),
            "--no-display-prompt",
            "-no-cnv",
            "-p",
            prompt,
            *self.settings.extra_args,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                # No inherited environment beyond what the binary needs; the
                # worker has no credentials and no upload endpoint.
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired as error:
            raise PrivacyWorkerUnavailable(f"privacy model timed out after {timeout_s}s") from error
        if completed.returncode != 0:
            # stderr is not logged: it echoes the prompt, which is transcript text.
            raise PrivacyWorkerUnavailable(f"privacy model exited with {completed.returncode}")
        return completed.stdout.decode("utf-8", errors="replace")
