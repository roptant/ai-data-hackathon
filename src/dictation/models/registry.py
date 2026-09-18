"""Model registry: exactly two learned roles, both deliberately unresolved.

Plan sections 3 and 4 set the rules this module enforces:

* there are two learned model roles and no third one appears silently,
* dependencies and model revisions are pinned,
* checksums and redistribution terms are verified before use,
* downloaded model code is never executed.

The shipped candidates carry **no URL and no checksum**.  That is the open
product decision: whisper base/small for recognition and a four-bit
Qwen3-4B-Instruct-2507 for the privacy role are *candidates to benchmark*, not
choices.  ``dictation models resolve`` records a URL, digest, size and licence
for a candidate; until then every worker that needs it refuses to run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from dictation.errors import ModelNotConfigured, ModelNotInstalled, ModelVerificationError

#: File types we are willing to load.  Both are inert weight containers read by
#: the runtime; nothing here is code, and nothing downloaded is ever executed.
ALLOWED_SUFFIXES = frozenset({".gguf", ".bin"})


class ModelRole(StrEnum):
    ASR = "asr"
    PRIVACY = "privacy"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A pinned model artifact, or an unresolved candidate for one."""

    identifier: str
    role: ModelRole
    display_name: str = ""
    #: HTTPS URL of the exact artifact.  Empty means unresolved.
    source_url: str = ""
    #: Lowercase hex SHA-256 of the artifact.  Empty means unresolved.
    sha256: str = ""
    size_bytes: int = 0
    #: Upstream revision or commit the artifact was produced from.
    revision: str = ""
    quantization: str = ""
    #: Licence of the *original* model and of the converted artifact, which can
    #: differ.  Both must be recorded before distribution (plan section 4).
    license_id: str = ""
    license_url: str = ""
    #: Who produced the quantized artifact, if not the original publisher.
    conversion_provenance: str = ""
    languages: tuple[str, ...] = ("en",)
    context_tokens: int = 0
    #: Measured peak resident memory in MiB, or 0 when unmeasured.  Derived from
    #: benchmarking, never from weight-file size (plan section 4).
    measured_peak_memory_mib: int = 0
    notes: str = ""
    filename: str = ""

    @property
    def local_filename(self) -> str:
        if self.filename:
            return self.filename
        if self.source_url:
            tail = self.source_url.rstrip("/").rsplit("/", 1)[-1]
            if tail:
                return tail
        return f"{self.identifier}.gguf"

    def unresolved_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.source_url:
            missing.append("source_url")
        if not self.sha256:
            missing.append("sha256")
        if not self.size_bytes:
            missing.append("size_bytes")
        if not self.license_id:
            missing.append("license_id")
        if not self.revision:
            missing.append("revision")
        return tuple(missing)

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved_fields()

    def validate_for_fetch(self) -> None:
        missing = self.unresolved_fields()
        if missing:
            raise ModelNotConfigured(
                f"{self.identifier} is an unresolved candidate; missing "
                f"{', '.join(missing)}. Record them with: dictation models resolve "
                f"{self.identifier} --url ... --sha256 ... --size ... --license ... --revision ..."
            )
        if not self.source_url.lower().startswith("https://"):
            raise ModelVerificationError(f"{self.identifier}: model source must be HTTPS")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ModelVerificationError(f"{self.identifier}: sha256 must be 64 hex characters")
        suffix = Path(self.local_filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ModelVerificationError(
                f"{self.identifier}: refusing {suffix or 'extension-less'} artifact; "
                f"allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
            )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["role"] = str(self.role)
        data["languages"] = list(self.languages)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelSpec:
        payload = dict(data)
        payload["role"] = ModelRole(str(payload.get("role", "asr")))
        languages = payload.get("languages") or ("en",)
        payload["languages"] = tuple(str(item) for item in languages)  # type: ignore[arg-type]
        known = {f for f in ModelSpec.__slots__}
        return cls(**{k: v for k, v in payload.items() if k in known})  # type: ignore[arg-type]


# -- candidates from the plan, intentionally unresolved -----------------------

CANDIDATES: tuple[ModelSpec, ...] = (
    ModelSpec(
        identifier="whisper-base-q5_1",
        role=ModelRole.ASR,
        display_name="Whisper base (quantized) - candidate",
        quantization="q5_1",
        languages=("en",),
        notes=(
            "Plan section 4: start ASR evaluation with quantized Whisper base and "
            "small; choose the smallest model that meets measured recognition and "
            "latency requirements. Word timestamps are experimental upstream."
        ),
        filename="whisper-base-q5_1.bin",
    ),
    ModelSpec(
        identifier="whisper-small-q5_1",
        role=ModelRole.ASR,
        display_name="Whisper small (quantized) - candidate",
        quantization="q5_1",
        languages=("en",),
        notes="Larger ASR candidate; benchmark against base on the 8 GB baseline.",
        filename="whisper-small-q5_1.bin",
    ),
    ModelSpec(
        identifier="qwen3-4b-instruct-2507-q4",
        role=ModelRole.PRIVACY,
        display_name="Qwen3-4B-Instruct-2507 four-bit - candidate",
        quantization="q4",
        languages=("en",),
        context_tokens=8192,
        notes=(
            "Plan section 4: a candidate, not a claim of adequate privacy recall "
            "or acceptable speed on every 8 GB machine. Validate the original "
            "licence, the conversion, quantization quality and artifact provenance "
            "before recording a URL here."
        ),
        filename="qwen3-4b-instruct-2507-q4.gguf",
    ),
)


@dataclass
class ModelRegistry:
    """Persisted model choices and specs.

    The registry never downloads anything itself; see :mod:`dictation.models.fetch`.
    """

    directory: Path
    specs: dict[str, ModelSpec] = field(default_factory=dict)
    choices: dict[ModelRole, str] = field(default_factory=dict)

    @property
    def registry_file(self) -> Path:
        return self.directory / "registry.json"

    @classmethod
    def load(cls, directory: Path) -> ModelRegistry:
        registry = cls(directory=Path(directory))
        registry.specs = {spec.identifier: spec for spec in CANDIDATES}
        try:
            raw = json.loads(registry.registry_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return registry
        for entry in raw.get("specs", []):
            try:
                spec = ModelSpec.from_dict(entry)
            except (TypeError, ValueError):
                continue
            registry.specs[spec.identifier] = spec
        for role_name, identifier in (raw.get("choices") or {}).items():
            try:
                registry.choices[ModelRole(role_name)] = str(identifier)
            except ValueError:
                continue
        return registry

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "specs": [spec.to_dict() for spec in self.specs.values()],
            "choices": {str(role): identifier for role, identifier in self.choices.items()},
        }
        temporary = self.registry_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.registry_file)

    # -- queries -------------------------------------------------------------

    def for_role(self, role: ModelRole) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self.specs.values() if spec.role is role)

    def get(self, identifier: str) -> ModelSpec:
        try:
            return self.specs[identifier]
        except KeyError:
            raise ModelNotConfigured(f"unknown model {identifier!r}") from None

    def chosen(self, role: ModelRole) -> ModelSpec:
        """The spec selected for a role, or a clear refusal."""
        identifier = self.choices.get(role)
        if not identifier:
            available = ", ".join(spec.identifier for spec in self.for_role(role)) or "none"
            raise ModelNotConfigured(
                f"no model chosen for the {role} role. Candidates: {available}. "
                f"Choose with: dictation models choose --role {role} <identifier>"
            )
        return self.get(identifier)

    def path_of(self, spec: ModelSpec) -> Path:
        return self.directory / spec.role / spec.local_filename

    def installed_path(self, spec: ModelSpec) -> Path:
        path = self.path_of(spec)
        if not path.exists():
            raise ModelNotInstalled(
                f"{spec.identifier} is not installed at {path}. "
                f"Run: dictation models fetch --role {spec.role}"
            )
        return path

    def is_installed(self, spec: ModelSpec) -> bool:
        return self.path_of(spec).exists()

    # -- mutation ------------------------------------------------------------

    def upsert(self, spec: ModelSpec) -> ModelSpec:
        self.specs[spec.identifier] = spec
        self.save()
        return spec

    def resolve(self, identifier: str, **updates: object) -> ModelSpec:
        """Record the URL, digest, size, licence and revision for a candidate."""
        spec = replace(self.get(identifier), **updates)  # type: ignore[arg-type]
        spec.validate_for_fetch()
        return self.upsert(spec)

    def choose(self, role: ModelRole, identifier: str) -> ModelSpec:
        spec = self.get(identifier)
        if spec.role is not role:
            raise ModelNotConfigured(f"{identifier} is a {spec.role} model, not {role}")
        self.choices[role] = identifier
        self.save()
        return spec

    def unresolved(self) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self.specs.values() if not spec.is_resolved)


def describe(specs: Iterable[ModelSpec]) -> str:
    """Human-readable listing used by the CLI."""
    lines: list[str] = []
    for spec in specs:
        status = "resolved" if spec.is_resolved else f"unresolved ({', '.join(spec.unresolved_fields())})"
        lines.append(f"  {spec.identifier:<34} {spec.role:<8} {status}")
        if spec.notes:
            lines.append(f"      {spec.notes}")
    return "\n".join(lines)
