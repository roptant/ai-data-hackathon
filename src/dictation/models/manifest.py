"""Installed-model manifest (plan sections 3, 4 and 10).

Records what is installed, from where, under which licence and with which
digest.  Two things depend on it: startup verification of the artifacts, and
the lineage graph that has to name the exact base model a personalized artifact
was derived from.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InstalledModel:
    identifier: str
    role: str
    filename: str
    sha256: str
    size_bytes: int
    revision: str = ""
    license_id: str = ""
    source_url: str = ""
    installed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> InstalledModel:
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]


@dataclass
class ModelManifest:
    directory: Path
    entries: dict[str, InstalledModel] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.directory / "installed.json"

    @classmethod
    def load(cls, directory: Path) -> ModelManifest:
        manifest = cls(directory=Path(directory))
        try:
            raw = json.loads(manifest.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return manifest
        for entry in raw.get("models", []):
            try:
                installed = InstalledModel.from_dict(entry)
            except (TypeError, ValueError):
                continue
            manifest.entries[installed.identifier] = installed
        return manifest

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"models": [entry.to_dict() for entry in self.entries.values()]}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def record(self, installed: InstalledModel) -> InstalledModel:
        self.entries[installed.identifier] = installed
        self.save()
        return installed

    def forget(self, identifier: str) -> None:
        self.entries.pop(identifier, None)
        self.save()

    def get(self, identifier: str) -> InstalledModel | None:
        return self.entries.get(identifier)
