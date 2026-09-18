"""Model download and verification (plan sections 3 and 4).

Model downloads are a separate, visible network operation; dictation itself
works offline once installed.  A fetch:

* refuses an unresolved candidate (no URL, digest, size, licence or revision),
* refuses anything but HTTPS and an allow-listed weight-file extension,
* streams to a ``.part`` file, verifies size and SHA-256, and only then renames
  into place, so an interrupted download can never be loaded,
* records a manifest entry, re-verified before the artifact is used,
* never extracts archives and never executes anything it downloaded.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dictation.errors import ModelError, ModelVerificationError
from dictation.logging_ import events
from dictation.models.manifest import InstalledModel, ModelManifest
from dictation.models.registry import ModelRegistry, ModelSpec

CHUNK_BYTES = 1024 * 1024
USER_AGENT = "local-dictation-model-fetcher/1"

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class FetchResult:
    spec: ModelSpec
    path: Path
    bytes_written: int
    verified: bool
    already_present: bool = False


def sha256_file(path: Path, *, chunk: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch_model(
    registry: ModelRegistry,
    spec: ModelSpec,
    *,
    manifest: ModelManifest | None = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
    opener: Callable[[urllib.request.Request], object] | None = None,
) -> FetchResult:
    """Download and verify one artifact.

    ``opener`` exists for tests: it takes a :class:`urllib.request.Request` and
    returns a context-manager response.  Production uses ``urlopen``.
    """
    spec.validate_for_fetch()
    destination = registry.path_of(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest or ModelManifest.load(registry.directory)

    if destination.exists() and not force:
        digest = sha256_file(destination)
        if digest != spec.sha256.lower():
            raise ModelVerificationError(
                f"{spec.identifier} already present at {destination} but its digest does not "
                f"match the pinned value; delete it or re-fetch with force"
            )
        manifest.record(_installed(spec, destination, digest))
        events.emit("model.already_installed", model=spec.identifier, role=spec.role)
        return FetchResult(spec, destination, destination.stat().st_size, True, already_present=True)

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(spec.source_url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    written = 0
    events.emit("model.fetch_started", model=spec.identifier, role=spec.role, bytes=spec.size_bytes)
    try:
        open_url = opener or urllib.request.urlopen
        with open_url(request) as response, partial.open("wb") as sink:  # type: ignore[union-attr]
            while True:
                block = response.read(CHUNK_BYTES)  # type: ignore[union-attr]
                if not block:
                    break
                written += len(block)
                if written > spec.size_bytes:
                    raise ModelVerificationError(
                        f"{spec.identifier}: download exceeds the pinned size of {spec.size_bytes} bytes"
                    )
                digest.update(block)
                sink.write(block)
                if progress is not None:
                    progress(written, spec.size_bytes)
    except urllib.error.URLError as error:
        partial.unlink(missing_ok=True)
        raise ModelError(f"{spec.identifier}: download failed ({error.reason})") from error
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    if written != spec.size_bytes:
        partial.unlink(missing_ok=True)
        raise ModelVerificationError(
            f"{spec.identifier}: expected {spec.size_bytes} bytes, received {written}"
        )
    actual = digest.hexdigest()
    if actual != spec.sha256.lower():
        partial.unlink(missing_ok=True)
        events.error("model.digest_mismatch", model=spec.identifier)
        raise ModelVerificationError(
            f"{spec.identifier}: SHA-256 mismatch; refusing to install the artifact"
        )

    partial.replace(destination)
    manifest.record(_installed(spec, destination, actual))
    events.emit("model.installed", model=spec.identifier, role=spec.role, bytes=written)
    return FetchResult(spec, destination, written, True)


def verify_installed(registry: ModelRegistry, spec: ModelSpec) -> bool:
    """Re-verify an installed artifact against its pinned digest.

    Called at startup: a model file that changed on disk - supply-chain
    tampering, a partially restored backup - must not be loaded (plan 11).
    """
    path = registry.path_of(spec)
    if not path.exists():
        return False
    if not spec.sha256:
        raise ModelVerificationError(f"{spec.identifier} has no pinned digest to verify against")
    matches = sha256_file(path) == spec.sha256.lower()
    if not matches:
        events.error("model.verification_failed", model=spec.identifier)
    return matches


def _installed(spec: ModelSpec, path: Path, digest: str) -> InstalledModel:
    return InstalledModel(
        identifier=spec.identifier,
        role=str(spec.role),
        filename=path.name,
        sha256=digest,
        size_bytes=path.stat().st_size,
        revision=spec.revision,
        license_id=spec.license_id,
        source_url=spec.source_url,
    )
