"""Upload package construction (plan section 9).

A package contains only what training needs:

    retained audio/text pairs, a random sample ID, sample rate, language,
    duration, quality metrics, model and policy versions, a consent reference
    and integrity checksums.

Everything else is stripped, and the strip is enforced rather than trusted:
:func:`assert_package_clean` refuses a manifest containing a session ID, a
tenant ID, source timestamps, a boundary manifest, removal spans, filenames,
user names, application titles or prompts.  Two of those deserve a note:

* **No tenant ID.**  Authentication determines customer ownership; a tenant
  field in the package would be a claim the server must not believe.
* **No removal map and no source timestamps.**  Uploading the positions of the
  cuts would explain what was deleted.  The boundary manifest and the source
  intervals stay on the device.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dictation.capture.audio_buffer import write_wav
from dictation.dataset.builder import BuildResult
from dictation.errors import DatasetRejected
from dictation.types import new_id
from dictation.version import ELIGIBILITY_VERSION, POLICY_VERSION

MANIFEST_NAME = "manifest.json"

ALLOWED_MANIFEST_KEYS = frozenset(
    {
        "eligibility_version",
        "sample_id",
        "sample_rate",
        "language",
        "duration_ms",
        "clip_count",
        "quality",
        "asr_model",
        "asr_model_revision",
        "privacy_model",
        "privacy_model_revision",
        "policy_version",
        "consent_version",
        "consent_reference",
        "clips",
        "checksums",
    }
)

ALLOWED_CLIP_KEYS = frozenset({"audio", "text", "duration_ms", "word_count", "sha256"})

#: Keys that must never appear anywhere in a package manifest, at any depth.
FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "tenant",
        "tenant_id",
        "customer_id",
        "account_id",
        "session_id",
        "session",
        "user",
        "username",
        "user_name",
        "hostname",
        "device_name",
        "machine",
        "path",
        "filename",
        "file_path",
        "app",
        "application",
        "window_title",
        "foreground_app",
        "prompt",
        "system_prompt",
        "spans",
        "removed",
        "removal_map",
        "boundary_manifest",
        "source_start",
        "source_end",
        "source_interval",
        "original_text",
        "full_transcript",
        "latitude",
        "longitude",
        "location",
        "ip",
        "ip_address",
        "timezone",
        "created_at",
    }
)


@dataclass(frozen=True, slots=True)
class UploadPackage:
    """An eligible, self-contained training sample."""

    sample_id: str
    manifest: dict[str, Any]
    files: tuple[tuple[str, bytes], ...]
    #: Digest of the serialised archive, used as the idempotency key.
    archive_sha256: str = ""
    archive_bytes: bytes = b""

    @property
    def size_bytes(self) -> int:
        return len(self.archive_bytes)

    @property
    def duration_ms(self) -> int:
        return int(self.manifest.get("duration_ms", 0))

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.archive_bytes)
        return path


def assert_package_clean(manifest: dict[str, Any]) -> None:
    """Refuse a manifest containing anything outside the allowlist."""
    unknown = set(manifest) - ALLOWED_MANIFEST_KEYS
    if unknown:
        raise DatasetRejected("package_field_not_allowed", ",".join(sorted(unknown)))
    _walk_forbidden(manifest, path="manifest")
    for index, clip in enumerate(manifest.get("clips", [])):
        extra = set(clip) - ALLOWED_CLIP_KEYS
        if extra:
            raise DatasetRejected("package_clip_field_not_allowed", f"clip {index}: {sorted(extra)}")


def _walk_forbidden(node: Any, *, path: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_MANIFEST_KEYS:
                raise DatasetRejected("package_forbidden_field", f"{path}.{key}")
            _walk_forbidden(value, path=f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk_forbidden(value, path=f"{path}[{index}]")


def build_package(
    result: BuildResult,
    *,
    consent_version: str,
    consent_reference: str,
    eligibility_version: int = ELIGIBILITY_VERSION,
    sample_id: str | None = None,
) -> UploadPackage:
    """Serialise an eligible build result into an upload package.

    Raises :class:`~dictation.errors.DatasetRejected` for a rejected result, so
    that a rejected session cannot be packaged by mistake.
    """
    if not result.eligible:
        raise DatasetRejected("not_eligible", result.reason)
    if not result.clips:
        raise DatasetRejected("no_retained_clips")

    identifier = sample_id or new_id("sample")
    files: list[tuple[str, bytes]] = []
    clips: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}

    for clip, pcm in zip(result.clips, result.clip_audio, strict=True):
        audio_name = f"clip-{clip.clip_index:03d}.wav"
        text_name = f"clip-{clip.clip_index:03d}.txt"
        buffer = io.BytesIO()
        with_wav = _wav_bytes(pcm, clip.sample_rate, buffer)
        text_bytes = clip.text.encode("utf-8")
        files.append((audio_name, with_wav))
        files.append((text_name, text_bytes))
        checksums[audio_name] = hashlib.sha256(with_wav).hexdigest()
        checksums[text_name] = hashlib.sha256(text_bytes).hexdigest()
        clips.append(
            {
                "audio": audio_name,
                "text": text_name,
                "duration_ms": clip.duration_ms,
                "word_count": len(clip.words),
                "sha256": checksums[audio_name],
            }
        )

    manifest: dict[str, Any] = {
        "eligibility_version": eligibility_version,
        "sample_id": identifier,
        "sample_rate": result.clips[0].sample_rate,
        "language": result.language,
        "duration_ms": result.total_retained_ms,
        "clip_count": len(result.clips),
        "quality": result.metrics.as_dict(),
        "asr_model": result.asr_model,
        "privacy_model": result.privacy_model,
        "policy_version": result.policy_version or POLICY_VERSION,
        "consent_version": consent_version,
        "consent_reference": consent_reference,
        "clips": clips,
        "checksums": checksums,
    }
    assert_package_clean(manifest)

    archive = _zip_bytes(manifest, files)
    return UploadPackage(
        sample_id=identifier,
        manifest=manifest,
        files=tuple(files),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        archive_bytes=archive,
    )


def _wav_bytes(pcm: bytes, sample_rate: int, buffer: io.BytesIO) -> bytes:
    import wave

    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def _zip_bytes(manifest: dict[str, Any], files: list[tuple[str, bytes]]) -> bytes:
    """Deterministic archive: fixed timestamps and sorted entries.

    Determinism matters for the idempotency key - the same sample must produce
    the same digest on a retry, so a retry cannot create a second server-side
    example.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        entries = [(MANIFEST_NAME, json.dumps(manifest, sort_keys=True).encode("utf-8")), *sorted(files)]
        for name, payload in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def read_package(archive_bytes: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read a package back.  Used by the server and by tests.

    Every malformed-input failure becomes :class:`DatasetRejected`, so the
    server refuses such an upload rather than raising out of admission: client
    checks are not the boundary against a malformed package (plan section 9).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            if MANIFEST_NAME not in names:
                raise DatasetRejected("package_missing_manifest")
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            payloads = {name: archive.read(name) for name in names if name != MANIFEST_NAME}
    except zipfile.BadZipFile:
        raise DatasetRejected("package_not_an_archive") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise DatasetRejected("package_manifest_unreadable") from None
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, DatasetRejected):
            raise
        raise DatasetRejected("package_unreadable") from None
    return manifest, payloads


def verify_package(archive_bytes: bytes) -> dict[str, Any]:
    """Validate structure, allowlist and checksums; return the manifest."""
    manifest, payloads = read_package(archive_bytes)
    if not isinstance(manifest, dict):
        raise DatasetRejected("package_manifest_not_object")
    assert_package_clean(manifest)
    checksums = manifest.get("checksums") or {}
    if set(checksums) != set(payloads):
        raise DatasetRejected("package_checksum_coverage_mismatch")
    for name, expected in checksums.items():
        actual = hashlib.sha256(payloads[name]).hexdigest()
        if actual != expected:
            raise DatasetRejected("package_checksum_mismatch", name)
    for clip in manifest.get("clips", []):
        for key in ("audio", "text"):
            if clip.get(key) not in payloads:
                raise DatasetRejected("package_missing_file", str(clip.get(key)))
    return manifest


@dataclass
class PackageWriter:
    """Writes packages into the queue directory as encrypted payloads.

    The plaintext archive never touches the filesystem: the caller hands the
    bytes to the encrypted store.  This helper exists for the offline export
    path used in evaluation, which writes to an explicitly chosen directory.
    """

    directory: Path
    written: list[Path] = field(default_factory=list)

    def write_plaintext_export(self, package: UploadPackage) -> Path:
        path = self.directory / f"{package.sample_id}.zip"
        package.write(path)
        self.written.append(path)
        return path

    def write_clips(self, result: BuildResult, prefix: str = "clip") -> list[Path]:
        paths: list[Path] = []
        for clip, pcm in zip(result.clips, result.clip_audio, strict=True):
            audio_path = self.directory / f"{prefix}-{clip.clip_index:03d}.wav"
            write_wav(audio_path, pcm, clip.sample_rate)
            text_path = audio_path.with_suffix(".txt")
            text_path.write_text(clip.text, encoding="utf-8")
            paths.extend([audio_path, text_path])
        self.written.extend(paths)
        return paths
