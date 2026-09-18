"""Encrypted payload store (plan section 6).

Every payload - audio, transcript, detected spans, upload package - is written
as an authenticated ciphertext under a key derived for that session and kind.
The database row records the ciphertext digest, the kind and the expiry; the
plaintext exists only in memory, for as long as a caller holds it.

The store also draws a boundary the plan requires: the upload worker must not
have read access to the raw-session store (plan section 9).  That is why
:meth:`SessionStore.package_reader` exists and why it refuses any kind other
than an upload package - the worker is handed a capability, not the store.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dictation.errors import StoreError
from dictation.logging_ import events
from dictation.paths import DataPaths
from dictation.store.crypto import PayloadContext, decrypt, encrypt
from dictation.store.db import Database
from dictation.store.keys import KeyStore
from dictation.types import new_id


class ArtifactKind(StrEnum):
    """What a stored payload is.

    The raw kinds are working data with a hard expiry; only ``PACKAGE`` may
    leave the machine, and only through the upload worker.
    """

    RAW_AUDIO = "raw_audio"
    RAW_TRANSCRIPT = "raw_transcript"
    SENSITIVE_SPANS = "sensitive_spans"
    SPLICED_EXPORT = "spliced_export"
    PACKAGE = "package"


RAW_KINDS = frozenset(
    {ArtifactKind.RAW_AUDIO, ArtifactKind.RAW_TRANSCRIPT, ArtifactKind.SENSITIVE_SPANS}
)


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    artifact_id: str
    session_id: str
    kind: ArtifactKind
    path: Path
    sha256: str
    size_bytes: int
    created_at: float
    expires_at: float


class SessionStore:
    """Encrypted, expiring payload storage."""

    def __init__(self, paths: DataPaths, database: Database, key_store: KeyStore) -> None:
        self.paths = paths
        self.db = database
        self.keys = key_store

    # -- writing -------------------------------------------------------------

    def put(
        self,
        session_id: str,
        kind: ArtifactKind,
        payload: bytes,
        *,
        expires_at: float,
        now: float | None = None,
    ) -> StoredArtifact:
        """Encrypt and store one payload."""
        stamp = time.time() if now is None else now
        artifact_id = new_id("artifact")
        directory = (
            self.paths.queue_dir(session_id)
            if kind is ArtifactKind.PACKAGE
            else self.paths.session_dir(session_id)
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{artifact_id}.enc"

        key = self.keys.session_key(session_id, str(kind))
        envelope = encrypt(key, payload, PayloadContext(session_id=session_id, kind=str(kind)))
        digest = hashlib.sha256(envelope).hexdigest()

        temporary = path.with_suffix(".enc.tmp")
        temporary.write_bytes(envelope)
        temporary.replace(path)

        self.db.insert_artifact(
            artifact_id,
            session_id,
            kind=str(kind),
            path=str(path),
            sha256=digest,
            size_bytes=len(envelope),
            created_at=stamp,
            expires_at=expires_at,
        )
        events.emit(
            "store.artifact_written",
            session=session_id,
            kind=kind,
            bytes=len(envelope),
            expires_in_s=int(expires_at - stamp),
        )
        return StoredArtifact(
            artifact_id=artifact_id,
            session_id=session_id,
            kind=kind,
            path=path,
            sha256=digest,
            size_bytes=len(envelope),
            created_at=stamp,
            expires_at=expires_at,
        )

    # -- reading -------------------------------------------------------------

    def get(self, artifact_id: str) -> bytes:
        artifact = self.describe(artifact_id)
        envelope = artifact.path.read_bytes()
        if hashlib.sha256(envelope).hexdigest() != artifact.sha256:
            raise StoreError(f"artifact {artifact_id} digest does not match the recorded value")
        key = self.keys.session_key(artifact.session_id, str(artifact.kind))
        return decrypt(
            key, envelope, PayloadContext(session_id=artifact.session_id, kind=str(artifact.kind))
        )

    def describe(self, artifact_id: str) -> StoredArtifact:
        row = self.db.artifact(artifact_id)
        if row is None:
            raise StoreError(f"unknown artifact {artifact_id}")
        return StoredArtifact(
            artifact_id=row["artifact_id"],
            session_id=row["session_id"],
            kind=ArtifactKind(row["kind"]),
            path=Path(row["path"]),
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    def find(self, session_id: str, kind: ArtifactKind) -> StoredArtifact | None:
        rows = self.db.artifacts_of(session_id, str(kind))
        if not rows:
            return None
        return self.describe(rows[-1]["artifact_id"])

    # -- deletion ------------------------------------------------------------

    def delete(self, artifact_id: str) -> None:
        """Remove one payload: overwrite, unlink, forget the row."""
        try:
            artifact = self.describe(artifact_id)
        except StoreError:
            return
        _shred(artifact.path)
        self.db.delete_artifact(artifact_id)
        events.emit("store.artifact_deleted", session=artifact.session_id, kind=artifact.kind)

    def delete_payloads(self, session_id: str, *, destroy_keys: bool = True) -> int:
        """Remove every payload of a session, keeping its metadata rows.

        The queue needs this: a cancelled job must keep its ``deleted`` state
        so that a receipt arriving afterwards is recognised as late rather than
        as a job nobody has heard of (plan section 9).

        Overwriting and unlinking removes the copies this application made.
        Combined with forgetting the derived key it makes the data
        inaccessible; it is not a guarantee of physical erasure from SSD
        wear-levelling, swap or external backups (plan section 6).
        """
        rows = self.db.artifacts_of(session_id)
        for row in rows:
            _shred(Path(row["path"]))
            self.db.delete_artifact(row["artifact_id"])
        for directory in (self.paths.session_dir(session_id), self.paths.queue_dir(session_id)):
            if directory.exists():
                for leftover in directory.iterdir():
                    _shred(leftover)
                try:
                    directory.rmdir()
                except OSError:
                    pass
        if destroy_keys:
            self.keys.destroy_session_key(session_id)
        events.emit("store.payloads_deleted", session=session_id, artifacts=len(rows))
        return len(rows)

    def delete_session(self, session_id: str, *, destroy_keys: bool = True) -> int:
        """Remove a session's payloads *and* its metadata rows.

        Used by retention and by a full deletion request, once the job row no
        longer has to be remembered.
        """
        removed = self.delete_payloads(session_id, destroy_keys=destroy_keys)
        self.db.delete_session(session_id)
        events.emit("store.session_deleted", session=session_id, artifacts=removed)
        return removed

    # -- restricted capability ----------------------------------------------

    def package_reader(self) -> PackageReader:
        """Hand out a reader that can only see upload packages."""
        return PackageReader(self)


class PackageReader:
    """Read-only view limited to :attr:`ArtifactKind.PACKAGE` artifacts.

    The upload worker holds one of these instead of the store, so "the upload
    worker cannot read raw audio" is a property of the object graph rather than
    a rule someone has to remember.
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def read(self, artifact_id: str) -> bytes:
        artifact = self._store.describe(artifact_id)
        if artifact.kind is not ArtifactKind.PACKAGE:
            raise StoreError(
                f"upload worker may not read {artifact.kind} artifacts; "
                f"only {ArtifactKind.PACKAGE} is permitted"
            )
        return self._store.get(artifact_id)

    def find_package(self, session_id: str) -> StoredArtifact | None:
        return self._store.find(session_id, ArtifactKind.PACKAGE)

    def delete_package(self, artifact_id: str) -> None:
        artifact = self._store.describe(artifact_id)
        if artifact.kind is not ArtifactKind.PACKAGE:
            raise StoreError("upload worker may only delete packages")
        self._store.delete(artifact_id)


def _shred(path: Path) -> None:
    """Overwrite a file's bytes, then unlink it."""
    try:
        if not path.exists() or not path.is_file():
            return
        size = path.stat().st_size
        with path.open("r+b") as handle:
            handle.write(b"\x00" * size)
            handle.flush()
        path.unlink()
    except OSError:
        # Deletion of the row still proceeds; the payload is unreadable once
        # its key is gone, and retention will retry the unlink.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
