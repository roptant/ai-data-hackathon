"""Key management through the OS credential store (plan section 6).

One master key lives in the platform credential store - Windows Credential
Manager, macOS Keychain, or a Secret Service implementation on Linux.  Per
session and per payload kind, a data key is derived with HKDF, so no per-session
secret has to be stored anywhere and a session's payloads can be made
inaccessible by forgetting a small amount of salt material.

If secure key storage is unavailable, this module raises
:class:`~dictation.errors.SecureStorageUnavailable`.  There is no plaintext
fallback: the caller disables persistent contribution storage instead.
:class:`EphemeralKeyStore` exists for tests and for the "contribution off" case
where audio never has to outlive the process.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from dictation.errors import SecureStorageUnavailable
from dictation.store.crypto import KEY_BYTES

SERVICE_NAME = "LocalDictation"
MASTER_KEY_ACCOUNT = "payload-master-key"
LOG_SALT_ACCOUNT = "log-digest-salt"


def _derive(master: bytes, *, salt: bytes, info: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        info=info.encode("utf-8"),
    ).derive(master)


@runtime_checkable
class KeyStore(Protocol):
    """Source of per-session payload keys."""

    @property
    def persistent(self) -> bool:
        """True when keys survive a restart, which persistent payloads need."""

    def session_key(self, session_id: str, kind: str) -> bytes: ...

    def log_salt(self) -> bytes: ...

    def destroy_session_key(self, session_id: str) -> None:
        """Forget the material for a session, making its payloads unreadable."""


@dataclass
class EphemeralKeyStore:
    """In-memory master key.  Nothing it protects survives a restart."""

    master: bytes = field(default_factory=lambda: os.urandom(KEY_BYTES))
    _salt: bytes = field(default_factory=lambda: os.urandom(32))
    _revoked: set[str] = field(default_factory=set)

    @property
    def persistent(self) -> bool:
        return False

    def session_key(self, session_id: str, kind: str) -> bytes:
        if session_id in self._revoked:
            raise SecureStorageUnavailable(f"key material for {session_id} was destroyed")
        return _derive(self.master, salt=session_id.encode("utf-8"), info=kind)

    def log_salt(self) -> bytes:
        return self._salt

    def destroy_session_key(self, session_id: str) -> None:
        self._revoked.add(session_id)


class KeyringKeyStore:
    """Master key in the OS credential store.

    The constructor verifies that the backend actually works by writing and
    reading back a probe value.  A backend that silently drops secrets would
    otherwise surface much later, as undecryptable payloads.
    """

    def __init__(self, service: str = SERVICE_NAME) -> None:
        self.service = service
        self._keyring = self._load_backend()
        self._master: bytes | None = None
        self._revoked: set[str] = set()

    @staticmethod
    def _load_backend() -> object:
        try:
            import keyring
            from keyring.backends import fail as fail_backend
        except ImportError as error:  # pragma: no cover - dependency present in practice
            raise SecureStorageUnavailable(
                "the keyring package is required for persistent encrypted storage"
            ) from error
        backend = keyring.get_keyring()
        if isinstance(backend, fail_backend.Keyring):
            raise SecureStorageUnavailable(
                "no OS credential store is available; persistent contribution "
                "storage stays disabled rather than falling back to plaintext"
            )
        return keyring

    @classmethod
    def available(cls) -> bool:
        try:
            cls._load_backend()
        except SecureStorageUnavailable:
            return False
        return True

    def _get(self, account: str) -> str | None:
        try:
            return self._keyring.get_password(self.service, account)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001 - backend specific failures
            raise SecureStorageUnavailable(f"credential store read failed: {type(error).__name__}") from error

    def _set(self, account: str, value: str) -> None:
        try:
            self._keyring.set_password(self.service, account, value)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001
            raise SecureStorageUnavailable(f"credential store write failed: {type(error).__name__}") from error

    def _get_or_create(self, account: str, length: int) -> bytes:
        existing = self._get(account)
        if existing:
            try:
                material = base64.b64decode(existing, validate=True)
            except (ValueError, TypeError) as error:
                raise SecureStorageUnavailable(
                    f"stored secret {account} is corrupt; refusing to continue with a new key "
                    f"because existing payloads would become undecryptable"
                ) from error
            if len(material) == length:
                return material
            raise SecureStorageUnavailable(f"stored secret {account} has the wrong length")
        material = os.urandom(length)
        self._set(account, base64.b64encode(material).decode("ascii"))
        verified = self._get(account)
        if verified is None or base64.b64decode(verified) != material:
            raise SecureStorageUnavailable("credential store did not retain the secret")
        return material

    @property
    def persistent(self) -> bool:
        return True

    def master_key(self) -> bytes:
        if self._master is None:
            self._master = self._get_or_create(MASTER_KEY_ACCOUNT, KEY_BYTES)
        return self._master

    def session_key(self, session_id: str, kind: str) -> bytes:
        if session_id in self._revoked:
            raise SecureStorageUnavailable(f"key material for {session_id} was destroyed")
        return _derive(self.master_key(), salt=session_id.encode("utf-8"), info=kind)

    def log_salt(self) -> bytes:
        return self._get_or_create(LOG_SALT_ACCOUNT, 32)

    def destroy_session_key(self, session_id: str) -> None:
        """Mark a session's keys unusable for the rest of this process.

        Because keys are derived, forgetting is process-local; durable deletion
        is the store removing the ciphertext.  Both happen, and neither is
        advertised as physical erasure.
        """
        self._revoked.add(session_id)

    def destroy_master_key(self) -> None:
        """Delete the master key.  Every existing payload becomes unreadable.

        Used by "delete my training data" as a belt-and-braces step after the
        ciphertexts have been removed.
        """
        try:
            self._keyring.delete_password(self.service, MASTER_KEY_ACCOUNT)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - absence is success here
            pass
        self._master = None


def open_key_store(*, allow_ephemeral: bool = False) -> KeyStore:
    """Return the persistent key store, or an ephemeral one if permitted.

    Callers that need to persist payloads must not pass ``allow_ephemeral``:
    silently writing payloads under a key that vanishes at exit would lose data
    and mislead the user about what was retained.
    """
    try:
        return KeyringKeyStore()
    except SecureStorageUnavailable:
        if allow_ephemeral:
            return EphemeralKeyStore()
        raise
