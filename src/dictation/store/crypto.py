"""Authenticated encryption for local payloads (plan section 6).

AES-256-GCM from ``cryptography`` - a vetted implementation, not a hand-rolled
construction.  Two details matter beyond "it is encrypted":

* **Context binding.**  The session ID, payload kind and format version are the
  additional authenticated data, so a ciphertext cannot be moved between
  sessions or relabelled as a different kind of payload without failing to
  decrypt.
* **Fresh nonces.**  Each payload gets a random 96-bit nonce.  Keys are
  per-session and derived, so nonce reuse across payloads of one session is the
  only risk, and randomness at 96 bits covers the volumes involved here.

Key destruction (:func:`dictation.store.keys.destroy_session_key`) makes a
ciphertext inaccessible in practice.  It is not a claim of physical erasure
from an SSD, a swap file or a third-party backup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from dictation.errors import DecryptionError

MAGIC = b"DCT1"
FORMAT_VERSION = 1
NONCE_BYTES = 12
KEY_BYTES = 32
HEADER_BYTES = len(MAGIC) + 1 + NONCE_BYTES


@dataclass(frozen=True, slots=True)
class PayloadContext:
    """Identifies what a ciphertext is, and binds it to that identity."""

    session_id: str
    kind: str
    version: int = FORMAT_VERSION

    def aad(self) -> bytes:
        return f"{self.version}|{self.session_id}|{self.kind}".encode("utf-8")


def encrypt(key: bytes, plaintext: bytes, context: PayloadContext) -> bytes:
    """Return ``MAGIC || version || nonce || ciphertext``."""
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes")
    nonce = os.urandom(NONCE_BYTES)
    sealed = AESGCM(key).encrypt(nonce, plaintext, context.aad())
    return MAGIC + bytes([context.version]) + nonce + sealed


def decrypt(key: bytes, envelope: bytes, context: PayloadContext) -> bytes:
    """Authenticate and decrypt.  Any mismatch raises ``DecryptionError``."""
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes")
    if len(envelope) <= HEADER_BYTES:
        raise DecryptionError("payload is too short to be an envelope")
    if envelope[: len(MAGIC)] != MAGIC:
        raise DecryptionError("payload is not a dictation envelope")
    version = envelope[len(MAGIC)]
    if version != context.version:
        raise DecryptionError(f"payload format version {version} != expected {context.version}")
    nonce = envelope[len(MAGIC) + 1 : HEADER_BYTES]
    try:
        return AESGCM(key).decrypt(nonce, envelope[HEADER_BYTES:], context.aad())
    except InvalidTag as error:
        raise DecryptionError(
            "payload failed authentication: wrong key, wrong context or tampering"
        ) from error


def overwrite(buffer: bytearray) -> None:
    """Zero a mutable buffer holding plaintext or key material.

    Best effort within this process.  Python strings and immutable bytes cannot
    be wiped, which is why key material is handled as ``bytearray`` where it
    matters and why the threat model does not promise erasure.
    """
    for index in range(len(buffer)):
        buffer[index] = 0
