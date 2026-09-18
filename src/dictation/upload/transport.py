"""Upload transport (plan section 9).

The transport carries an opaque package and an idempotency key.  It does not
know what is inside, and it cannot read the raw-session store - the worker hands
it bytes that came through the restricted
:class:`~dictation.store.session_store.PackageReader`.

Requirements enforced here: HTTPS only, bearer credentials in a header rather
than a URL, the tenant determined by authentication rather than by the package,
a size ceiling, and a checksum the server can verify independently.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dictation.errors import ServerRejected, UploadError


@dataclass(frozen=True, slots=True)
class Receipt:
    """Server acknowledgement of one sample."""

    receipt_id: str
    accepted: bool = True
    reason: str = ""
    duplicate: bool = False


@runtime_checkable
class Transport(Protocol):
    def upload(
        self,
        archive: bytes,
        *,
        idempotency_key: str,
        sample_id: str,
        sha256: str,
    ) -> Receipt: ...


@dataclass(slots=True)
class HttpsTransport:
    """Real transport over authenticated TLS."""

    endpoint: str
    token: str
    timeout_s: float = 120.0
    max_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.endpoint.lower().startswith("https://"):
            raise UploadError("upload endpoint must be HTTPS")
        if not self.token:
            raise UploadError("upload transport requires a bearer token")

    def upload(
        self,
        archive: bytes,
        *,
        idempotency_key: str,
        sample_id: str,
        sha256: str,
    ) -> Receipt:
        if len(archive) > self.max_bytes:
            raise ServerRejected("package_too_large")
        request = urllib.request.Request(
            self.endpoint,
            data=archive,
            method="POST",
            headers={
                # Credentials in a header, never in the query string.
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/zip",
                "Idempotency-Key": idempotency_key,
                "X-Sample-Id": sample_id,
                "X-Content-SHA256": sha256,
                # No tenant header: the server derives ownership from the token.
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:200]
            if error.code in {400, 409, 413, 422}:
                # A permanent refusal: the package is deleted, never retried as
                # a different artifact.
                raise ServerRejected(_reason_from(body, default=f"http_{error.code}")) from None
            raise UploadError(f"upload failed with HTTP {error.code}") from None
        except urllib.error.URLError as error:
            raise UploadError(f"upload transport error: {error.reason}") from None
        return _receipt_from(payload)


@dataclass(slots=True)
class LoopbackTransport:
    """Delivers straight into an in-process server, for tests and dry runs."""

    server: object  # dictation.server.app.IngestServer
    tenant_token: str = "test-token"
    calls: list[str] = field(default_factory=list)

    def upload(
        self,
        archive: bytes,
        *,
        idempotency_key: str,
        sample_id: str,
        sha256: str,
    ) -> Receipt:
        self.calls.append(idempotency_key)
        return self.server.admit(  # type: ignore[attr-defined]
            archive,
            token=self.tenant_token,
            idempotency_key=idempotency_key,
            declared_sha256=sha256,
        )


@dataclass(slots=True)
class FlakyTransport:
    """Fails a fixed number of times, then delegates.  Exercises retries."""

    inner: Transport
    failures: int = 1
    attempts: int = 0

    def upload(self, archive: bytes, **kwargs: object) -> Receipt:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise UploadError(f"synthetic transport failure {self.attempts}")
        return self.inner.upload(archive, **kwargs)  # type: ignore[arg-type]


@dataclass(slots=True)
class RejectingTransport:
    """Always refuses.  The package must be deleted, not re-sent raw."""

    reason: str = "package_invalid"

    def upload(self, archive: bytes, **kwargs: object) -> Receipt:
        raise ServerRejected(self.reason)


def _receipt_from(payload: object) -> Receipt:
    if not isinstance(payload, dict):
        raise UploadError("server response was not an object")
    receipt_id = str(payload.get("receipt_id", ""))
    if not receipt_id:
        raise UploadError("server response contained no receipt id")
    return Receipt(
        receipt_id=receipt_id,
        accepted=bool(payload.get("accepted", True)),
        reason=str(payload.get("reason", "")),
        duplicate=bool(payload.get("duplicate", False)),
    )


def _reason_from(body: str, *, default: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return default
    if isinstance(parsed, dict) and parsed.get("reason"):
        return str(parsed["reason"])
    return default
