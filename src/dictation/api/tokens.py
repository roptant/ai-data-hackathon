"""Client pairing, scoped tokens and revocation (plan section 8).

Loopback is not authentication.  Any local process can reach a loopback port,
so every request carries a token that was issued through an explicit pairing
step in the desktop UI, and only token *hashes* are stored.

Scopes are separated so that reading captions does not imply permission to
start a microphone:

============================  ==================================================
``status:read``               capabilities and coarse recording state, no text
``transcript:live``           partial hypotheses while recording
``transcript:final``          the final canonical transcript
``session:control``           start, stop and cancel recording
============================  ==================================================

Raw audio and session history are not exposed at all in this API version.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from dictation.errors import ScopeDenied, Unauthorized
from dictation.logging_ import events
from dictation.store.db import Database
from dictation.types import new_id


class Scope(StrEnum):
    STATUS_READ = "status:read"
    TRANSCRIPT_LIVE = "transcript:live"
    TRANSCRIPT_FINAL = "transcript:final"
    SESSION_CONTROL = "session:control"


ALL_SCOPES = frozenset(Scope)


def parse_scopes(raw: str) -> frozenset[Scope]:
    out: set[Scope] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(Scope(part))
        except ValueError:
            continue
    return frozenset(out)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TokenRecord:
    token_id: str
    name: str
    scopes: frozenset[Scope]
    created_at: float
    revoked_at: float | None = None
    last_used: float | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def require(self, scope: Scope) -> None:
        if scope not in self.scopes:
            raise ScopeDenied(f"token lacks scope {scope}")

    def summary(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "name": self.name,
            "scopes": sorted(str(scope) for scope in self.scopes),
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
            "last_used": self.last_used,
        }


@dataclass(slots=True)
class RateLimiter:
    """Fixed-window limiter for pairing and authentication attempts."""

    limit: int
    window_s: float = 60.0
    _attempts: deque[float] = field(default_factory=deque)

    def allow(self, *, now: float | None = None) -> bool:
        stamp = time.time() if now is None else now
        while self._attempts and stamp - self._attempts[0] > self.window_s:
            self._attempts.popleft()
        if len(self._attempts) >= self.limit:
            return False
        self._attempts.append(stamp)
        return True

    def reset(self) -> None:
        self._attempts.clear()


@dataclass(frozen=True, slots=True)
class PairingRequest:
    """A short-lived code the user confirms in the desktop UI."""

    code: str
    requested_name: str
    scopes: frozenset[Scope]
    expires_at: float


class TokenManager:
    """Issues, authenticates and revokes API tokens."""

    def __init__(
        self,
        database: Database,
        *,
        auth_attempts_per_minute: int = 10,
        pairing_window_seconds: int = 120,
    ) -> None:
        self.db = database
        self.pairing_window_seconds = pairing_window_seconds
        self._limiter = RateLimiter(limit=auth_attempts_per_minute)
        self._pending: dict[str, PairingRequest] = {}

    # -- pairing -------------------------------------------------------------

    def begin_pairing(
        self,
        client_name: str,
        scopes: frozenset[Scope],
        *,
        now: float | None = None,
    ) -> PairingRequest:
        """Start pairing.  The code is displayed by the desktop UI, not the client."""
        stamp = time.time() if now is None else now
        if not self._limiter.allow(now=stamp):
            raise Unauthorized("too many pairing attempts; try again shortly")
        unknown = scopes - ALL_SCOPES
        if unknown:
            raise Unauthorized(f"unknown scopes requested: {sorted(str(s) for s in unknown)}")
        request = PairingRequest(
            code=f"{secrets.randbelow(1_000_000):06d}",
            requested_name=client_name[:64],
            scopes=frozenset(scopes),
            expires_at=stamp + self.pairing_window_seconds,
        )
        self._pending[request.code] = request
        events.emit("api.pairing_started", scopes=sorted(str(s) for s in scopes))
        return request

    def pending_pairings(self, *, now: float | None = None) -> tuple[PairingRequest, ...]:
        stamp = time.time() if now is None else now
        return tuple(r for r in self._pending.values() if r.expires_at > stamp)

    def approve_pairing(self, code: str, *, now: float | None = None) -> tuple[TokenRecord, str]:
        """User approved the code in the UI: issue the token once."""
        stamp = time.time() if now is None else now
        request = self._pending.pop(code, None)
        if request is None or request.expires_at <= stamp:
            raise Unauthorized("pairing code is unknown or expired")
        return self.issue(request.requested_name, request.scopes, now=stamp)

    def deny_pairing(self, code: str) -> None:
        self._pending.pop(code, None)

    # -- tokens --------------------------------------------------------------

    def issue(
        self,
        name: str,
        scopes: frozenset[Scope],
        *,
        now: float | None = None,
    ) -> tuple[TokenRecord, str]:
        """Create a token.  The plaintext is returned once and never stored."""
        stamp = time.time() if now is None else now
        token = secrets.token_urlsafe(32)
        token_id = new_id("token")
        self.db.insert_token(
            token_id,
            name=name[:64],
            token_hash=hash_token(token),
            scopes=",".join(sorted(str(scope) for scope in scopes)),
            now=stamp,
        )
        events.emit("api.token_issued", token_id=token_id, scopes=sorted(str(s) for s in scopes))
        record = TokenRecord(
            token_id=token_id,
            name=name[:64],
            scopes=frozenset(scopes),
            created_at=stamp,
        )
        return record, token

    def authenticate(self, token: str | None, *, now: float | None = None) -> TokenRecord:
        """Resolve a bearer token, rate-limiting failures."""
        stamp = time.time() if now is None else now
        if not token:
            raise Unauthorized("missing bearer token")
        if not self._limiter.allow(now=stamp):
            raise Unauthorized("too many authentication attempts; try again shortly")
        row = self.db.token_by_hash(hash_token(token))
        if row is None:
            # Constant-time compare against a dummy keeps the failure path
            # from leaking whether a prefix matched.
            hmac.compare_digest(hash_token(token), "0" * 64)
            raise Unauthorized("unknown or revoked token")
        self.db.touch_token(row["token_id"], now=stamp)
        return TokenRecord(
            token_id=row["token_id"],
            name=row["name"],
            scopes=parse_scopes(row["scopes"]),
            created_at=row["created_at"],
            revoked_at=row["revoked_at"],
            last_used=stamp,
        )

    def revoke(self, token_id: str, *, now: float | None = None) -> bool:
        stamp = time.time() if now is None else now
        revoked = self.db.revoke_token(token_id, now=stamp)
        if revoked:
            events.emit("api.token_revoked", token_id=token_id)
        return revoked

    def tokens(self) -> tuple[TokenRecord, ...]:
        return tuple(
            TokenRecord(
                token_id=row["token_id"],
                name=row["name"],
                scopes=parse_scopes(row["scopes"]),
                created_at=row["created_at"],
                revoked_at=row["revoked_at"],
                last_used=row["last_used"],
            )
            for row in self.db.tokens()
        )

    def reset_rate_limit(self) -> None:
        self._limiter.reset()
