"""Tenant registry for the training service (plan sections 9 and 10).

Authentication determines customer ownership.  A tenant ID supplied in an
upload package is never trusted - packages do not carry one, and
:func:`dictation.dataset.package.assert_package_clean` refuses one that does.

Only token hashes are stored, so a leaked database does not yield working
credentials.  Storage is separated per tenant: no mixing between customers, and
no promotion of one customer's data into a shared model.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dictation.errors import Unauthorized


def hash_token(token: str) -> str:
    """SHA-256 of the token.  Compared with a constant-time comparison."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Tenant:
    tenant_id: str
    name: str
    token_hash: str
    created_at: float = field(default_factory=time.time)
    #: Per-customer caps on storage and compute (plan section 12, milestone 5).
    max_samples: int = 5_000
    max_storage_bytes: int = 2 * 1024 * 1024 * 1024
    #: Set when the customer has withdrawn; admission then refuses.
    withdrawn_at: float | None = None

    @property
    def active(self) -> bool:
        return self.withdrawn_at is None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Tenant:
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]


@dataclass
class TenantRegistry:
    root: Path
    tenants: dict[str, Tenant] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.root / "tenants.json"

    @classmethod
    def load(cls, root: Path) -> TenantRegistry:
        registry = cls(root=Path(root))
        try:
            raw = json.loads(registry.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return registry
        for entry in raw.get("tenants", []):
            try:
                tenant = Tenant.from_dict(entry)
            except (TypeError, ValueError):
                continue
            registry.tenants[tenant.tenant_id] = tenant
        return registry

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"tenants": [tenant.to_dict() for tenant in self.tenants.values()]}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def create(self, name: str, *, tenant_id: str | None = None) -> tuple[Tenant, str]:
        """Create a tenant and return it with its one-time plaintext token."""
        token = secrets.token_urlsafe(32)
        identifier = tenant_id or f"tenant-{secrets.token_hex(6)}"
        tenant = Tenant(tenant_id=identifier, name=name, token_hash=hash_token(token))
        self.tenants[identifier] = tenant
        self.save()
        return tenant, token

    def authenticate(self, token: str) -> Tenant:
        """Resolve a bearer token to a tenant, or refuse."""
        digest = hash_token(token)
        for tenant in self.tenants.values():
            if hmac.compare_digest(tenant.token_hash, digest):
                if not tenant.active:
                    raise Unauthorized("tenant has withdrawn")
                return tenant
        raise Unauthorized("unknown or revoked upload credential")

    def storage_root(self, tenant: Tenant) -> Path:
        """Per-tenant object-storage stand-in.  No shared directory."""
        path = self.root / "tenants" / tenant.tenant_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mark_withdrawn(self, tenant_id: str, *, now: float | None = None) -> Tenant:
        stamp = time.time() if now is None else now
        tenant = self.tenants[tenant_id]
        updated = Tenant(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            token_hash=tenant.token_hash,
            created_at=tenant.created_at,
            max_samples=tenant.max_samples,
            max_storage_bytes=tenant.max_storage_bytes,
            withdrawn_at=stamp,
        )
        self.tenants[tenant_id] = updated
        self.save()
        return updated
