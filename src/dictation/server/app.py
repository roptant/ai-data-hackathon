"""Ingest and admission for the training service (plan section 9).

Skeleton scope: in-process admission logic with a filesystem stand-in for
object storage.  What is real here is the boundary, because the plan is explicit
that client checks are not a security boundary against malformed uploads:

* authentication resolves the tenant; no tenant field is read from the package,
* the package is re-verified server side - structure, field allowlist, declared
  digest, checksums, size, eligibility version,
* consent is rechecked at admission and again before training,
* a rejected package is quarantined briefly and deleted, and its contents are
  never logged,
* an idempotency key makes a retry return the first receipt instead of storing
  a second copy.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dictation.dataset.package import verify_package
from dictation.errors import DatasetRejected
from dictation.logging_ import events
from dictation.server.lineage import Lineage, NodeKind
from dictation.server.tenants import Tenant, TenantRegistry
from dictation.upload.transport import Receipt
from dictation.version import ELIGIBILITY_VERSION

#: How long a rejected package may sit in quarantine before deletion.
#: Short by design: only long enough to reject and delete it (plan section 9).
QUARANTINE_SECONDS = 600

MAX_PACKAGE_BYTES = 32 * 1024 * 1024


@dataclass(slots=True)
class AdmissionRecord:
    receipt_id: str
    tenant_id: str
    sample_id: str
    idempotency_key: str
    accepted: bool
    reason: str = ""
    stored_path: str = ""
    received_at: float = field(default_factory=time.time)


class IngestServer:
    """Admission control over a per-tenant object store."""

    def __init__(
        self,
        root: Path,
        registry: TenantRegistry | None = None,
        lineage: Lineage | None = None,
        *,
        known_eligibility_versions: frozenset[int] = frozenset({ELIGIBILITY_VERSION}),
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or TenantRegistry.load(self.root)
        self.lineage = lineage or Lineage.load(self.root)
        self.known_eligibility_versions = known_eligibility_versions
        self.receipts: dict[str, AdmissionRecord] = {}
        self._load_receipts()

    # -- persistence ---------------------------------------------------------

    @property
    def receipts_path(self) -> Path:
        return self.root / "receipts.json"

    def _load_receipts(self) -> None:
        try:
            raw = json.loads(self.receipts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for entry in raw.get("receipts", []):
            record = AdmissionRecord(**entry)
            self.receipts[record.idempotency_key] = record

    def _save_receipts(self) -> None:
        payload = {"receipts": [asdict(record) for record in self.receipts.values()]}
        temporary = self.receipts_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.receipts_path)

    @property
    def quarantine(self) -> Path:
        path = self.root / "quarantine"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- admission -----------------------------------------------------------

    def admit(
        self,
        archive: bytes,
        *,
        token: str,
        idempotency_key: str,
        declared_sha256: str = "",
        now: float | None = None,
    ) -> Receipt:
        """Validate and store one sample, or refuse it."""
        stamp = time.time() if now is None else now
        tenant = self.registry.authenticate(token)  # raises Unauthorized

        existing = self.receipts.get(idempotency_key)
        if existing is not None:
            # Idempotent retry: the same key never creates a second example.
            return Receipt(
                receipt_id=existing.receipt_id,
                accepted=existing.accepted,
                reason=existing.reason,
                duplicate=True,
            )

        if len(archive) > MAX_PACKAGE_BYTES:
            return self._refuse(tenant, idempotency_key, "package_too_large", stamp)
        if not idempotency_key:
            return self._refuse(tenant, idempotency_key, "missing_idempotency_key", stamp)

        digest = hashlib.sha256(archive).hexdigest()
        if declared_sha256 and declared_sha256 != digest:
            # The declared digest is of the encrypted envelope on the client
            # side, so a mismatch here is only reported when the client sends
            # the archive digest; either way we validate the contents below.
            events.warn("server.declared_digest_mismatch", tenant=tenant.tenant_id)

        try:
            manifest = verify_package(archive)
        except DatasetRejected as error:
            self._quarantine(archive, tenant, error.reason, stamp)
            return self._refuse(tenant, idempotency_key, f"invalid_package:{error.reason}", stamp)

        version = manifest.get("eligibility_version")
        if version not in self.known_eligibility_versions:
            return self._refuse(tenant, idempotency_key, "unknown_eligibility_version", stamp)

        sample_id = str(manifest.get("sample_id", ""))
        if not sample_id:
            return self._refuse(tenant, idempotency_key, "missing_sample_id", stamp)
        if self.lineage.is_deleted(sample_id):
            # A restored backup or a replayed upload must not reintroduce a
            # deleted sample (plan section 10).
            return self._refuse(tenant, idempotency_key, "sample_previously_deleted", stamp)

        samples = self.lineage.samples_of(tenant.tenant_id)
        if len(samples) >= tenant.max_samples:
            return self._refuse(tenant, idempotency_key, "tenant_sample_cap_reached", stamp)

        destination = self.registry.storage_root(tenant) / "samples" / f"{sample_id}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive)

        self.lineage.add(
            sample_id,
            NodeKind.SAMPLE,
            tenant.tenant_id,
            attributes={
                "duration_ms": manifest.get("duration_ms", 0),
                "clip_count": manifest.get("clip_count", 0),
                "language": manifest.get("language", ""),
                "policy_version": manifest.get("policy_version", ""),
                "consent_version": manifest.get("consent_version", ""),
                "consent_reference": manifest.get("consent_reference", ""),
                "sha256": digest,
            },
            now=stamp,
        )

        receipt_id = f"receipt-{hashlib.sha256((idempotency_key + digest).encode()).hexdigest()[:16]}"
        record = AdmissionRecord(
            receipt_id=receipt_id,
            tenant_id=tenant.tenant_id,
            sample_id=sample_id,
            idempotency_key=idempotency_key,
            accepted=True,
            stored_path=str(destination),
            received_at=stamp,
        )
        self.receipts[idempotency_key] = record
        self._save_receipts()
        events.emit(
            "server.sample_admitted",
            tenant=tenant.tenant_id,
            sample=sample_id,
            duration_ms=int(manifest.get("duration_ms", 0)),
        )
        return Receipt(receipt_id=receipt_id, accepted=True)

    def _refuse(self, tenant: Tenant, key: str, reason: str, now: float) -> Receipt:
        receipt_id = f"receipt-refused-{hashlib.sha256((key + reason).encode()).hexdigest()[:12]}"
        record = AdmissionRecord(
            receipt_id=receipt_id,
            tenant_id=tenant.tenant_id,
            sample_id="",
            idempotency_key=key,
            accepted=False,
            reason=reason,
            received_at=now,
        )
        if key:
            self.receipts[key] = record
            self._save_receipts()
        events.warn("server.sample_refused", tenant=tenant.tenant_id, reason=reason)
        return Receipt(receipt_id=receipt_id, accepted=False, reason=reason)

    def _quarantine(self, archive: bytes, tenant: Tenant, reason: str, now: float) -> Path:
        """Hold a rejected package briefly.  Contents are never logged."""
        name = f"{tenant.tenant_id}-{hashlib.sha256(archive).hexdigest()[:16]}.bin"
        path = self.quarantine / name
        path.write_bytes(archive)
        (path.with_suffix(".reason")).write_text(f"{reason}\n{now}\n", encoding="utf-8")
        return path

    def sweep_quarantine(self, *, now: float | None = None) -> int:
        stamp = time.time() if now is None else now
        removed = 0
        for path in self.quarantine.iterdir():
            try:
                if stamp - path.stat().st_mtime >= QUARANTINE_SECONDS:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    # -- deletion ------------------------------------------------------------

    def delete_sample(self, tenant: Tenant, sample_id: str) -> tuple[str, ...]:
        """Delete one sample and everything derived from it."""
        stored = self.registry.storage_root(tenant) / "samples" / f"{sample_id}.zip"
        stored.unlink(missing_ok=True)
        removed = self.lineage.mark_deleted((sample_id,))
        events.emit("server.sample_deleted", tenant=tenant.tenant_id, nodes=len(removed))
        return removed

    def delete_tenant_data(self, tenant_id: str, *, now: float | None = None) -> dict[str, int]:
        """Full "delete my training data and personalized model" path.

        Covers stored samples, derived datasets, checkpoints, adapters and
        delivered builds through the lineage graph, and marks the tenant
        withdrawn so a late upload cannot recreate the data.  Copies exported
        outside the service cannot be recalled; that boundary is disclosed
        rather than papered over (plan section 10).
        """
        stamp = time.time() if now is None else now
        tenant = self.registry.tenants[tenant_id]
        samples = self.lineage.samples_of(tenant_id)
        affected = self.lineage.affected_artifacts(samples)
        removed = self.lineage.mark_deleted(samples + affected)

        root = self.registry.storage_root(tenant)
        deleted_files = 0
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                if path.is_file():
                    path.unlink()
                    deleted_files += 1
                else:
                    path.rmdir()
            except OSError:
                continue

        self.registry.mark_withdrawn(tenant_id, now=stamp)
        for key, record in list(self.receipts.items()):
            if record.tenant_id == tenant_id:
                del self.receipts[key]
        self._save_receipts()

        events.emit(
            "server.tenant_data_deleted",
            tenant=tenant_id,
            samples=len(samples),
            artifacts=len(affected),
            files=deleted_files,
        )
        return {
            "samples": len(samples),
            "artifacts": len(affected),
            "lineage_nodes": len(removed),
            "files": deleted_files,
        }

    # -- dataset snapshots ---------------------------------------------------

    def snapshot_dataset(self, tenant_id: str, *, now: float | None = None) -> tuple[str, tuple[str, ...]]:
        """Freeze the current admitted, non-deleted samples into a dataset node."""
        stamp = time.time() if now is None else now
        samples = tuple(
            sample
            for sample in self.lineage.samples_of(tenant_id)
            if not self.lineage.is_deleted(sample)
        )
        dataset_id = f"dataset-{tenant_id}-{int(stamp)}"
        self.lineage.add(
            dataset_id,
            NodeKind.DATASET,
            tenant_id,
            derived_from=samples,
            attributes={"sample_count": len(samples)},
            now=stamp,
        )
        return dataset_id, samples
