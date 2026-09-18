"""Upload worker (plan section 9).

One pass over the queue: claim an eligible job, read *only* its package through
the restricted reader, verify the archive against its own checksums, send it
with an idempotency key, and record the receipt.

What the worker deliberately cannot do:

* read raw audio or transcripts - it holds a
  :class:`~dictation.store.session_store.PackageReader`, not the store,
* re-send anything other than the package it verified, so neither a transport
  failure nor a server rejection can escalate into uploading the original
  recording,
* proceed on stale consent - the queue rechecks at the claim, and the worker
  rechecks again after a slow transfer.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from dictation.consent.consent import ConsentManager
from dictation.dataset.package import verify_package
from dictation.errors import DatasetRejected, ServerRejected, StateTransitionConflict, UploadError
from dictation.logging_ import events
from dictation.store.session_store import PackageReader
from dictation.upload.queue import Job, UploadQueue
from dictation.upload.transport import Receipt, Transport


@dataclass(slots=True)
class UploadReport:
    attempted: int = 0
    acknowledged: int = 0
    retried: int = 0
    rejected: int = 0
    cancelled: int = 0
    reasons: list[str] = field(default_factory=list)


class UploadWorker:
    def __init__(
        self,
        queue: UploadQueue,
        transport: Transport,
        reader: PackageReader,
        consent: ConsentManager,
    ) -> None:
        self.queue = queue
        self.transport = transport
        self.reader = reader
        self.consent = consent

    def run_once(self, *, now: float | None = None, limit: int = 10) -> UploadReport:
        stamp = time.time() if now is None else now
        report = UploadReport()
        for job in self.queue.ready_for_upload(now=stamp)[:limit]:
            self._process(job, report, now=stamp)
        return report

    def _process(self, job: Job, report: UploadReport, *, now: float) -> None:
        report.attempted += 1
        try:
            claimed = self.queue.claim_for_upload(job.job_id, now=now)
        except StateTransitionConflict:
            report.reasons.append("claim_conflict")
            return
        if claimed.state.value != "uploading":
            # The claim turned into a cancellation or a deferral.
            report.cancelled += 1
            report.reasons.append(claimed.reason or claimed.state.value)
            return

        package = self.reader.find_package(claimed.session_id)
        if package is None:
            self.queue.reject(claimed.job_id, "package_missing", now=now)
            report.rejected += 1
            report.reasons.append("package_missing")
            return

        try:
            archive = self.reader.read(package.artifact_id)
            verify_package(archive)
        except DatasetRejected as error:
            self.queue.reject(claimed.job_id, error.reason, now=now)
            report.rejected += 1
            report.reasons.append(error.reason)
            return
        except Exception:  # noqa: BLE001 - unreadable package is not retriable
            self.queue.reject(claimed.job_id, "package_unreadable", now=now)
            report.rejected += 1
            report.reasons.append("package_unreadable")
            return

        try:
            receipt: Receipt = self.transport.upload(
                archive,
                idempotency_key=claimed.idempotency_key,
                sample_id=claimed.sample_id,
                # Digest of the archive as sent, so the server can verify
                # the bytes it received rather than a local ciphertext digest.
                sha256=hashlib.sha256(archive).hexdigest(),
            )
        except ServerRejected as error:
            self.queue.reject(claimed.job_id, f"server_rejected:{error.reason}", now=now)
            report.rejected += 1
            report.reasons.append(error.reason)
            return
        except UploadError as error:
            self.queue.fail_upload(claimed.job_id, "transport_error", retriable=True, now=now)
            report.retried += 1
            report.reasons.append(str(error)[:64])
            return

        if not receipt.accepted:
            self.queue.reject(claimed.job_id, f"server_refused:{receipt.reason}", now=now)
            report.rejected += 1
            report.reasons.append(receipt.reason or "server_refused")
            return

        # A transfer can outlast a withdrawal.  Recheck before treating the
        # receipt as an acknowledgement; a late receipt is recorded as such.
        after = self.consent.recheck(claimed.consent_id)
        if not after:
            self.queue.handle_late_receipt(claimed.job_id, receipt.receipt_id)
            self.queue.cancel(claimed.job_id, after.reason)
            report.cancelled += 1
            report.reasons.append(after.reason)
            return

        self.queue.acknowledge(claimed.job_id, receipt.receipt_id, now=now)
        report.acknowledged += 1
        events.emit(
            "upload.acknowledged",
            job=claimed.job_id,
            sample=claimed.sample_id,
            duplicate=receipt.duplicate,
        )
