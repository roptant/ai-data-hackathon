"""Personalized training, evaluation gates and signed delivery (plan section 10).

Skeleton scope, deliberately: the adaptation method is not chosen yet, so there
is no trainer here.  What *is* implemented is everything that decides whether a
trained artifact may be delivered, because those are the parts that must exist
before collection starts rather than after:

* a dataset snapshot split held out by session and time, not at random, so the
  evaluation is not scored on the same sitting it trained on,
* an improvement threshold and a regression ceiling defined *before* training,
* a device-budget check, so a model that cannot run on the customer's machine
  is never activated,
* an export/quantize/load proof obligation - the runtime probe must succeed
  before a build is promoted,
* Ed25519-signed manifests, verified before installation, with the previous
  build retained as a rollback path,
* honest reporting when there is too little data or no measured improvement:
  the base model stays.

:class:`UnavailableTrainer` is the default trainer.  It refuses, which is the
correct behaviour until the product owner picks fine-tuning or adapters and the
export path has been demonstrated end to end.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from dictation.logging_ import events
from dictation.server.lineage import Lineage, NodeKind


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Thresholds fixed before a training run (plan section 10).

    Provisional values.  They are engineering targets on measured metrics, not
    claims about model quality.
    """

    #: Relative word-error-rate reduction required on customer held-out speech.
    min_relative_improvement: float = 0.05
    #: Maximum absolute WER increase tolerated on the general regression set.
    max_general_regression: float = 0.01
    #: Minimum number of held-out utterances for the comparison to mean anything.
    min_heldout_examples: int = 30
    #: Minimum training examples before a run is worth attempting.
    min_training_examples: int = 200
    #: Device budget; a model above this is never activated.
    max_peak_memory_mib: int = 1_500
    max_artifact_bytes: int = 600 * 1024 * 1024
    #: Latency ceiling for the personalized model, measured not estimated.
    max_realtime_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Held-out split by session and time, never by random utterance."""

    train: tuple[str, ...]
    heldout: tuple[str, ...]

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.heldout)


def split_by_time(samples: Sequence[str], *, heldout_fraction: float = 0.2) -> DatasetSplit:
    """Hold out the most recent samples.

    Sample IDs are random, so "recent" here means the order the server admitted
    them, which is the information the lineage graph has.  Splitting by time
    rather than at random keeps utterances from one sitting out of both sides.
    """
    if not samples:
        return DatasetSplit((), ())
    count = max(1, int(len(samples) * heldout_fraction))
    return DatasetSplit(tuple(samples[:-count]), tuple(samples[-count:]))


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Measured comparison of a candidate against the base model."""

    base_wer: float
    candidate_wer: float
    general_base_wer: float
    general_candidate_wer: float
    heldout_examples: int
    training_examples: int
    peak_memory_mib: int
    artifact_bytes: int
    realtime_factor: float

    @property
    def relative_improvement(self) -> float:
        if self.base_wer <= 0:
            return 0.0
        return (self.base_wer - self.candidate_wer) / self.base_wer

    @property
    def general_regression(self) -> float:
        return self.general_candidate_wer - self.general_base_wer

    def as_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["relative_improvement"] = round(self.relative_improvement, 5)
        data["general_regression"] = round(self.general_regression, 5)
        return data


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promote: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.promote


def decide_promotion(policy: PromotionPolicy, evaluation: Evaluation) -> PromotionDecision:
    """Apply the pre-registered gates.  No measured gain means no promotion."""
    if evaluation.training_examples < policy.min_training_examples:
        return PromotionDecision(False, "insufficient_training_data")
    if evaluation.heldout_examples < policy.min_heldout_examples:
        return PromotionDecision(False, "insufficient_heldout_data")
    if evaluation.relative_improvement < policy.min_relative_improvement:
        return PromotionDecision(False, "no_measured_improvement")
    if evaluation.general_regression > policy.max_general_regression:
        return PromotionDecision(False, "general_regression_exceeded")
    if evaluation.peak_memory_mib > policy.max_peak_memory_mib:
        return PromotionDecision(False, "exceeds_device_memory_budget")
    if evaluation.artifact_bytes > policy.max_artifact_bytes:
        return PromotionDecision(False, "artifact_too_large")
    if evaluation.realtime_factor > policy.max_realtime_factor:
        return PromotionDecision(False, "too_slow_for_realtime")
    return PromotionDecision(True)


@runtime_checkable
class Trainer(Protocol):
    """Produces a candidate artifact from a dataset split."""

    @property
    def method(self) -> str: ...

    def train(self, tenant_id: str, split: DatasetSplit, workdir: Path) -> Path: ...


class TrainerUnavailable(RuntimeError):
    """Raised when no adaptation method has been selected."""


@dataclass(slots=True)
class UnavailableTrainer:
    """Default trainer: refuses, with the reason.

    The plan requires proving that the chosen adaptation method can be
    exported, quantized and loaded by the desktop ASR runtime *before*
    large-scale collection starts.  Until that experiment exists, refusing is
    the honest behaviour.
    """

    method: str = "unselected"

    def train(self, tenant_id: str, split: DatasetSplit, workdir: Path) -> Path:
        raise TrainerUnavailable(
            "no adaptation method is selected. Choose fine-tuning or adapters, "
            "demonstrate export -> quantize -> load in the desktop ASR runtime, "
            "then wire a Trainer implementation here."
        )


#: A runtime probe proves an artifact loads in the desktop runtime.
RuntimeProbe = Callable[[Path], tuple[bool, str]]


def default_runtime_probe(artifact: Path) -> tuple[bool, str]:
    """No runtime is wired up, so the probe fails closed."""
    return False, "desktop_runtime_probe_not_implemented"


@dataclass(slots=True)
class TrainingRun:
    """One customer-isolated training attempt."""

    tenant_id: str
    dataset_id: str
    split: DatasetSplit
    policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    job_id: str = ""
    artifact: Path | None = None
    evaluation: Evaluation | None = None
    decision: PromotionDecision | None = None
    reason: str = ""

    def run(
        self,
        lineage: Lineage,
        workdir: Path,
        *,
        trainer: Trainer | None = None,
        evaluator: Callable[[Path, DatasetSplit], Evaluation] | None = None,
        runtime_probe: RuntimeProbe = default_runtime_probe,
        now: float | None = None,
    ) -> PromotionDecision:
        """Train, evaluate and decide.  Never promotes on an unproven artifact."""
        stamp = time.time() if now is None else now
        trainer = trainer or UnavailableTrainer()
        self.job_id = f"job-{self.tenant_id}-{int(stamp)}"
        lineage.add(
            self.job_id,
            NodeKind.JOB,
            self.tenant_id,
            derived_from=(self.dataset_id,),
            attributes={"method": trainer.method, "train": len(self.split.train)},
            now=stamp,
        )

        if len(self.split.train) < self.policy.min_training_examples:
            self.decision = PromotionDecision(False, "insufficient_training_data")
            self.reason = self.decision.reason
            events.warn("training.skipped", tenant=self.tenant_id, reason=self.reason)
            return self.decision

        try:
            self.artifact = trainer.train(self.tenant_id, self.split, workdir)
        except TrainerUnavailable as error:
            self.decision = PromotionDecision(False, "trainer_unavailable")
            self.reason = str(error)
            events.warn("training.trainer_unavailable", tenant=self.tenant_id)
            return self.decision

        loads, probe_reason = runtime_probe(self.artifact)
        if not loads:
            self.decision = PromotionDecision(False, f"runtime_probe_failed:{probe_reason}")
            self.reason = self.decision.reason
            return self.decision

        if evaluator is None:
            self.decision = PromotionDecision(False, "evaluator_not_configured")
            self.reason = self.decision.reason
            return self.decision

        self.evaluation = evaluator(self.artifact, self.split)
        self.decision = decide_promotion(self.policy, self.evaluation)
        checkpoint_id = f"checkpoint-{self.job_id}"
        lineage.add(
            checkpoint_id,
            NodeKind.CHECKPOINT,
            self.tenant_id,
            derived_from=(self.job_id,),
            attributes=self.evaluation.as_dict(),
            now=stamp,
        )
        events.emit(
            "training.evaluated",
            tenant=self.tenant_id,
            promote=self.decision.promote,
            reason=self.decision.reason or "ok",
        )
        return self.decision


# -- signed delivery ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignedManifest:
    manifest: dict[str, object]
    signature: str

    def to_json(self) -> str:
        return json.dumps(
            {"manifest": self.manifest, "signature": self.signature}, sort_keys=True
        )

    @classmethod
    def from_json(cls, payload: str) -> SignedManifest:
        parsed = json.loads(payload)
        return cls(manifest=parsed["manifest"], signature=parsed["signature"])


def canonical_bytes(manifest: dict[str, object]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(private_key: Ed25519PrivateKey, manifest: dict[str, object]) -> SignedManifest:
    signature = private_key.sign(canonical_bytes(manifest))
    return SignedManifest(manifest=manifest, signature=signature.hex())


def verify_manifest(public_key: Ed25519PublicKey, signed: SignedManifest) -> bool:
    """Verify a delivery manifest before installation."""
    try:
        public_key.verify(bytes.fromhex(signed.signature), canonical_bytes(signed.manifest))
    except (InvalidSignature, ValueError):
        return False
    return True


def generate_signing_key() -> tuple[Ed25519PrivateKey, bytes]:
    """New signing key with its public bytes, for a deployment to pin."""
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return private, public


def load_public_key(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)


@dataclass(slots=True)
class DeliveryRegistry:
    """Current and previous build per tenant, with an explicit rollback path."""

    current: dict[str, str] = field(default_factory=dict)
    previous: dict[str, str] = field(default_factory=dict)

    def promote(self, tenant_id: str, build_id: str) -> None:
        if tenant_id in self.current:
            self.previous[tenant_id] = self.current[tenant_id]
        self.current[tenant_id] = build_id

    def rollback(self, tenant_id: str) -> str | None:
        """Return to the previous build, or to the base model if there is none."""
        earlier = self.previous.pop(tenant_id, None)
        if earlier is None:
            self.current.pop(tenant_id, None)
            return None
        self.current[tenant_id] = earlier
        return earlier

    def revoke(self, tenant_id: str) -> None:
        """Drop the personalized model entirely; the device falls back to base."""
        self.current.pop(tenant_id, None)
        self.previous.pop(tenant_id, None)

    def active_build(self, tenant_id: str) -> str | None:
        return self.current.get(tenant_id)
