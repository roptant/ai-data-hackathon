"""Server admission, lineage, deletion and the promotion gates (plan 9-10)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from dictation.dataset.builder import build_dataset
from dictation.dataset.package import MANIFEST_NAME, build_package
from dictation.errors import Unauthorized
from dictation.privacy.classifier.mock import RuleEchoClassifier
from dictation.privacy.pipeline import analyze
from dictation.server.app import IngestServer
from dictation.server.lineage import Lineage, NodeKind
from dictation.server.tenants import TenantRegistry, hash_token
from dictation.server.training import (
    DatasetSplit,
    DeliveryRegistry,
    Evaluation,
    PromotionPolicy,
    TrainerUnavailable,
    TrainingRun,
    UnavailableTrainer,
    decide_promotion,
    default_runtime_probe,
    generate_signing_key,
    load_public_key,
    sign_manifest,
    split_by_time,
    verify_manifest,
)
from dictation.version import CONSENT_VERSION

from tests.conftest import Session


@pytest.fixture
def ingest(tmp_path):
    registry = TenantRegistry.load(tmp_path / "server")
    tenant, token = registry.create("Customer A", tenant_id="tenant-a")
    other, other_token = registry.create("Customer B", tenant_id="tenant-b")
    server = IngestServer(tmp_path / "server", registry)
    return server, tenant, token, other, other_token


def package_for(session: Session):
    analysis = analyze(session.transcript, RuleEchoClassifier())
    result = build_dataset(analysis, session.pcm)
    assert result.eligible, result.reason
    return build_package(result, consent_version=CONSENT_VERSION, consent_reference="ref-1")


# -- authentication ----------------------------------------------------------


def test_only_the_token_hash_is_stored(ingest) -> None:
    server, tenant, token, *_ = ingest
    assert tenant.token_hash == hash_token(token)
    assert token not in server.registry.path.read_text(encoding="utf-8")


def test_unknown_credential_is_refused(ingest, parcel_session: Session) -> None:
    server, *_ = ingest
    package = package_for(parcel_session)
    with pytest.raises(Unauthorized):
        server.admit(package.archive_bytes, token="wrong", idempotency_key="k1")


def test_withdrawn_tenant_cannot_upload(ingest, parcel_session: Session) -> None:
    server, tenant, token, *_ = ingest
    server.registry.mark_withdrawn(tenant.tenant_id)
    with pytest.raises(Unauthorized):
        server.admit(package_for(parcel_session).archive_bytes, token=token, idempotency_key="k")


# -- admission ---------------------------------------------------------------


def test_valid_package_is_admitted(ingest, parcel_session: Session) -> None:
    server, tenant, token, *_ = ingest
    package = package_for(parcel_session)
    receipt = server.admit(package.archive_bytes, token=token, idempotency_key="k1")
    assert receipt.accepted
    assert server.lineage.samples_of(tenant.tenant_id) == (package.sample_id,)


def test_retry_with_the_same_key_returns_the_first_receipt(ingest, parcel_session) -> None:
    server, tenant, token, *_ = ingest
    package = package_for(parcel_session)
    first = server.admit(package.archive_bytes, token=token, idempotency_key="k1")
    second = server.admit(package.archive_bytes, token=token, idempotency_key="k1")
    assert second.receipt_id == first.receipt_id
    assert second.duplicate
    assert len(server.lineage.samples_of(tenant.tenant_id)) == 1


def test_corrupt_archive_is_refused_and_quarantined(ingest) -> None:
    server, tenant, token, *_ = ingest
    receipt = server.admit(b"not a zip file", token=token, idempotency_key="k1")
    assert not receipt.accepted
    assert receipt.reason.startswith("invalid_package")
    assert list(server.quarantine.iterdir())


def test_checksum_tampering_is_detected(ingest, parcel_session: Session) -> None:
    server, tenant, token, *_ = ingest
    package = package_for(parcel_session)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(package.archive_bytes)) as source:
        names = source.namelist()
        with zipfile.ZipFile(buffer, "w") as target:
            for name in names:
                payload = source.read(name)
                if name.endswith(".txt"):
                    payload = b"different text"
                target.writestr(name, payload)
    receipt = server.admit(buffer.getvalue(), token=token, idempotency_key="k1")
    assert not receipt.accepted


def test_package_with_a_tenant_field_is_refused(ingest, parcel_session: Session) -> None:
    """Client-side stripping is not the security boundary; the server rechecks."""
    server, tenant, token, *_ = ingest
    package = package_for(parcel_session)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(package.archive_bytes)) as source:
        manifest = json.loads(source.read(MANIFEST_NAME))
        manifest["tenant_id"] = "tenant-b"
        with zipfile.ZipFile(buffer, "w") as target:
            for name in source.namelist():
                payload = (
                    json.dumps(manifest).encode() if name == MANIFEST_NAME else source.read(name)
                )
                target.writestr(name, payload)
    receipt = server.admit(buffer.getvalue(), token=token, idempotency_key="k1")
    assert not receipt.accepted
    assert "forbidden_field" in receipt.reason or "not_allowed" in receipt.reason


def test_unknown_eligibility_version_is_refused(tmp_path, parcel_session: Session) -> None:
    registry = TenantRegistry.load(tmp_path / "s")
    _, token = registry.create("Customer", tenant_id="t")
    server = IngestServer(tmp_path / "s", registry, known_eligibility_versions=frozenset({99}))
    receipt = server.admit(
        package_for(parcel_session).archive_bytes, token=token, idempotency_key="k"
    )
    assert receipt.reason == "unknown_eligibility_version"


def test_missing_idempotency_key_is_refused(ingest, parcel_session: Session) -> None:
    server, tenant, token, *_ = ingest
    receipt = server.admit(package_for(parcel_session).archive_bytes, token=token, idempotency_key="")
    assert not receipt.accepted


def test_tenant_sample_cap_is_enforced(ingest, parcel_session, clean_session) -> None:
    server, tenant, token, *_ = ingest
    server.registry.tenants[tenant.tenant_id] = type(tenant)(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        token_hash=tenant.token_hash,
        max_samples=1,
    )
    server.admit(package_for(parcel_session).archive_bytes, token=token, idempotency_key="k1")
    receipt = server.admit(
        package_for(clean_session).archive_bytes, token=token, idempotency_key="k2"
    )
    assert receipt.reason == "tenant_sample_cap_reached"


def test_tenants_are_isolated_on_disk(ingest, parcel_session, clean_session) -> None:
    server, tenant, token, other, other_token = ingest
    server.admit(package_for(parcel_session).archive_bytes, token=token, idempotency_key="k1")
    server.admit(package_for(clean_session).archive_bytes, token=other_token, idempotency_key="k2")
    a_root = server.registry.storage_root(tenant)
    b_root = server.registry.storage_root(other)
    assert a_root != b_root
    assert len(list((a_root / "samples").iterdir())) == 1
    assert len(list((b_root / "samples").iterdir())) == 1
    assert len(server.lineage.samples_of(tenant.tenant_id)) == 1


def test_quarantine_is_swept(ingest) -> None:
    server, tenant, token, *_ = ingest
    server.admit(b"junk", token=token, idempotency_key="k1")
    assert server.sweep_quarantine(now=1e12) >= 1
    assert list(server.quarantine.iterdir()) == []


# -- deletion ----------------------------------------------------------------


def test_deleting_a_sample_marks_it_and_prevents_reupload(ingest, parcel_session) -> None:
    server, tenant, token, *_ = ingest
    package = package_for(parcel_session)
    server.admit(package.archive_bytes, token=token, idempotency_key="k1")
    server.delete_sample(tenant, package.sample_id)
    assert server.lineage.is_deleted(package.sample_id)
    receipt = server.admit(package.archive_bytes, token=token, idempotency_key="k2")
    assert receipt.reason == "sample_previously_deleted"


def test_full_deletion_covers_derived_artifacts(ingest, parcel_session: Session) -> None:
    server, tenant, token, *_ = ingest
    package = package_for(parcel_session)
    server.admit(package.archive_bytes, token=token, idempotency_key="k1")
    dataset_id, samples = server.snapshot_dataset(tenant.tenant_id)
    server.lineage.add("job-1", NodeKind.JOB, tenant.tenant_id, derived_from=(dataset_id,))
    server.lineage.add("adapter-1", NodeKind.ADAPTER, tenant.tenant_id, derived_from=("job-1",))
    server.lineage.add("build-1", NodeKind.BUILD, tenant.tenant_id, derived_from=("adapter-1",))

    affected = server.lineage.affected_artifacts(samples)
    assert set(affected) == {"adapter-1", "build-1"}

    report = server.delete_tenant_data(tenant.tenant_id)
    assert report["samples"] == 1
    assert report["artifacts"] == 2
    assert server.lineage.is_deleted("adapter-1")
    assert not list((server.registry.storage_root(tenant) / "samples").glob("*.zip"))


def test_restore_guard_refuses_a_deleted_node(tmp_path) -> None:
    lineage = Lineage.load(tmp_path)
    lineage.add("sample-1", NodeKind.SAMPLE, "t")
    lineage.mark_deleted(("sample-1",))
    with pytest.raises(ValueError):
        lineage.restore_guard("sample-1")


def test_lineage_round_trips(tmp_path) -> None:
    lineage = Lineage.load(tmp_path)
    lineage.add("sample-1", NodeKind.SAMPLE, "t")
    lineage.add("dataset-1", NodeKind.DATASET, "t", derived_from=("sample-1",))
    reopened = Lineage.load(tmp_path)
    assert reopened.descendants("sample-1") == ("dataset-1",)
    assert reopened.ancestors("dataset-1") == ("sample-1",)


# -- training and promotion --------------------------------------------------


def good_evaluation(**overrides) -> Evaluation:
    values = {
        "base_wer": 0.20,
        "candidate_wer": 0.16,
        "general_base_wer": 0.10,
        "general_candidate_wer": 0.10,
        "heldout_examples": 50,
        "training_examples": 400,
        "peak_memory_mib": 900,
        "artifact_bytes": 50 * 1024 * 1024,
        "realtime_factor": 0.6,
    }
    values.update(overrides)
    return Evaluation(**values)  # type: ignore[arg-type]


def test_promotion_requires_measured_improvement() -> None:
    assert decide_promotion(PromotionPolicy(), good_evaluation())
    assert not decide_promotion(PromotionPolicy(), good_evaluation(candidate_wer=0.199))
    assert (
        decide_promotion(PromotionPolicy(), good_evaluation(candidate_wer=0.199)).reason
        == "no_measured_improvement"
    )


def test_promotion_refuses_a_general_regression() -> None:
    decision = decide_promotion(PromotionPolicy(), good_evaluation(general_candidate_wer=0.15))
    assert decision.reason == "general_regression_exceeded"


def test_promotion_refuses_too_little_data() -> None:
    assert (
        decide_promotion(PromotionPolicy(), good_evaluation(training_examples=10)).reason
        == "insufficient_training_data"
    )
    assert (
        decide_promotion(PromotionPolicy(), good_evaluation(heldout_examples=2)).reason
        == "insufficient_heldout_data"
    )


def test_promotion_refuses_a_model_over_the_device_budget() -> None:
    assert (
        decide_promotion(PromotionPolicy(), good_evaluation(peak_memory_mib=9000)).reason
        == "exceeds_device_memory_budget"
    )
    assert (
        decide_promotion(PromotionPolicy(), good_evaluation(realtime_factor=2.0)).reason
        == "too_slow_for_realtime"
    )


def test_split_holds_out_the_most_recent_samples() -> None:
    split = split_by_time([f"s{i}" for i in range(10)], heldout_fraction=0.2)
    assert split.heldout == ("s8", "s9")
    assert "s8" not in split.train
    assert split.sizes == (8, 2)


def test_no_trainer_means_no_promotion(tmp_path) -> None:
    lineage = Lineage.load(tmp_path)
    run = TrainingRun(
        tenant_id="t",
        dataset_id="dataset-1",
        split=DatasetSplit(tuple(f"s{i}" for i in range(500)), ("s500",)),
    )
    decision = run.run(lineage, tmp_path)
    assert not decision
    assert decision.reason == "trainer_unavailable"


def test_unavailable_trainer_explains_the_missing_decision(tmp_path) -> None:
    with pytest.raises(TrainerUnavailable, match="export"):
        UnavailableTrainer().train("t", DatasetSplit((), ()), tmp_path)


def test_runtime_probe_fails_closed(tmp_path) -> None:
    loads, reason = default_runtime_probe(tmp_path / "artifact.gguf")
    assert not loads
    assert reason == "desktop_runtime_probe_not_implemented"


def test_artifact_that_cannot_load_is_not_promoted(tmp_path) -> None:
    lineage = Lineage.load(tmp_path)

    class Fake:
        method = "adapter"

        def train(self, tenant_id, split, workdir):
            path = workdir / "adapter.gguf"
            path.write_bytes(b"weights")
            return path

    run = TrainingRun(
        tenant_id="t",
        dataset_id="d",
        split=DatasetSplit(tuple(f"s{i}" for i in range(500)), ("s500",)),
    )
    decision = run.run(lineage, tmp_path, trainer=Fake())
    assert not decision
    assert decision.reason.startswith("runtime_probe_failed")


def test_a_promotable_run_records_lineage(tmp_path) -> None:
    lineage = Lineage.load(tmp_path)

    class Fake:
        method = "adapter"

        def train(self, tenant_id, split, workdir):
            path = workdir / "adapter.gguf"
            path.write_bytes(b"weights")
            return path

    run = TrainingRun(
        tenant_id="t",
        dataset_id="d",
        split=DatasetSplit(tuple(f"s{i}" for i in range(500)), tuple(f"h{i}" for i in range(50))),
    )
    decision = run.run(
        lineage,
        tmp_path,
        trainer=Fake(),
        evaluator=lambda artifact, split: good_evaluation(),
        runtime_probe=lambda path: (True, ""),
    )
    assert decision.promote
    assert any(node.kind is NodeKind.CHECKPOINT for node in lineage.nodes.values())


# -- signed delivery ---------------------------------------------------------


def test_manifest_signature_round_trips() -> None:
    private, public_bytes = generate_signing_key()
    manifest = {"build": "build-1", "sha256": "a" * 64, "size": 123}
    signed = sign_manifest(private, manifest)
    assert verify_manifest(load_public_key(public_bytes), signed)


def test_tampered_manifest_fails_verification() -> None:
    private, public_bytes = generate_signing_key()
    signed = sign_manifest(private, {"build": "build-1"})
    tampered = type(signed)(manifest={"build": "build-2"}, signature=signed.signature)
    assert not verify_manifest(load_public_key(public_bytes), tampered)


def test_manifest_from_another_key_fails_verification() -> None:
    private, _ = generate_signing_key()
    _, other_public = generate_signing_key()
    signed = sign_manifest(private, {"build": "build-1"})
    assert not verify_manifest(load_public_key(other_public), signed)


def test_delivery_registry_supports_rollback_and_revocation() -> None:
    registry = DeliveryRegistry()
    registry.promote("t", "build-1")
    registry.promote("t", "build-2")
    assert registry.active_build("t") == "build-2"
    assert registry.rollback("t") == "build-1"
    assert registry.rollback("t") is None  # falls back to the base model
    registry.promote("t", "build-3")
    registry.revoke("t")
    assert registry.active_build("t") is None
