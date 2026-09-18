"""Encrypted store, guarded transitions and retention (plan section 6)."""

from __future__ import annotations

import time

import pytest

from dictation.config import RetentionSettings
from dictation.errors import DecryptionError, SecureStorageUnavailable, StateTransitionConflict, StoreError
from dictation.paths import DataPaths
from dictation.store.crypto import PayloadContext, decrypt, encrypt, overwrite
from dictation.store.db import Database
from dictation.store.keys import EphemeralKeyStore, KeyringKeyStore
from dictation.store.retention import (
    RetentionPolicy,
    delete_after_decision,
    run_cleanup,
    sweep_orphans,
)
from dictation.store.session_store import ArtifactKind, SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    paths = DataPaths.create(tmp_path / "data")
    database = Database.open(paths.database)
    return SessionStore(paths, database, EphemeralKeyStore())


# -- crypto ------------------------------------------------------------------


def test_payload_round_trips() -> None:
    key = b"k" * 32
    context = PayloadContext(session_id="s1", kind="raw_audio")
    envelope = encrypt(key, b"secret audio", context)
    assert b"secret audio" not in envelope
    assert decrypt(key, envelope, context) == b"secret audio"


def test_ciphertext_is_bound_to_the_session() -> None:
    key = b"k" * 32
    envelope = encrypt(key, b"payload", PayloadContext(session_id="s1", kind="raw_audio"))
    with pytest.raises(DecryptionError):
        decrypt(key, envelope, PayloadContext(session_id="s2", kind="raw_audio"))


def test_ciphertext_is_bound_to_the_kind() -> None:
    key = b"k" * 32
    envelope = encrypt(key, b"payload", PayloadContext(session_id="s1", kind="raw_audio"))
    with pytest.raises(DecryptionError):
        decrypt(key, envelope, PayloadContext(session_id="s1", kind="package"))


def test_tampering_is_detected() -> None:
    key = b"k" * 32
    context = PayloadContext(session_id="s1", kind="raw_audio")
    envelope = bytearray(encrypt(key, b"payload", context))
    envelope[-1] ^= 0x01
    with pytest.raises(DecryptionError):
        decrypt(key, bytes(envelope), context)


def test_nonces_differ_between_payloads() -> None:
    key = b"k" * 32
    context = PayloadContext(session_id="s1", kind="raw_audio")
    first = encrypt(key, b"same", context)
    second = encrypt(key, b"same", context)
    assert first != second


def test_short_payload_is_not_mistaken_for_an_envelope() -> None:
    with pytest.raises(DecryptionError):
        decrypt(b"k" * 32, b"tiny", PayloadContext(session_id="s", kind="k"))


def test_wrong_key_length_is_rejected() -> None:
    with pytest.raises(ValueError):
        encrypt(b"short", b"payload", PayloadContext(session_id="s", kind="k"))


def test_overwrite_zeroes_a_buffer() -> None:
    buffer = bytearray(b"secret")
    overwrite(buffer)
    assert bytes(buffer) == b"\x00" * 6


# -- key store ---------------------------------------------------------------


def test_session_keys_are_distinct_per_session_and_kind() -> None:
    keys = EphemeralKeyStore()
    a = keys.session_key("s1", "raw_audio")
    b = keys.session_key("s2", "raw_audio")
    c = keys.session_key("s1", "package")
    assert len({a, b, c}) == 3
    assert keys.session_key("s1", "raw_audio") == a


def test_destroying_a_session_key_makes_it_unusable() -> None:
    keys = EphemeralKeyStore()
    keys.session_key("s1", "raw_audio")
    keys.destroy_session_key("s1")
    with pytest.raises(SecureStorageUnavailable):
        keys.session_key("s1", "raw_audio")


def test_ephemeral_store_reports_that_it_is_not_persistent() -> None:
    assert EphemeralKeyStore().persistent is False


def test_keyring_backend_availability_is_a_query_not_a_crash() -> None:
    # Whatever this machine has, asking must not raise.
    assert isinstance(KeyringKeyStore.available(), bool)


# -- session store -----------------------------------------------------------


def test_stored_payload_is_encrypted_on_disk(store: SessionStore) -> None:
    artifact = store.put(
        "s1", ArtifactKind.RAW_AUDIO, b"raw audio bytes", expires_at=time.time() + 60
    )
    on_disk = artifact.path.read_bytes()
    assert b"raw audio bytes" not in on_disk
    assert store.get(artifact.artifact_id) == b"raw audio bytes"


def test_digest_mismatch_is_detected(store: SessionStore) -> None:
    artifact = store.put("s1", ArtifactKind.PACKAGE, b"package", expires_at=time.time() + 60)
    artifact.path.write_bytes(artifact.path.read_bytes() + b"x")
    with pytest.raises(StoreError):
        store.get(artifact.artifact_id)


def test_deleting_a_session_removes_the_files(store: SessionStore) -> None:
    artifact = store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=time.time() + 60)
    path = artifact.path
    store.delete_session("s1")
    assert not path.exists()
    assert store.db.artifacts_of("s1") == []


def test_package_reader_cannot_read_raw_audio(store: SessionStore) -> None:
    raw = store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=time.time() + 60)
    package = store.put("s1", ArtifactKind.PACKAGE, b"zip", expires_at=time.time() + 60)
    reader = store.package_reader()
    assert reader.read(package.artifact_id) == b"zip"
    with pytest.raises(StoreError):
        reader.read(raw.artifact_id)


def test_package_reader_cannot_delete_raw_audio(store: SessionStore) -> None:
    raw = store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=time.time() + 60)
    with pytest.raises(StoreError):
        store.package_reader().delete_package(raw.artifact_id)


def test_payload_root_is_outside_the_repository(store: SessionStore) -> None:
    assert not store.paths.inside_repository()


def test_unsafe_session_ids_cannot_escape_the_root(store: SessionStore) -> None:
    with pytest.raises(ValueError):
        store.paths.session_dir("../escape")


# -- database transitions ----------------------------------------------------


def test_guarded_transition_rejects_a_wrong_source_state() -> None:
    database = Database.in_memory()
    now = time.time()
    database.insert_job("j1", "s1", state="eligible", consent_id="c1", expires_at=now + 60, now=now)
    database.transition_job("j1", allowed_from=["eligible"], to_state="uploading", now=now)
    with pytest.raises(StateTransitionConflict):
        database.transition_job("j1", allowed_from=["eligible"], to_state="uploading", now=now)


def test_transition_updates_extra_fields() -> None:
    database = Database.in_memory()
    now = time.time()
    database.insert_job("j1", "s1", state="building", consent_id="c1", expires_at=now + 60, now=now)
    row = database.transition_job(
        "j1", allowed_from=["building"], to_state="eligible", now=now, sample_id="sample-1"
    )
    assert row["sample_id"] == "sample-1"


def test_one_job_per_session() -> None:
    import sqlite3

    database = Database.in_memory()
    now = time.time()
    database.insert_job("j1", "s1", state="local_pending", consent_id="c1", expires_at=now, now=now)
    with pytest.raises(sqlite3.IntegrityError):
        database.insert_job(
            "j2", "s1", state="local_pending", consent_id="c1", expires_at=now, now=now
        )


def test_volume_accumulates_per_day() -> None:
    database = Database.in_memory()
    database.add_volume("2026-09-19", packages=1, seconds=30)
    database.add_volume("2026-09-19", packages=2, seconds=10)
    assert database.volume("2026-09-19") == (3, 40)
    assert database.volume("2026-09-20") == (0, 0)


# -- retention ---------------------------------------------------------------


def test_retention_expiry_matches_the_plan_table() -> None:
    policy = RetentionPolicy(RetentionSettings())
    now = 1_000_000.0
    raw = policy.expiry_for(ArtifactKind.RAW_AUDIO, now=now)
    package = policy.expiry_for(ArtifactKind.PACKAGE, now=now)
    assert raw - now == pytest.approx(24 * 3600)
    assert package - now == pytest.approx(7 * 24 * 3600)


def test_cleanup_deletes_expired_artifacts(store: SessionStore) -> None:
    now = time.time()
    store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=now - 1, now=now - 10)
    fresh = store.put("s2", ArtifactKind.RAW_AUDIO, b"audio", expires_at=now + 600, now=now)
    report = run_cleanup(store, store.db, now=now)
    assert report.artifacts_deleted == 1
    assert store.get(fresh.artifact_id) == b"audio"


def test_cleanup_deletes_expired_sessions(store: SessionStore) -> None:
    now = time.time()
    store.db.insert_session("s1", started_at=now - 100, expires_at=now - 1)
    store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=now + 600, now=now)
    report = run_cleanup(store, store.db, now=now)
    assert report.sessions_deleted == 1
    assert store.db.session("s1") is None


def test_orphan_files_are_swept(store: SessionStore) -> None:
    directory = store.paths.session_dir("s-orphan")
    directory.mkdir(parents=True, exist_ok=True)
    orphan = directory / "leftover.enc"
    orphan.write_bytes(b"ciphertext")
    assert sweep_orphans(store, store.db) == 1
    assert not orphan.exists()


def test_post_decision_cleanup_keeps_only_the_package(store: SessionStore) -> None:
    now = time.time()
    store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=now + 600, now=now)
    store.put("s1", ArtifactKind.RAW_TRANSCRIPT, b"text", expires_at=now + 600, now=now)
    package = store.put("s1", ArtifactKind.PACKAGE, b"zip", expires_at=now + 600, now=now)
    deleted = delete_after_decision(store, "s1", keep_package=True)
    assert deleted == 2
    assert store.get(package.artifact_id) == b"zip"


def test_post_decision_cleanup_on_rejection_keeps_nothing(store: SessionStore) -> None:
    now = time.time()
    store.put("s1", ArtifactKind.RAW_AUDIO, b"audio", expires_at=now + 600, now=now)
    store.put("s1", ArtifactKind.PACKAGE, b"zip", expires_at=now + 600, now=now)
    deleted = delete_after_decision(store, "s1", keep_package=False)
    assert deleted == 2
    assert store.db.artifacts_of("s1") == []
