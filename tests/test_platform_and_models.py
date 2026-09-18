"""Platform capability reporting, insertion policy and the model registry."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from dictation.config import InsertionSettings
from dictation.errors import ModelNotConfigured, ModelNotInstalled, ModelVerificationError
from dictation.models.fetch import fetch_model, sha256_file, verify_installed
from dictation.models.manifest import ModelManifest
from dictation.models.registry import CANDIDATES, ModelRegistry, ModelRole, ModelSpec
from dictation.platform_.base import Capability, FocusTarget, Support
from dictation.platform_.insertion import DeliveryMethod, DeliveryReason, ResultPanel, TextDelivery
from dictation.platform_.posix import NullAdapter, WaylandAdapter, X11Adapter
from dictation.platform_.probe import (
    adapter_name_for_session,
    probe_all,
    select_adapter,
    summarize,
    write_matrix,
)
from dictation.platform_.windows import WindowsAdapter

# -- capability reporting ----------------------------------------------------


def test_every_adapter_reports_every_capability() -> None:
    for name, report in probe_all().items():
        for capability in Capability:
            status = report.status(capability)
            assert status.support in set(Support), f"{name}/{capability}"
            assert status.note or status.support is Support.AVAILABLE


def test_unprobed_capability_is_unknown_not_available() -> None:
    report = NullAdapter().probe()
    assert not report.usable(Capability.TEXT_INSERTION_NATIVE)


def test_wayland_does_not_claim_focus_tracking() -> None:
    report = WaylandAdapter().probe()
    assert report.status(Capability.FOCUS_TARGET_TRACKING).support is Support.UNAVAILABLE


def test_windows_does_not_claim_press_release_shortcuts() -> None:
    report = WindowsAdapter().probe()
    assert report.status(Capability.GLOBAL_SHORTCUT_PRESS_RELEASE).support is Support.UNKNOWN


def test_no_adapter_claims_complete_password_field_detection() -> None:
    for report in probe_all().values():
        assert report.status(Capability.PASSWORD_FIELD_DETECTION).support is not Support.AVAILABLE


def test_x11_report_documents_its_security_context() -> None:
    report = X11Adapter().probe()
    assert any("inject" in note for note in report.notes)


def test_matrix_is_written_as_json_and_markdown(tmp_path) -> None:
    report = select_adapter().probe()
    json_path, markdown_path = write_matrix(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload["capabilities"]) == {str(capability) for capability in Capability}
    text = markdown_path.read_text(encoding="utf-8")
    assert "Unknown means not determined" in text
    assert summarize(report)


def test_adapter_selection_matches_the_session_type() -> None:
    assert adapter_name_for_session("wayland") in {"windows", "macos", "linux-wayland"}
    assert adapter_name_for_session("headless") in {"windows", "macos", "null"}


# -- insertion policy --------------------------------------------------------


class RecordingAdapter:
    """Adapter stub that reports a focus target and records insertions."""

    def __init__(self, target: FocusTarget, *, insert_ok: bool = True) -> None:
        self.target = target
        self.insert_ok = insert_ok
        self.inserted: list[str] = []
        self.clipboard: str | None = "previous clipboard"
        self.clipboard_writes: list[str] = []

    @property
    def name(self) -> str:
        return "recording"

    def probe(self):  # pragma: no cover - not used here
        return NullAdapter().probe()

    def current_focus(self) -> FocusTarget:
        return self.target

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        if not self.insert_ok:
            return False
        self.inserted.append(text)
        return True

    def copy_to_clipboard(self, text: str) -> bool:
        self.clipboard = text
        self.clipboard_writes.append(text)
        return True

    def read_clipboard(self) -> str | None:
        return self.clipboard

    def microphone_permission(self) -> Support:
        return Support.UNKNOWN


def editor() -> FocusTarget:
    return FocusTarget(handle="win-1", application="editor", editable=True, password_field=False)


def test_text_is_inserted_into_the_original_target() -> None:
    target = editor()
    adapter = RecordingAdapter(target)
    delivery = TextDelivery(adapter)
    result = delivery.deliver("the full dictation", target, session_id="s1")
    assert result.method is DeliveryMethod.NATIVE
    assert adapter.inserted == ["the full dictation"]


def test_focus_change_holds_the_text_in_the_result_panel() -> None:
    adapter = RecordingAdapter(FocusTarget(handle="win-2", application="terminal", editable=True))
    delivery = TextDelivery(adapter)
    result = delivery.deliver("secret plan", editor(), session_id="s1")
    assert result.method is DeliveryMethod.RESULT_PANEL
    assert result.reason is DeliveryReason.FOCUS_CHANGED
    assert adapter.inserted == []
    assert delivery.panel.take("s1") == "secret plan"


def test_unknown_focus_never_matches() -> None:
    adapter = RecordingAdapter(FocusTarget())
    delivery = TextDelivery(adapter)
    result = delivery.deliver("text", FocusTarget(), session_id="s1")
    assert result.method is DeliveryMethod.RESULT_PANEL


def test_password_field_gets_nothing() -> None:
    target = FocusTarget(handle="win-1", application="browser", editable=True, password_field=True)
    adapter = RecordingAdapter(target)
    delivery = TextDelivery(adapter)
    result = delivery.deliver("text", target, session_id="s1")
    assert result.reason is DeliveryReason.PASSWORD_FIELD
    assert adapter.inserted == []


def test_excluded_application_gets_nothing() -> None:
    target = editor()
    adapter = RecordingAdapter(target)
    delivery = TextDelivery(adapter, InsertionSettings(excluded_apps=("editor",)))
    result = delivery.deliver("text", target, session_id="s1")
    assert result.reason is DeliveryReason.APP_EXCLUDED


def test_cancelled_session_never_inserts() -> None:
    target = editor()
    adapter = RecordingAdapter(target)
    delivery = TextDelivery(adapter)
    result = delivery.deliver("text", target, session_id="s1", cancelled=True)
    assert result.method is DeliveryMethod.NONE
    assert result.reason is DeliveryReason.CANCELLED
    assert adapter.inserted == []
    assert delivery.panel.entries == []


def test_clipboard_fallback_is_off_unless_disclosed() -> None:
    target = editor()
    adapter = RecordingAdapter(target, insert_ok=False)
    delivery = TextDelivery(adapter)
    result = delivery.deliver("text", target, session_id="s1")
    assert result.reason is DeliveryReason.CLIPBOARD_DISABLED
    assert adapter.clipboard_writes == []


def test_clipboard_fallback_restores_previous_contents() -> None:
    target = editor()
    adapter = RecordingAdapter(target, insert_ok=False)
    delivery = TextDelivery(
        adapter, InsertionSettings(clipboard_fallback_allowed=True, restore_clipboard=True)
    )
    result = delivery.deliver("dictated text", target, session_id="s1")
    assert result.method is DeliveryMethod.CLIPBOARD
    assert adapter.clipboard_writes == ["dictated text", "previous clipboard"]


def test_clipboard_is_not_restored_over_a_newer_copy() -> None:
    target = editor()

    class Clobbering(RecordingAdapter):
        def read_clipboard(self) -> str | None:
            # Something else copied after us.
            return "user copied this later"

    adapter = Clobbering(target, insert_ok=False)
    delivery = TextDelivery(adapter, InsertionSettings(clipboard_fallback_allowed=True))
    delivery.deliver("dictated text", target, session_id="s1")
    assert adapter.clipboard_writes == ["dictated text"]


def test_empty_transcript_is_not_delivered() -> None:
    target = editor()
    adapter = RecordingAdapter(target)
    result = TextDelivery(adapter).deliver("   ", target, session_id="s1")
    assert result.reason is DeliveryReason.EMPTY


def test_result_panel_is_bounded() -> None:
    panel = ResultPanel(limit=2)
    for index in range(4):
        panel.hold(f"s{index}", f"text {index}")
    assert len(panel.entries) == 2
    assert panel.take() == "text 3"


def test_enter_is_never_synthesised_by_default() -> None:
    assert InsertionSettings().submit_with_enter is False


# -- model registry ----------------------------------------------------------


def test_shipped_candidates_are_unresolved() -> None:
    assert CANDIDATES
    for spec in CANDIDATES:
        assert not spec.is_resolved
        assert "source_url" in spec.unresolved_fields()


def test_exactly_two_learned_roles_exist() -> None:
    assert {role for role in ModelRole} == {ModelRole.ASR, ModelRole.PRIVACY}


def test_choosing_an_unresolved_candidate_then_fetching_refuses(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    registry.choose(ModelRole.ASR, "whisper-base-q5_1")
    spec = registry.chosen(ModelRole.ASR)
    with pytest.raises(ModelNotConfigured):
        fetch_model(registry, spec)


def test_no_model_chosen_is_an_actionable_refusal(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    with pytest.raises(ModelNotConfigured, match="no model chosen"):
        registry.chosen(ModelRole.PRIVACY)


def test_choosing_a_model_for_the_wrong_role_is_refused(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    with pytest.raises(ModelNotConfigured):
        registry.choose(ModelRole.ASR, "qwen3-4b-instruct-2507-q4")


def test_resolution_requires_https(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    with pytest.raises(ModelVerificationError, match="HTTPS"):
        registry.resolve(
            "whisper-base-q5_1",
            source_url="http://example.com/model.bin",
            sha256="a" * 64,
            size_bytes=10,
            license_id="MIT",
            revision="r1",
        )


def test_resolution_requires_a_weight_file_extension(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    with pytest.raises(ModelVerificationError, match="refusing"):
        registry.resolve(
            "whisper-base-q5_1",
            source_url="https://example.com/model.zip",
            sha256="a" * 64,
            size_bytes=10,
            license_id="MIT",
            revision="r1",
            filename="model.zip",
        )


def test_resolution_requires_a_plausible_digest(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    with pytest.raises(ModelVerificationError, match="sha256"):
        registry.resolve(
            "whisper-base-q5_1",
            source_url="https://example.com/model.bin",
            sha256="short",
            size_bytes=10,
            license_id="MIT",
            revision="r1",
        )


def resolved_spec(payload: bytes, *, digest: str | None = None) -> ModelSpec:
    return ModelSpec(
        identifier="test-asr",
        role=ModelRole.ASR,
        source_url="https://example.invalid/test-asr.bin",
        sha256=digest or hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="TEST-1.0",
        revision="rev-1",
        filename="test-asr.bin",
    )


def fake_opener(payload: bytes):
    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def opener(request: object) -> _Response:
        return _Response(payload)

    return opener


def test_fetch_verifies_and_installs(tmp_path) -> None:
    payload = b"weights" * 100
    registry = ModelRegistry.load(tmp_path)
    spec = registry.upsert(resolved_spec(payload))
    result = fetch_model(registry, spec, opener=fake_opener(payload))
    assert result.verified
    assert result.path.exists()
    assert sha256_file(result.path) == spec.sha256
    assert verify_installed(registry, spec)


def test_fetch_refuses_a_digest_mismatch_and_leaves_nothing(tmp_path) -> None:
    payload = b"weights" * 100
    registry = ModelRegistry.load(tmp_path)
    spec = registry.upsert(resolved_spec(payload, digest="b" * 64))
    with pytest.raises(ModelVerificationError):
        fetch_model(registry, spec, opener=fake_opener(payload))
    assert not registry.path_of(spec).exists()
    assert not registry.path_of(spec).with_suffix(".bin.part").exists()


def test_fetch_refuses_a_size_mismatch(tmp_path) -> None:
    payload = b"weights" * 100
    registry = ModelRegistry.load(tmp_path)
    spec = registry.upsert(
        ModelSpec(
            identifier="test-asr",
            role=ModelRole.ASR,
            source_url="https://example.invalid/test-asr.bin",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload) + 10,
            license_id="TEST-1.0",
            revision="rev-1",
            filename="test-asr.bin",
        )
    )
    with pytest.raises(ModelVerificationError):
        fetch_model(registry, spec, opener=fake_opener(payload))


def test_fetch_refuses_an_oversized_download(tmp_path) -> None:
    payload = b"x" * 5000
    registry = ModelRegistry.load(tmp_path)
    spec = registry.upsert(
        ModelSpec(
            identifier="test-asr",
            role=ModelRole.ASR,
            source_url="https://example.invalid/test-asr.bin",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=100,
            license_id="TEST-1.0",
            revision="rev-1",
            filename="test-asr.bin",
        )
    )
    with pytest.raises(ModelVerificationError):
        fetch_model(registry, spec, opener=fake_opener(payload))


def test_installed_model_is_recorded_in_the_manifest(tmp_path) -> None:
    payload = b"weights" * 100
    registry = ModelRegistry.load(tmp_path)
    spec = registry.upsert(resolved_spec(payload))
    fetch_model(registry, spec, opener=fake_opener(payload))
    manifest = ModelManifest.load(registry.directory)
    entry = manifest.get("test-asr")
    assert entry is not None
    assert entry.license_id == "TEST-1.0"
    assert entry.sha256 == spec.sha256


def test_tampered_artifact_fails_verification(tmp_path) -> None:
    payload = b"weights" * 100
    registry = ModelRegistry.load(tmp_path)
    spec = registry.upsert(resolved_spec(payload))
    result = fetch_model(registry, spec, opener=fake_opener(payload))
    result.path.write_bytes(payload + b"tampered")
    assert not verify_installed(registry, spec)


def test_missing_model_reports_how_to_install_it(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    spec = registry.get("whisper-base-q5_1")
    with pytest.raises(ModelNotInstalled, match="models fetch"):
        registry.installed_path(spec)


def test_registry_round_trips_choices_and_specs(tmp_path) -> None:
    registry = ModelRegistry.load(tmp_path)
    payload = b"weights"
    registry.upsert(resolved_spec(payload))
    registry.choose(ModelRole.ASR, "test-asr")
    reopened = ModelRegistry.load(tmp_path)
    assert reopened.chosen(ModelRole.ASR).identifier == "test-asr"
    assert reopened.get("test-asr").license_id == "TEST-1.0"
