"""The CLI commands, one per milestone, run end to end.

Every test runs with ``DICTATION_EPHEMERAL_KEYS=1`` and a temporary payload
root, so no test touches the developer's real credential store or application
data directory.
"""

from __future__ import annotations

import json

import pytest

from dictation.cli import main


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DICTATION_EPHEMERAL_KEYS", "1")
    monkeypatch.setenv("DICTATION_DATA_ROOT", str(tmp_path / "data"))
    return tmp_path


def run(*args: str) -> int:
    return main(list(args))


def test_probe_prints_a_capability_matrix(capsys) -> None:
    assert run("probe") == 0
    output = capsys.readouterr().out
    assert "adapter:" in output
    assert "text_insertion_native" in output


def test_probe_writes_the_matrix_files(capsys, isolated_environment) -> None:
    assert run("probe", "--write") == 0
    written = list((isolated_environment / "data" / "metadata").glob("capability-matrix.*"))
    assert {path.suffix for path in written} == {".json", ".md"}
    payload = json.loads(
        (isolated_environment / "data" / "metadata" / "capability-matrix.json").read_text()
    )
    assert payload["capabilities"]


def test_probe_all_covers_every_adapter(capsys) -> None:
    assert run("probe", "--all") == 0
    output = capsys.readouterr().out
    for name in ("windows", "macos", "linux-x11", "linux-wayland", "null"):
        assert f"== {name} ==" in output


def test_doctor_reports_the_unmet_upload_gates(capsys) -> None:
    assert run("doctor") == 0
    output = capsys.readouterr().out
    assert "upload gates:   NOT MET" in output
    assert "not chosen" in output  # neither model role is resolved


def test_models_list_shows_unresolved_candidates(capsys) -> None:
    assert run("models", "list") == 0
    output = capsys.readouterr().out
    assert "whisper-base-q5_1" in output
    assert "unresolved" in output
    assert "qwen3-4b-instruct-2507-q4" in output


def test_choosing_an_unresolved_candidate_warns(capsys) -> None:
    assert run("models", "choose", "--role", "asr", "whisper-base-q5_1") == 0
    output = capsys.readouterr().out
    assert "warning" in output
    assert "unresolved" in output


def test_fetching_an_unresolved_candidate_fails_with_guidance(capsys) -> None:
    run("models", "choose", "--role", "asr", "whisper-base-q5_1")
    assert run("models", "fetch", "--role", "asr") == 1
    assert "unresolved candidate" in capsys.readouterr().err


def test_resolving_records_the_pinned_artifact(capsys) -> None:
    code = run(
        "models",
        "resolve",
        "whisper-base-q5_1",
        "--url",
        "https://example.invalid/whisper-base-q5_1.bin",
        "--sha256",
        "a" * 64,
        "--size",
        "1024",
        "--license",
        "MIT",
        "--revision",
        "rev-1",
    )
    assert code == 0
    assert "resolved whisper-base-q5_1" in capsys.readouterr().out
    assert run("models", "list") == 0
    listing = capsys.readouterr().out
    base_line = next(line for line in listing.splitlines() if "whisper-base-q5_1 " in line)
    assert "resolved" in base_line and "unresolved" not in base_line
    # The other candidate is untouched and still unresolved.
    small_line = next(line for line in listing.splitlines() if "whisper-small-q5_1 " in line)
    assert "unresolved" in small_line


def test_detect_shows_what_the_rules_find(capsys) -> None:
    assert run("detect", "--text", "My name is Jane Doe and my card is 4111 1111 1111 1111.") == 0
    output = capsys.readouterr().out
    assert "direct_identifier" in output
    assert "financial" in output


def test_detect_reports_nothing_for_clean_text(capsys) -> None:
    assert run("detect", "--text", "The meeting starts tomorrow and I will bring the agenda.") == 0
    output = capsys.readouterr().out
    assert "detectors fired: none" in output


def test_dataset_builds_a_filtered_training_copy(capsys, isolated_environment) -> None:
    out = isolated_environment / "clips"
    code = run(
        "dataset",
        "--text",
        "Send the parcel to Jane at 14 Oak Street. "
        "The meeting starts tomorrow and I will bring the printed agenda. "
        "Remind me to water the plants before we leave the office.",
        "--out",
        str(out),
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "decision: eligible" in output
    assert "jane" not in output.lower()
    written = sorted(path.name for path in out.iterdir())
    assert any(name.endswith(".wav") for name in written)
    assert any(name.endswith(".zip") for name in written)
    for path in out.glob("*.txt"):
        assert "Oak" not in path.read_text(encoding="utf-8")


def test_dataset_honours_a_private_term(capsys) -> None:
    code = run(
        "dataset",
        "--text",
        "The Northwind Foundry contract is signed. "
        "The meeting starts tomorrow and I will bring the printed agenda. "
        "Remind me to water the plants before we leave the office.",
        "--private-term",
        "Northwind Foundry",
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "northwind" not in output.lower()


def test_dictate_with_synthetic_audio_and_stub_models(capsys) -> None:
    assert run("dictate", "--synthetic", "--stub-models", "--no-contribution") == 0
    output = capsys.readouterr().out
    assert "transcript:" in output
    assert "scripted stand-in" in output or "scripted stand in" in output.lower()


def test_bench_writes_a_report(capsys, isolated_environment) -> None:
    assert run("bench") == 0
    output = capsys.readouterr().out
    assert "machine:" in output
    report = json.loads(
        (isolated_environment / "data" / "metadata" / "benchmark.json").read_text()
    )
    assert report["machine"]["cpu_count"] >= 1
    assert any("stand-in" in note for note in report["notes"])


def test_consent_show_then_grant_then_withdraw(capsys) -> None:
    assert run("consent", "show") == 0
    disclosure = capsys.readouterr().out
    assert "identifiable" in disclosure

    assert run("consent", "grant") == 0
    granted = capsys.readouterr().out
    assert "granted" in granted
    assert "gates" in granted  # the honest note that upload still will not run

    assert run("consent", "withdraw") == 0
    assert "withdrawn" in capsys.readouterr().out


def test_queue_is_empty_by_default(capsys) -> None:
    assert run("queue") == 0
    assert "queue is empty" in capsys.readouterr().out


def test_api_pairing_issues_a_token_once(capsys) -> None:
    assert run("api", "pair", "captions", "--scope", "transcript:live") == 0
    output = capsys.readouterr().out
    assert "token:" in output
    assert "shown once" in output

    assert run("api", "tokens") == 0
    listing = capsys.readouterr().out
    assert "captions" in listing
    token_id = listing.split()[0]

    assert run("api", "revoke", token_id) == 0
    assert "revoked" in capsys.readouterr().out


def test_unknown_command_prints_help() -> None:
    assert main([]) == 2
