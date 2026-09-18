"""Command line interface.

The commands follow the plan's milestones, in the order it asks for them:

    dictation probe            platform capability matrix      (milestone 1)
    dictation bench            model and pipeline benchmarks   (milestone 1)
    dictation models ...       choose, resolve and fetch models
    dictation dictate          one dictation session           (milestone 2)
    dictation api ...          local integration API           (milestone 2)
    dictation dataset          build a training copy offline   (milestone 3)
    dictation consent ...      opt in, pause, withdraw, delete  (milestone 4)
    dictation queue ...        inspect and drain the queue      (milestone 4)
    dictation doctor           what works and what is missing

``doctor`` is the honest summary the plan asks each handoff to state: what
works, the tested platform matrix, measured model performance, and which upload
gates remain unmet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dictation.app import Application
from dictation.capture.devices import FileSource, MemorySource, MicrophoneSource, synthetic_speech
from dictation.consent.consent import Purpose
from dictation.dataset.builder import build_dataset, make_recheck
from dictation.dataset.package import PackageWriter, build_package
from dictation.errors import DictationError
from dictation.logging_ import configure
from dictation.models.fetch import fetch_model, verify_installed
from dictation.models.registry import ModelRole, describe
from dictation.platform_.probe import probe_all, select_adapter, summarize, write_matrix
from dictation.privacy import rules
from dictation.privacy.pipeline import analyze
from dictation.types import CANONICAL_SAMPLE_RATE, Decision
from dictation.version import (
    APP_VERSION,
    CONSENT_VERSION,
    POLICY_VERSION,
    UPLOAD_GATES_MET,
    VALIDATED_UPLOAD_LANGUAGES,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    configure(logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except DictationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dictation", description=__doc__)
    parser.add_argument("--version", action="version", version=APP_VERSION)
    parser.add_argument("--verbose", action="store_true", help="log every event")
    parser.add_argument("--data-root", type=Path, help="override the payload directory")
    sub = parser.add_subparsers(dest="command")

    probe = sub.add_parser("probe", help="probe platform capabilities (milestone 1)")
    probe.add_argument("--all", action="store_true", help="probe every adapter, not just this one")
    probe.add_argument("--write", action="store_true", help="write the capability matrix files")
    probe.set_defaults(handler=_cmd_probe)

    bench = sub.add_parser("bench", help="benchmark the two model roles (milestone 1)")
    bench.add_argument("--write", type=Path, help="write the JSON report here")
    bench.set_defaults(handler=_cmd_bench)

    models = sub.add_parser("models", help="model registry: list, resolve, choose, fetch")
    models_sub = models.add_subparsers(dest="models_command")

    models_list = models_sub.add_parser("list", help="list candidates and their status")
    models_list.set_defaults(handler=_cmd_models_list)

    models_resolve = models_sub.add_parser(
        "resolve", help="record the URL, digest, size, licence and revision for a candidate"
    )
    models_resolve.add_argument("identifier")
    models_resolve.add_argument("--url", required=True)
    models_resolve.add_argument("--sha256", required=True)
    models_resolve.add_argument("--size", required=True, type=int)
    models_resolve.add_argument("--license", required=True)
    models_resolve.add_argument("--revision", required=True)
    models_resolve.add_argument("--license-url", default="")
    models_resolve.add_argument("--provenance", default="")
    models_resolve.set_defaults(handler=_cmd_models_resolve)

    models_choose = models_sub.add_parser("choose", help="select the model for a role")
    models_choose.add_argument("identifier")
    models_choose.add_argument("--role", required=True, choices=[str(role) for role in ModelRole])
    models_choose.set_defaults(handler=_cmd_models_choose)

    models_fetch = models_sub.add_parser("fetch", help="download and verify a chosen model")
    models_fetch.add_argument("--role", required=True, choices=[str(role) for role in ModelRole])
    models_fetch.add_argument("--force", action="store_true")
    models_fetch.set_defaults(handler=_cmd_models_fetch)

    models_verify = models_sub.add_parser("verify", help="re-verify installed artifacts")
    models_verify.set_defaults(handler=_cmd_models_verify)

    dictate = sub.add_parser("dictate", help="run one dictation session (milestone 2)")
    dictate.add_argument("--wav", type=Path, help="16 kHz mono WAV to dictate instead of the mic")
    dictate.add_argument("--synthetic", action="store_true", help="use synthetic tone audio")
    dictate.add_argument("--stub-models", action="store_true", help="allow scripted stand-ins")
    dictate.add_argument("--no-contribution", action="store_true", help="never build a training copy")
    dictate.set_defaults(handler=_cmd_dictate)

    dataset = sub.add_parser("dataset", help="build a training copy from a transcript (milestone 3)")
    dataset.add_argument("--text", required=True, help="transcript text to treat as dictated")
    dataset.add_argument("--out", type=Path, help="directory for the retained clips")
    dataset.add_argument("--private-term", action="append", default=[])
    dataset.set_defaults(handler=_cmd_dataset)

    detect = sub.add_parser("detect", help="show what the deterministic rules find in text")
    detect.add_argument("--text", required=True)
    detect.add_argument("--private-term", action="append", default=[])
    detect.set_defaults(handler=_cmd_detect)

    api = sub.add_parser("api", help="local integration API (milestone 2)")
    api_sub = api.add_subparsers(dest="api_command")
    api_pair = api_sub.add_parser("pair", help="issue a scoped token for a client")
    api_pair.add_argument("name")
    api_pair.add_argument("--scope", action="append", default=["status:read"])
    api_pair.set_defaults(handler=_cmd_api_pair)
    api_tokens = api_sub.add_parser("tokens", help="list issued tokens")
    api_tokens.set_defaults(handler=_cmd_api_tokens)
    api_revoke = api_sub.add_parser("revoke", help="revoke a token")
    api_revoke.add_argument("token_id")
    api_revoke.set_defaults(handler=_cmd_api_revoke)

    consent = sub.add_parser("consent", help="contribution consent (milestone 4)")
    consent_sub = consent.add_subparsers(dest="consent_command")
    consent_show = consent_sub.add_parser("show", help="show the disclosure and current state")
    consent_show.set_defaults(handler=_cmd_consent_show)
    consent_grant = consent_sub.add_parser("grant", help="opt in to contribution")
    consent_grant.set_defaults(handler=_cmd_consent_grant)
    consent_pause = consent_sub.add_parser("pause", help="pause contribution")
    consent_pause.set_defaults(handler=_cmd_consent_pause)
    consent_withdraw = consent_sub.add_parser("withdraw", help="withdraw and request deletion")
    consent_withdraw.set_defaults(handler=_cmd_consent_withdraw)

    queue = sub.add_parser("queue", help="inspect the upload queue (milestone 4)")
    queue.set_defaults(handler=_cmd_queue)

    doctor = sub.add_parser("doctor", help="what works, what is missing, which gates are unmet")
    doctor.set_defaults(handler=_cmd_doctor)

    return parser


# -- commands ----------------------------------------------------------------


def _cmd_probe(args: argparse.Namespace) -> int:
    if args.all:
        for name, report in probe_all().items():
            print(f"== {name} ==")
            print(summarize(report))
            print()
        return 0
    adapter = select_adapter()
    report = adapter.probe()
    print(summarize(report))
    if args.write:
        with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
            json_path, markdown_path = write_matrix(report, app.paths.metadata)
        print(f"\nwrote {json_path}\nwrote {markdown_path}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from dictation.bench.harness import run_benchmarks

    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        asr = app.build_asr(allow_stub=True)
        classifier = app.build_classifier(allow_stub=True)
        transcript = _scripted_transcript(
            "The meeting starts tomorrow and I will bring the printed agenda. "
            "Remind me to water the plants before we leave the office."
        )
        report = run_benchmarks(asr, classifier, transcript, settings=app.settings)
        print(report.summary())
        target = args.write or (app.paths.metadata / "benchmark.json")
        report.write(target)
        print(f"\nwrote {target}")
        for warning in app.warnings:
            print(f"warning: {warning}")
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        for role in ModelRole:
            print(f"{role}:")
            print(describe(app.registry.for_role(role)) or "  (none)")
            try:
                chosen = app.registry.chosen(role)
                installed = "installed" if app.registry.is_installed(chosen) else "not installed"
                print(f"  chosen: {chosen.identifier} ({installed})")
            except DictationError as error:
                print(f"  chosen: none - {error}")
            print()
    return 0


def _cmd_models_resolve(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        spec = app.registry.resolve(
            args.identifier,
            source_url=args.url,
            sha256=args.sha256.lower(),
            size_bytes=args.size,
            license_id=args.license,
            license_url=args.license_url,
            revision=args.revision,
            conversion_provenance=args.provenance,
        )
        print(f"resolved {spec.identifier}: {spec.source_url}")
    return 0


def _cmd_models_choose(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        spec = app.registry.choose(ModelRole(args.role), args.identifier)
        print(f"{args.role} role uses {spec.identifier}")
        if not spec.is_resolved:
            print(
                "warning: this candidate is unresolved "
                f"({', '.join(spec.unresolved_fields())}); fetching will refuse"
            )
    return 0


def _cmd_models_fetch(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        spec = app.registry.chosen(ModelRole(args.role))

        def progress(written: int, total: int) -> None:
            if total:
                print(f"\r{spec.identifier}: {written * 100 // total}%", end="", flush=True)

        result = fetch_model(app.registry, spec, progress=progress, force=args.force)
        print(f"\n{'verified' if result.verified else 'unverified'}: {result.path}")
    return 0


def _cmd_models_verify(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        any_model = False
        for spec in app.registry.specs.values():
            if not app.registry.is_installed(spec):
                continue
            any_model = True
            ok = verify_installed(app.registry, spec)
            print(f"{spec.identifier}: {'ok' if ok else 'DIGEST MISMATCH'}")
        if not any_model:
            print("no models are installed")
    return 0


def _cmd_dictate(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        coordinator = app.build_coordinator(
            allow_stub=args.stub_models, contribution=not args.no_contribution
        )
        for warning in app.warnings:
            print(f"warning: {warning}")

        if args.wav:
            source = FileSource(args.wav)
        elif args.synthetic:
            source = MemorySource(
                synthetic_speech([(0.2, 2.2), (2.6, 4.4)], total_seconds=5.0)
            )
        elif MicrophoneSource.available():
            source = MicrophoneSource()
        else:
            print(
                "no capture backend is wired up. Pass --wav or --synthetic, or implement "
                "MicrophoneSource for this platform.",
                file=sys.stderr,
            )
            return 1

        outcome = coordinator.run_session(source)
        if outcome.error:
            print(f"session failed: {outcome.error}", file=sys.stderr)
            return 1
        print(f"session: {outcome.session_id}")
        print(f"transcript: {outcome.text}")
        if outcome.delivery is not None:
            print(f"delivery: {outcome.delivery.method} ({outcome.delivery.reason})")
        print(f"training copy: {outcome.training_reason or 'not built'}")
        if outcome.training is not None and outcome.training.decision is Decision.ELIGIBLE:
            print(
                f"  retained {outcome.training.metrics.clip_count} clips, "
                f"{outcome.training.metrics.duration_ms} ms; removed "
                f"{outcome.training.metrics.removed_duration_ms} ms"
            )
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from dictation.privacy.classifier.mock import RuleEchoClassifier

    transcript, pcm = _scripted_session(args.text)
    terms = tuple(args.private_term)
    classifier = RuleEchoClassifier(private_terms=terms)
    analysis = analyze(transcript, classifier, private_terms=terms)
    result = build_dataset(
        analysis,
        pcm,
        recheck=make_recheck(lambda t: (not rules.detect(t, private_terms=terms).spans, "sensitive_after_removal")),
    )
    print(f"decision: {result.decision}{'' if result.eligible else f' ({result.reason})'}")
    if not result.eligible:
        return 0
    print(f"retained {result.metrics.clip_count} clips / {result.metrics.duration_ms} ms")
    print(f"removed  {result.metrics.removed_interval_count} intervals / "
          f"{result.metrics.removed_duration_ms} ms")
    for clip in result.clips:
        print(f"  clip {clip.clip_index}: {clip.duration_ms} ms  {clip.text!r}")
    if args.out:
        writer = PackageWriter(args.out)
        writer.write_clips(result)
        package = build_package(
            result, consent_version=CONSENT_VERSION, consent_reference="offline-export"
        )
        path = writer.write_plaintext_export(package)
        print(f"wrote {len(writer.written)} files, package {path}")
        print(
            "note: this export is plaintext for evaluation. The application stores "
            "packages encrypted; delete these files when you are done."
        )
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    transcript, _ = _scripted_session(args.text)
    findings = rules.detect(transcript, private_terms=tuple(args.private_term))
    print(f"detectors fired: {', '.join(findings.detectors) or 'none'}")
    print(f"injection suspected: {findings.injection_suspected}")
    for span in findings.spans:
        words = transcript.slice_by_ids(span.start_word_id, span.end_word_id_exclusive)
        print(
            f"  [{span.start_word_id}, {span.end_word_id_exclusive}) {span.category} "
            f"via {span.detector}: {' '.join(word.text for word in words)!r}"
        )
    return 0


def _cmd_api_pair(args: argparse.Namespace) -> int:
    from dictation.api.tokens import parse_scopes

    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        scopes = parse_scopes(",".join(args.scope))
        record, token = app.tokens.issue(args.name, scopes)
        print(f"token id: {record.token_id}")
        print(f"scopes:   {', '.join(sorted(str(scope) for scope in record.scopes))}")
        print(f"token:    {token}")
        print("This token is shown once. Store it in the client, not in a shell history.")
    return 0


def _cmd_api_tokens(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        for record in app.tokens.tokens():
            state = "revoked" if record.revoked_at else "active"
            print(
                f"{record.token_id}  {state:<8} {record.name:<20} "
                f"{', '.join(sorted(str(scope) for scope in record.scopes))}"
            )
    return 0


def _cmd_api_revoke(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        print("revoked" if app.tokens.revoke(args.token_id) else "no such active token")
    return 0


def _cmd_consent_show(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        print(json.dumps(app.consent.disclosure(), indent=2, sort_keys=True))
        record = app.consent.active()
        print("\ncurrent:", json.dumps(record.summary(), indent=2) if record else "no active consent")
    return 0


def _cmd_consent_grant(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        record = app.consent.grant((Purpose.CUSTOMER_PERSONALIZATION,))
        print(f"granted {record.consent_id} under {record.version}")
        if not UPLOAD_GATES_MET:
            print(
                "note: the evaluation gates in plan section 12 are not recorded as met, "
                "so eligible sessions will still not upload."
            )
    return 0


def _cmd_consent_pause(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        record = app.consent.pause()
        cancelled = app.queue.on_withdrawal()
        print(f"paused {record.consent_id}; cancelled {len(cancelled)} queued jobs")
    return 0


def _cmd_consent_withdraw(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        request = app.consent.withdraw()
        cancelled = app.queue.on_withdrawal()
        print(f"withdrawn; deletion request {request}; cancelled {len(cancelled)} jobs")
        print("Server-side deletion must be confirmed by the service before this is complete.")
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        jobs = app.queue.jobs()
        if not jobs:
            print("the queue is empty")
        for job in jobs:
            print(
                f"{job.job_id}  {job.state:<14} attempts={job.attempts} "
                f"{job.duration_ms}ms {job.reason}"
            )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    adapter = select_adapter()
    report = adapter.probe()
    with Application.open(args.data_root, allow_ephemeral_keys=True) as app:
        print(f"version:        {APP_VERSION}")
        print(f"policy:         {POLICY_VERSION}")
        print(f"data root:      {app.paths.root}")
        print(f"secure storage: {'yes' if app.secure_storage else 'NO - contribution disabled'}")
        print(f"platform:       {report.adapter} ({report.session_type})")
        missing = ", ".join(str(capability) for capability in report.missing())
        print(f"capabilities:   {missing or 'all available'} not available")
        print(f"capture:        {'backend found' if MicrophoneSource.available() else 'NO microphone backend'}")
        for role in ModelRole:
            try:
                spec = app.registry.chosen(role)
                state = "installed" if app.registry.is_installed(spec) else "chosen, not installed"
                print(f"{str(role) + ' model:':<15} {spec.identifier} ({state})")
            except DictationError:
                print(f"{str(role) + ' model:':<15} not chosen")
        consent_record = app.consent.active()
        print(f"consent:        {consent_record.consent_id if consent_record else 'none'}")
        print(f"upload gates:   {'met' if UPLOAD_GATES_MET else 'NOT MET - automatic upload refused'}")
        print(f"languages:      {', '.join(sorted(VALIDATED_UPLOAD_LANGUAGES))}")
        print(f"queue:          {len(app.queue.jobs())} jobs")
        for warning in app.warnings:
            print(f"warning:        {warning}")
    return 0


# -- helpers -----------------------------------------------------------------


def _scripted_transcript(text: str):
    transcript, _ = _scripted_session(text)
    return transcript


def _scripted_session(text: str):
    """Build a deterministic transcript and matching audio from plain text.

    Used by the offline dataset and detection commands so that the privacy
    pipeline can be exercised without a recogniser.  The audio is tone bursts,
    not speech.
    """
    from dictation.asr.mock import ScriptedAsr, script_from_text

    script = script_from_text(text)
    total_ms = (script[-1].end_ms if script else 0) + 400
    pcm = synthetic_speech(
        [(word.start_ms / 1000, word.end_ms / 1000) for word in script],
        total_seconds=total_ms / 1000,
    )
    asr = ScriptedAsr(script)
    asr.warm_up()
    return asr.transcribe(pcm, session_id="cli", sample_rate=CANONICAL_SAMPLE_RATE), pcm


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
