# Local dictation with privacy-filtered personalization

A reference implementation of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md):
a dictation application that keeps recognition local, delivers the **full**
transcript to the user, and builds a separate, optional, privacy-filtered
training copy that only uploads when every gate passes.

This is the state of the work, stated the way the plan's section 13 asks for it.

## What works today

| Area | State |
| --- | --- |
| Recording state machine (hold, toggle, lock, cancel, duration limit) | implemented, 17 tests |
| Bounded capture buffers, non-neural VAD, silence guard | implemented |
| Streaming partial reconciliation (segment identity and revisions) | implemented |
| Insertion policy: focus target, exclusions, clipboard fallback, result panel | implemented |
| Deterministic privacy detectors, including spoken numbers and spelled addresses | implemented |
| Privacy-model contract: windowing, coverage, JSON schema, independent validation | implemented |
| Audio/text removal: sentence expansion, padding, merge, word pull-in, complement | implemented |
| Quality and privacy gates, duplicate detection, post-removal recheck | implemented |
| Upload package construction with an enforced field allowlist | implemented |
| Encrypted local store (AES-256-GCM), OS credential store, guarded transitions, retention | implemented |
| Consent, pause, withdrawal, deletion requests, no retroactive uploads | implemented |
| Upload queue, idempotent transport, retries, withdrawal races | implemented |
| Loopback HTTP/WebSocket API with pairing, scopes, Host/Origin checks | implemented |
| Platform capability probes for Windows, macOS, X11, Wayland | probes implemented; matrix published per machine |
| Windows text insertion (`SendInput` Unicode) and focus tracking | implemented, dry-run by default |
| Training server: tenant isolation, admission, lineage, deletion | skeleton, real gates |
| Personalized training and signed delivery | gates and signing implemented; **no trainer** |

Test suite: **398 tests**, `python -m pytest`.

## What is deliberately not done

- **No model is chosen.** The registry ships the plan's candidates - quantized
  Whisper base/small and a four-bit Qwen3-4B-Instruct-2507 - with no URL and no
  checksum. Every worker that needs one refuses until you record the artifact
  and fetch it. See [docs/MODELS.md](docs/MODELS.md).
- **No microphone backend.** Capture goes through an `AudioSource`; the file and
  synthetic sources work, and `MicrophoneSource` reports the missing backend
  rather than recording silence.
- **Automatic upload is refused.** `UPLOAD_GATES_MET` is `False` because no
  recall benchmark has been run. Opting in is possible; uploading is not, unless
  an operator explicitly points the client at an isolated non-production server.
- **No trainer.** The adaptation method has not been selected and the
  export → quantize → load path has not been demonstrated, so
  `UnavailableTrainer` refuses and says why.
- **No desktop shell.** This is the core plus a CLI, not the Tauri UI.

See [docs/DEVIATIONS.md](docs/DEVIATIONS.md) for the full list, including the
language change from the plan's Rust core.

## Install

```bash
python -m pip install -e ".[dev]"      # Python 3.12+
python -m pytest
```

Runtime dependencies are `cryptography` (payload encryption) and `keyring` (OS
credential store). `requests` is optional and only used by the HTTPS upload
transport.

## Try it

```bash
# Milestone 1: what can this machine actually do?
dictation probe --write
dictation bench

# What the deterministic detectors find, before any model is involved:
dictation detect --text "My name is Jane Doe and my card is 4111 1111 1111 1111."

# Milestone 3: build a privacy-filtered training copy from a transcript
dictation dataset --out ./clips --text \
  "Send the parcel to Jane at 14 Oak Street. \
   The meeting starts tomorrow and I will bring the printed agenda. \
   Remind me to water the plants before we leave the office."

# One dictation session end to end, with scripted stand-ins for the models
dictation dictate --synthetic --stub-models --no-contribution

# Honest status: what is missing and which gates are unmet
dictation doctor
```

On the example above, `dictation dataset` removes the first sentence in both
audio and text (3.6 s), and keeps the remaining contiguous audio as one 10 s
clip whose text is exactly the words spoken in it. The dictation delivered to
the user still contains the name and the address.

## Where things live

`src/dictation/` mirrors the plan's module table; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
maps each module to the plan section that specifies it.

```
capture/     recording state machine, buffers, VAD, devices, coordinator
asr/         worker interface, whisper.cpp backend, scripted backends
privacy/     rules, output schema, windowing, classifier backends, pipeline
dataset/     interval mathematics, quality gates, builder, upload package
store/       AEAD payloads, key management, SQLite state, retention
consent/     consent records, eligibility decisions, withdrawal
upload/       job states, queue, transport, worker
api/         tokens, events, loopback HTTP/WebSocket server
platform_/   capability model, per-platform adapters, insertion policy
models/      registry of candidates, verified fetch, manifest
server/      tenant isolation, admission, lineage, training gates
bench/       benchmark harness for the two model roles
```

## Honest limits

Removing personal words is data minimization, not anonymization. A voice stays
identifiable through its acoustics, and context can identify someone without
naming them, so retained audio, linked transcripts and customer-specific model
artifacts are personal data. Two local models cannot detect all sensitive
speech: the privacy model only sees ASR output, so a mistranscribed identifier
can survive - there is a test that demonstrates exactly that
(`test_a_mistranscribed_identifier_can_escape_the_rules`).

Nothing here is a GDPR compliance claim. See
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) and
[docs/EVALUATION.md](docs/EVALUATION.md) for what would have to be measured and
documented before production collection.

## Test map

| File | Tests | Covers |
| --- | --- | --- |
| `tests/test_dataset_builder.py` | 33 | plan section 7 end to end: the worked example, fail-closed paths, packaging |
| `tests/test_logging_and_audio.py` | 41 | log-leak guards, bounded buffers, VAD, reconciliation, whisper.cpp parsing |
| `tests/test_api.py` | 36 | pairing, scopes, Host/Origin, idempotent control, WebSocket framing |
| `tests/test_platform_and_models.py` | 36 | capability honesty, insertion policy, verified model fetch |
| `tests/test_privacy_schema.py` | 32 | the classifier output contract and its validator |
| `tests/test_rules.py` | 31 | deterministic detectors, spoken forms, injection attempts |
| `tests/test_server_side.py` | 31 | admission, tenant isolation, lineage deletion, promotion gates |
| `tests/test_time_map.py` | 31 | interval algebra, destination mapping, resampling, alignment checks |
| `tests/test_store.py` | 29 | AEAD payloads, key handling, guarded transitions, retention |
| `tests/test_consent_and_queue.py` | 26 | consent, eligibility, the queue and the withdrawal races |
| `tests/test_coordinator.py` | 22 | whole sessions: delivery, cancellation, shortcuts, training copy |
| `tests/test_cli.py` | 18 | every command, with isolated keys and payload root |
| `tests/test_state_machine.py` | 17 | plan section 5 constraints, including the lost key-up |
| `tests/test_privacy_pipeline.py` | 15 | windowing, coverage, the rules/model union |
