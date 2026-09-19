# Architecture

> This is the architecture of the retained Python behavioral reference. The
> production Rust/Tauri workspace and its current status are mapped in the
> repository README and [`RUST_MIGRATION.md`](RUST_MIGRATION.md).

Written for: engineers picking this codebase up against the plan.

The module layout follows the plan's module table (section 3). This document
maps code to plan sections so a reviewer can check an intent against its
implementation, and describes the two flows that matter.

## The two paths

```
                shortcut / tray / authorized API control
                                  |
                        capture/coordinator.py
                                  |
                capture/audio_buffer.py + asr/*  (local recognition)
                       /                          \
      FULL transcript                         encrypted session copy
              |                                        |
   api/events.py (paired clients)            consent/consent.py gate
   platform_/insertion.py (focused field)              |
                                          privacy/rules.py + privacy/classifier/*
                                                       |
                                            dataset/intervals.py  (cuts)
                                                       |
                                            dataset/builder.py    (clips)
                                                       |
                                            dataset/quality.py    (gates)
                                                       |
                                            dataset/package.py    (allowlist)
                                                       |
                                            upload/queue.py + worker.py
                                                       |
                                            server/app.py  (admission)
                                                       |
                                            server/training.py (promotion gates)
                                                       |
                                            server/lineage.py  (deletion)
```

The left branch never waits for the right one. `CaptureCoordinator._finalize`
delivers the transcript first and only then calls `_training_copy`; every
failure in the right branch ends in a rejection with a reason code.

## Module map

| Plan module | Code | Plan sections |
| --- | --- | --- |
| Capture coordinator | `capture/coordinator.py`, `capture/state_machine.py`, `capture/audio_buffer.py`, `capture/devices.py`, `capture/vad.py` | 5, 6 |
| Platform adapters | `platform_/base.py`, `platform_/windows.py`, `platform_/posix.py`, `platform_/probe.py`, `platform_/insertion.py` | 5, 12 (milestone 1) |
| ASR worker | `asr/base.py`, `asr/whisper_cpp.py`, `asr/reconciler.py`, `asr/mock.py` | 3, 4, 6 |
| Privacy worker | `privacy/rules.py`, `privacy/schema.py`, `privacy/classifier/*`, `privacy/pipeline.py` | 4, 7 |
| Dataset builder | `dataset/intervals.py`, `dataset/builder.py`, `dataset/quality.py`, `dataset/package.py`, `time_map.py` | 7 |
| Local store | `store/crypto.py`, `store/keys.py`, `store/db.py`, `store/session_store.py`, `store/retention.py`, `paths.py` | 6 |
| Local API | `api/tokens.py`, `api/events.py`, `api/websocket.py`, `api/server.py` | 8 |
| Upload worker | `upload/states.py`, `upload/queue.py`, `upload/transport.py`, `upload/worker.py`, `consent/consent.py` | 9 |
| Server | `server/app.py`, `server/tenants.py`, `server/lineage.py`, `server/training.py` | 9, 10 |
| (cross-cutting) | `types.py`, `config.py`, `version.py`, `errors.py`, `logging_.py`, `app.py`, `cli.py`, `bench/harness.py` | 1, 4, 11, 12 |

## Invariants worth knowing before you change anything

**Intervals are half-open integer sample ranges.** `types.SampleInterval`
enforces it. Removal intervals round outward, retained intervals round inward
(`time_map.ResampleMap`), so a rounding error can never leave part of a removed
word in the output.

**A transcript revision is frozen.** Word IDs, texts and timings do not change
after `Transcript` is constructed. Spans refer to word IDs, so the privacy model
cannot name a position it was not shown, and `privacy/schema.py` rejects any ID
outside the window.

**The union is one-directional.** Rule and user-term hits carry
`DetectionSource.RULES`/`USER_TERMS` and are `is_mandatory`. The model can add
spans; nothing removes a mandatory one. `privacy/pipeline.analyze` takes the set
union and never subtracts.

**Everything fails closed.** A worker crash, timeout, schema violation, missing
timing, silence, uncertainty flag, coverage gap or unvalidated language produces
`Decision.REJECTED` with a reason. `dataset/builder.py` has no path from a
failure to an eligible artifact.

**Removal geometry converges.** `dataset/intervals.plan_removals` pads, merges,
then pulls in every word whose audio touches a cut, and iterates to a fixed
point. Otherwise a padded cut could delete audio for a word still listed in the
retained text.

**The upload worker cannot read raw audio.** It holds a
`store.session_store.PackageReader`, which refuses any artifact kind other than
`PACKAGE`. The restriction is in the object graph, not in a comment.

**Nothing content-shaped reaches a log.** `logging_.sanitize` accepts numbers,
booleans, enums and token-shaped strings only, and refuses field names like
`text`, `prompt` or `spans`. Three times during development it caught a field
that would have leaked; that is what it is for.

**State changes are guarded SQL updates.** `store/db.transition_job` updates
`WHERE state IN (allowed)` and raises `StateTransitionConflict` on a lost race.
A crash between building and eligibility cannot promote a partial artifact.

## Threading

`CaptureCoordinator` owns one capture thread per session, reading fixed-size
chunks into the bounded buffer and calling `RecordingStateMachine.tick()` for
the duration limit. The state machine itself is pure and is always entered under
the coordinator's lock. The API server runs its own threads; it reaches the
coordinator only through the `SessionController` methods.

## Where to plug real components in

| Need | Seam |
| --- | --- |
| Real microphone | implement `capture/devices.MicrophoneSource` |
| Real ASR | `asr/whisper_cpp.WhisperCppWorker` once a model is resolved and fetched |
| Real privacy model | `privacy/classifier/llama_cpp.LlamaCppPrivacyWorker`, same |
| Press/release shortcuts | platform adapter, then flip the capability in `probe()` |
| Real upload server | `upload/transport.HttpsTransport` plus a deployment of `server/app.py` |
| Real trainer | implement `server/training.Trainer`, and a `RuntimeProbe` that loads the artifact in the desktop ASR runtime |
