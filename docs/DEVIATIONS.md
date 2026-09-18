# Deviations from the plan, and what is still a skeleton

Written for: whoever reviews this against `IMPLEMENTATION_PLAN.md` and decides
what to build next.

## Deviations

### 1. Python, not a Rust core with a Tauri UI

The plan specifies a Rust desktop core, a Tauri interface and a TypeScript UI
(section 3). This implementation is Python. That was an explicit instruction
for this pass; the machine it was built on has no Rust or Node toolchain, and
the alternative was writing several thousand lines of Rust that nobody could
compile or test.

What that costs:

- **No process isolation as specified.** The plan keeps microphone capture,
  secrets, filesystem access, inference and upload policy out of the UI's
  reach. Here the boundaries exist as object-graph restrictions (for example
  `PackageReader`) rather than as separate processes with different
  permissions. A Rust port keeps the same seams but can make them real.
- **No packaging or code signing**, so plan milestone 6 is untouched.
- **Python-level memory hygiene is weaker.** Key material is handled as
  `bytearray` where it matters, but immutable `bytes` and `str` cannot be wiped.

What carries over unchanged if you port it:

- the interval mathematics and the removal plan (`time_map.py`,
  `dataset/intervals.py`, `dataset/builder.py`),
- the classifier output contract and its validator (`privacy/schema.py`,
  including the GBNF grammar),
- the state machine, the job state table, the retention table, the package
  field allowlist, the API surface and every reason code,
- the test suite, which is written against behaviour rather than internals and
  is close to a specification of the privacy-critical parts.

### 2. Two model roles are wired but unresolved

The plan names candidates to benchmark. The registry contains them with no URL
and no checksum, and every consumer refuses until one is recorded. This is a
deviation only in the sense that nothing runs a model yet; it is what the plan
asks for at this stage (choose from evidence, pin the artifact, verify it).

### 3. No microphone capture backend

`AudioSource` is implemented for files, memory and synthetic audio.
`MicrophoneSource` probes for `sounddevice`/`pyaudio`/`soundcard` and raises
`CapabilityUnavailable` with instructions. A silent stand-in would have made the
whole capture path look finished while recording nothing.

### 4. Streaming partials are not produced by a real backend

`asr/reconciler.py` and the scripted backend implement and test revision
handling, but `WhisperCppWorker.stream` refuses: incremental re-decoding through
a one-shot subprocess would not meet the ~1 second partial target. The
capability is reported rather than faked, so an API client sees
`partials_available: false`.

## Skeletons, and what "skeleton" means for each

### Server admission (plan section 9) - skeleton with real gates

Implemented: token-hash authentication resolving the tenant, per-tenant storage
separation, package re-verification (structure, field allowlist, checksums,
size, eligibility version), idempotency, brief quarantine with no content
logging, receipts, sample caps, refusal to re-admit a deleted sample.

Not implemented: an HTTP surface (admission is in-process; the loopback
transport uses it directly), TLS termination, real object storage, an identity
provider, rate limiting, and the server-side consent register that a real
deployment would check independently of the client.

### Personalized training (plan section 10) - gates without a trainer

Implemented: dataset snapshots, held-out split by time rather than at random,
pre-registered promotion policy, `decide_promotion` with improvement threshold,
regression ceiling, device-memory budget, artifact size and latency checks, a
runtime probe obligation that fails closed, Ed25519-signed manifests with
verification, and a delivery registry with rollback and revocation.

Not implemented: the trainer itself. `UnavailableTrainer` raises with the
reason, because the plan requires proving that the chosen adaptation method can
be exported, quantized and loaded by the desktop ASR runtime *before*
large-scale collection.

### Desktop shell and indicator (plan section 5) - not built

The state machine emits the states the indicator needs
(`recording`, `locked`, `finalizing`, permission and microphone failure), and
`status()` exposes elapsed duration and a coarse state with no transcript. There
is no overlay, no tray icon, and no accessible announcement, because there is no
UI layer in this pass.

## Provisional values, all in one place

`config.py` and `version.py` hold every number the plan marks provisional:
250 ms padding, the 10-minute session cap, the 24-hour/7-day/30-day retention
defaults, window sizes and timeouts, quality thresholds, daily collection caps,
the 150 ms/1 s/2 s latency targets and the 4 GB memory target. None of them has
been measured on a reference machine.

## What the next handoff should state

The plan asks each handoff to say what works, the tested platform matrix, the
measured model performance and which upload gates remain unmet. Today:

- **Works:** everything in the README's first table, under test.
- **Tested platform matrix:** Windows only, and only the probe plus dry-run
  insertion - `dictation probe --write` on each target machine produces the
  artifact to publish. macOS, X11 and Wayland report `unknown`/`unavailable`
  with reasons and have not been run on real hardware.
- **Measured model performance:** none. `dictation bench` runs and writes a
  report, and labels the figures as measuring the pipeline rather than a model.
- **Unmet upload gates:** all of them. No sensitive-span recall measurement, no
  secrets/identifier regression suite over real recordings, no human listening
  pass over retained audio. `UPLOAD_GATES_MET` stays `False`, so automatic
  upload is refused even after opt-in.
