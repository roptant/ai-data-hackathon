# Local Dictation

Local-first desktop dictation with an optional, separately constructed
privacy-filtered training copy. The production architecture is Rust + Tauri;
the original Python package remains as a behavioral reference while platform
and privacy evidence is completed.

The full transcript is delivered locally. Contribution is a different path:
rules and a local classifier remove sensitive sentences from both audio and
text, quality gates reject uncertain material, and upload remains disabled
unless every consent and evaluation gate passes.

## Current state

| Area | State |
| --- | --- |
| Recording, bounded capture, VAD, partial reconciliation | Rust implementation with tests |
| Microphone capture and canonical 16 kHz conversion | Rust/CPAL implementation |
| Whisper and privacy-model workers | Private framed IPC with deadlines and Linux confinement |
| Privacy rules, schema validation, removal geometry, dataset packages | Rust implementation with fail-closed tests |
| Encrypted local stores, retention, consent, queue and withdrawal | Rust implementation |
| Loopback HTTP/WebSocket API and caption client | Rust implementation with scoped tokens and security tests |
| Training server, tenant isolation, lineage and deletion | Rust implementation with end-to-end round-trip test |
| Signed personalized-model delivery and rollback | Rust implementation; trainer experiment is incomplete |
| Tauri desktop, settings UI, tray, shortcuts and indicator | Implemented; platform validation is incomplete |

Production automatic upload is disabled. The recorded Qwen evaluation catches
35/40 high-risk spans (87.5%) and 15/15 mandatory secrets, below the provisional
99% high-risk recall gate. See
[`docs/results/privacy-eval-qwen3-4b.json`](docs/results/privacy-eval-qwen3-4b.json).

## Build and test

For custom Whisper imports, Hugging Face conversion, recording shortcut modes,
and connecting the standalone caption overlay, see
[Custom models and Live Caption](docs/CUSTOM_MODELS_AND_CAPTIONS.md).

Prerequisites are Rust 1.85 or newer, Node.js/npm, CMake, and the native
dependencies required by Tauri and the audio backend.

```bash
npm --prefix ui ci
npm --prefix ui run build
cargo test --workspace --all-targets
```

The model binaries and weights are not committed. For a release bundle, build
and stage the two worker sidecars before invoking Tauri with the bundle overlay:

```bash
node scripts/stage-workers.mjs
cargo tauri build --config src-tauri/tauri.bundle.conf.json
```

Normal Cargo builds intentionally do not require pre-staged release sidecars.

## Workspace

| Path | Responsibility |
| --- | --- |
| `crates/dictation-core` | Pure recording, transcript, privacy, interval, dataset and policy logic |
| `crates/dictation-storage` | Encrypted content scopes, settings, API clients and contribution state |
| `crates/dictation-worker` | Bounded private worker protocol and supervision primitives |
| `workers/asr`, `workers/privacy` | whisper.cpp and llama.cpp worker executables |
| `crates/dictation-engine` | Model orchestration, training-copy jobs and uploads |
| `crates/dictation-api` | Paired loopback API and caption example client |
| `crates/dictation-server` | Admission, tenant storage, training, lineage and delivery |
| `crates/dictation-models` | Pinned model registry, verification and signed installation |
| `crates/dictation-platform` | Microphone and native desktop adapters |
| `src-tauri`, `ui` | Trusted desktop process and narrow TypeScript UI |
| `src/dictation`, `tests` | Frozen Python behavioral reference |

## Verified and still open

- KDE Wayland has passed an end-to-end test with file-backed audio, live
  captions, final ASR and focus-verified AT-SPI insertion into a GTK field.
- Apple Silicon macOS compiles and passes the Rust workspace tests. Automatic
  insertion fails closed to the result panel because a safe native focus/window
  adapter is not yet implemented.
- Windows-specific platform code has been cross-compiled, but a full Windows
  desktop run and installer validation are still required.
- GNOME Wayland, X11, real microphone/voice runs across the platform matrix,
  packaging, signing, crash recovery and permission-denial exercises remain.
- The reference trainer exports fine-tuned Whisper weights, and the server
  enforces `q5_1` quantization before evaluation or signing. The real-model
  export → quantize → desktop-load comparison has not been completed.
- Privacy evaluation still needs real authorized speech, ASR-error cases and a
  human listening pass over retained audio. The synthetic transcript benchmark
  is not sufficient to enable collection.

Removing sensitive words does not anonymize a voice, and this project makes no
GDPR certification claim. Read [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md),
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), and
[`docs/EVALUATION.md`](docs/EVALUATION.md) before changing any upload gate.

Migration-specific status and invariants are in
[`docs/RUST_MIGRATION.md`](docs/RUST_MIGRATION.md).
