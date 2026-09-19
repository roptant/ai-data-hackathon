# Rust + Tauri migration

This repository is moving from the Python reference implementation to the
Rust/Tauri architecture specified in `IMPLEMENTATION_PLAN.md`. The Python code
stays in place until the Rust implementation has behavioral parity and the
platform and privacy evidence required by the plan.

## Rules for the migration

1. Port behavior, invariants, and tests—not Python implementation details.
2. Keep the privacy-critical domain crate independent of Tauri and operating
   system APIs.
3. Keep microphone, secrets, storage, inference, and upload policy outside the
   WebView. Tauri capabilities start minimal and expand only with reviewed use.
4. Never claim a capability merely because an interface exists. Recording,
   models, contribution, and delivery remain disabled until their real backend
   and tests are present.
5. Keep `UPLOAD_GATES_MET` false in the legacy implementation and add no Rust
   production-upload switch until the evaluation in `docs/EVALUATION.md` has
   been completed.

## Workspace

| Path | Purpose |
| --- | --- |
| `crates/dictation-core` | Pure state, timing, privacy, and policy logic |
| `crates/dictation-storage` | Encrypted content scopes and independent retention |
| `crates/dictation-worker` | Bounded, timeout-enforced private model IPC |
| `crates/dictation-platform` | Native capability probes and microphone capture |
| `src-tauri` | Trusted desktop process and narrow WebView command boundary |
| `ui` | TypeScript status/settings UI with no filesystem or shell permission |
| `src/dictation`, `tests` | Python behavioral reference during migration |

## Progress

- [x] Cargo workspace and operating-system-independent domain crate
- [x] Recording state machine port with Rust unit tests
- [x] Half-open interval, padding, complement, destination, and resampling port
- [x] Minimal Tauri 2 shell with a restrictive capability file and CSP
- [x] TypeScript migration-status UI that does not imply recording works
- [x] Frozen transcript and strict classifier-schema port
- [x] Sentence expansion and fixed-point audio removal port
- [x] Quality gates and clean upload-package validator
- [x] Encrypted local storage and independently expiring metadata/package rows
- [x] Asynchronous capture coordinator with separately owned training jobs
- [ ] Platform microphone, shortcut, indicator, focus, and insertion adapters
  - [x] Default-device microphone capture with bounded canonical PCM conversion
  - [ ] Remappable native shortcuts and nonactivating indicator
  - [ ] Native focus tracking and insertion implementations
- [ ] Private whisper.cpp and llama.cpp worker IPC
- [ ] Loopback API, consent, queue, upload worker, and hardened server
- [ ] Trainer, signed delivery, packaging, signing, and release validation

## Known defects that must not be ported

The migration treats these as regression tests and design constraints:

- Server identifiers must be parsed as random IDs and must never become an
  unchecked path component. Package validation must require all fields, types,
  clips, safe filenames, size limits, and checksum coverage.
- Raw audio must not be written as a plaintext WAV in a shared temporary
  directory. Workers receive it over private IPC or through encrypted,
  application-owned storage with bounded lifetime.
- Training processing owns no live capture state. It runs asynchronously after
  delivery, and completion of an old training job cannot mutate a newer session.
- Raw-session, eligible-package, and operational-metadata expiration are
  independent. Periodic cleanup is a supervised task, not startup-only work.
- Authentication throttling counts failed attempts without locking valid clients
  out after ordinary status requests. Revocation is rechecked for live streams.
- Server idempotency and lineage keys are tenant-scoped, and a declared transport
  checksum mismatch is a rejection.

## Build prerequisites

- Rust 1.85 or newer with Cargo
- Node.js and npm
- Platform prerequisites from the official Tauri 2 documentation

Once installed:

```bash
cargo test -p dictation-core
npm --prefix ui install
npm --prefix ui run build
cargo tauri dev --manifest-path src-tauri/Cargo.toml
```

## Verification status

The Rust/Tauri workspace was compiled with Rust 1.98.1. Workspace formatting,
Clippy with warnings denied, and all 64 Rust tests pass. The TypeScript source
and JSON configuration pass syntax checks. The frontend dependency build has
not been run because npm is not installed in the current environment.
