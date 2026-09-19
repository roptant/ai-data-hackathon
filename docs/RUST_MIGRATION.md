# Rust + Tauri migration

Rust/Tauri is now the primary implementation. The Python package stays in the
repository as a behavioral reference until the remaining platform, model and
release evidence is complete.

## Migration rules

1. Port behavior, invariants and tests, not Python implementation details.
2. Keep privacy-critical domain logic independent of Tauri and OS APIs.
3. Keep microphone access, secrets, storage, inference and upload policy out of
   the WebView.
4. Report observed capabilities only. Missing or unverifiable focus must route
   output to the result panel rather than risk inserting into the wrong target.
5. Production upload remains disabled until every requirement in
   `docs/EVALUATION.md` is met.

## Implemented in Rust

- Pure recording state machine, shortcut interpretation and lifecycle policy
- Canonical audio buffering, resampling, VAD and CPAL microphone capture
- Frozen transcripts, partial reconciliation and word timing validation
- Deterministic privacy rules and a strict, window-bounded classifier schema
- Sentence expansion, fixed-point audio removal, interval mapping and packages
- Quality, language, duplicate, collection-cap and post-removal gates
- Authenticated encrypted storage with independently expiring data classes
- Consent, contribution jobs, upload retries, receipts, withdrawal and deletion
- Private persistent whisper.cpp/llama.cpp workers with deadlines
- Scoped loopback API, pairing, WebSockets and a caption example client
- Tenant-isolated server admission, lineage, training orchestration and deletion
- Signed model delivery, atomic install and rollback
- Tauri settings/status UI, tray, shortcut plumbing and recording indicator

## Remaining work

- Implement and exercise safe macOS focus tracking/native insertion. It is
  deliberately unavailable today; dictated text stays in the result panel.
- Complete real desktop validation on Windows, macOS, X11, GNOME Wayland and
  the supported KDE Wayland versions, including denial and lifecycle cases.
- Complete the fine-tune export → `q5_1` quantize → desktop worker load
  experiment and record held-out/regression results and resource use. The
  server now runs the configured whisper.cpp quantizer and verifies the output
  header before evaluation or signing; the real-model comparison is still
  outstanding.
- Build a human-annotated authorized audio benchmark and conduct the required
  listening review. The current synthetic transcript result is 87.5% high-risk
  recall and therefore fails the provisional upload gate.
- Finish installer packaging, signing/notarization, update/rollback exercises,
  crash recovery and the release security audit.
- Complete deployment governance and production server infrastructure. No
  production credentials or upload switch should be added before then.

## Platform evidence

| Platform | Evidence | Honest capability state |
| --- | --- | --- |
| KDE Wayland | End-to-end test with ASR, captions and AT-SPI GTK insertion | Tested on one development machine; regular overlay disabled because it stole focus |
| macOS arm64 | 158 Rust tests, strict Clippy, UI build, and normal/release-overlay Tauri builds | Focus tracking/native insertion unavailable; runtime hardware test pending |
| Windows | `dictation-platform` target check and strict Clippy for `x86_64-pc-windows-msvc`; Windows-native CI added | CI has not run for these local changes; full desktop, sidecar and installer runtime tests pending |
| GNOME Wayland / X11 | Adapters implemented | Runtime validation pending |

## Current verification

- Rust 1.98.1: workspace builds and tests pass on Apple Silicon macOS.
- TypeScript 5.9.3: UI build passes.
- GitHub Actions now runs the locked Rust, Python and TypeScript checks on
  macOS and Windows, including platform-specific tests and sidecar path checks.
- The Qwen transcript benchmark catches 35/40 high-risk cases and 15/15
  secrets; production upload remains disabled.
- Strict Clippy passes with warnings denied. Workspace-wide `cargo fmt --check`
  still reports legacy formatting drift from the large prior migration commit.

## Important invariants

- Clip ranges are half-open integer sample intervals; removal rounds outward
  and retention rounds inward.
- Transcript word IDs and timings are frozen before privacy analysis.
- Rules and private-term detections are mandatory; the model can only add cuts.
- Any worker, schema, timing, language or quality failure rejects the training
  copy and can never default to upload eligibility.
- Upload workers cannot access the raw-session store.
- Consent is rechecked before transfer, admission and training; withdrawal wins
  races and initiates server-side deletion.
- Server identifiers are validated random IDs, idempotency is tenant-scoped,
  and all declared checksums are verified.
