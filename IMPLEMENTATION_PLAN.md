# Local dictation and privacy-filtered personalization: implementation plan

Prepared 18 September 2026. This is a self-contained handoff for an implementation model. It specifies a future application; it does not claim that the application or its privacy guarantees have been implemented or verified.

## 1. Product and confirmed decisions

Build a desktop dictation application that runs two AI models locally: a speech-to-text model and an instruction model that identifies sensitive passages in transcripts. A global shortcut starts microphone capture, an unobtrusive indicator confirms recording, and releasing the shortcut finishes dictation and inserts the complete transcript into the focused text field. Also support toggle recording and locking an active hold-to-record session. Provide a local API for applications such as live caption overlays.

During active recording, capture audio and its transcript locally. After transcription, process a separate training copy: detect sensitive passages, remove corresponding audio and text, and retain aligned, privacy-filtered examples. Following explicit opt-in, eligible examples upload automatically to a server for customer-specific speech recognition improvement.

Confirmed by the product owner:

- Support Windows, macOS, and Linux from the first release.
- Target modest, average consumer hardware.
- Training uploads are optional; after opt-in they happen automatically without requiring review of every recording.
- Text inserted into other applications remains the full dictation. Redaction applies to training data.
- The present deliverable is this Markdown plan only. Do not implement the application as part of the planning task.

Provisional engineering assumptions, not additional user decisions:

- Baseline: 8 GB total RAM, a reasonably recent four-core CPU, integrated graphics, no dedicated GPU. A 16 GB machine is the recommended development and benchmarking tier.
- English is the initial validated language; use a multilingual-capable ASR architecture. Other languages and code-switching require separate privacy evaluation before their recordings become eligible for automatic upload.
- Personalization means improving recognition of a customer's speech, accent, and vocabulary. Voice synthesis, voice cloning, speaker identification, and emotion inference are out of scope.
- Begin with individual adult users and isolated customer training. Enterprise employee deployment, minors, and shared-model training need separate product and legal decisions.
- Use EU/EEA hosting initially. Exact supported OS versions, hardware reference machines, and deployment country must be recorded before release.

## 2. Recommended privacy design

Use two independent paths: immediate local dictation for the user's task, and asynchronous, optional training-data processing. Do not delay ordinary text insertion while loading the privacy model. Record only while the user explicitly activates recording; do not monitor ambient audio between sessions.

Removing personal words is data minimization, not proof of anonymization. A voice can remain identifiable through its acoustic characteristics, and context can identify someone without naming them. Treat retained audio, linked transcripts, and customer-specific model artifacts as personal data. EDPB guidance also requires a case-specific assessment before claiming an AI model is anonymous. [EDPB opinion on AI models](https://www.edpb.europa.eu/documents/opinion-of-the-board-art-64/opinion-282024-on-certain-data-protection-aspects-related-to_en)

The preferred improvement over indiscriminate collection is to retain fewer, higher-quality examples: discard sensitive or uncertain sentences, keep coherent nonsensitive clips, cap collection volume, and offer optional prompted readings of nonsensitive text. Prompted readings provide reliable labels and reduce accidental collection. Local vocabulary correction should be available without cloud participation. On-device adaptation is a later option if hardware permits; federated learning is not an automatic privacy guarantee and is outside the first release.

Two local models cannot guarantee detection of all sensitive speech. The privacy model sees ASR output, so it can miss something the recognizer omitted or mistranscribed. Automatic uploads must be presented as risk-reduced personal-data processing, never as guaranteed anonymous sharing. A privacy-model failure must prevent training upload while preserving ordinary local dictation.

## 3. Architecture and technology direction

Use a Rust desktop core with a Tauri interface, TypeScript UI, and native platform adapters. Keep microphone capture, secrets, filesystem access, inference, and upload policy in the core or isolated workers. The UI must not receive broad filesystem or shell permissions. The particular UI toolkit is replaceable if native integration experiments identify a blocker.

Use a local ASR worker based on whisper.cpp and a local instruction-model worker based on llama.cpp. Communicate through private IPC, not externally reachable inference servers. Ship exactly two learned model roles initially; use a non-neural voice activity detector such as WebRTC VAD and deterministic text detectors rather than silently adding a third learned model.

```text
Global shortcut / tray control / authorized API control
                         |
                  Capture coordinator
                         |
            Audio buffer + local ASR worker
                  /                 \
 Full live/final transcript       Encrypted local session
           |                              |
 Approved local clients          Training consent gate
 + focused text insertion                 |
                             Rules + local privacy model
                                          |
                             Audio/text interval removal
                                          |
                              Quality + privacy gates
                                          |
                               Encrypted upload queue
                                          |
                             Customer-isolated training
                                          |
                              Validated model delivery
```

Modules and responsibilities:

| Module | Responsibility |
| --- | --- |
| Capture coordinator | Recording state, audio device lifecycle, session IDs, cancellation, resource limits |
| Platform adapters | Shortcuts, permissions, nonactivating indicator, target focus, text insertion |
| ASR worker | Streaming hypotheses, final transcript, word/segment timing, language metadata |
| Privacy worker | Structured sensitive-span classification with no tools or network access |
| Dataset builder | Validated offsets, time mapping, clip removal, aligned outputs, quality checks |
| Local store | Encrypted payloads, queue transactions, retention, deletion, consent versions |
| Local API | Client pairing, scopes, authenticated events, revocation, bounded queues |
| Upload worker | Eligible artifacts only, consent rechecks, resumable transport, receipts |
| Server | Tenant isolation, dataset lineage, training jobs, artifact registry, deletion workflows |

Pin inference dependencies and model revisions. Verify model checksums and redistribution terms; never execute downloaded model code. Bundle runtime binaries so normal users do not need Python, compilers, or a model-serving application. Model downloads and updates are separate, visible network operations; dictation works offline after installation.

## 4. Model selection and resource scheduling

Start ASR evaluation with quantized Whisper base and small variants. Choose the smallest model that meets measured recognition and latency requirements for the validated language. whisper.cpp supports desktop platforms, CPU inference, quantization, and acceleration; its word timestamps are experimental, so they are not sufficient evidence that a privacy cut is correct. [whisper.cpp documentation](https://github.com/ggml-org/whisper.cpp)

For the privacy role, benchmark a four-bit quantized Qwen3-4B-Instruct-2507 candidate through llama.cpp. It is a candidate, not a claim of adequate privacy recall or acceptable speed on every 8 GB machine. Validate the original model license, conversion, quantization quality, and exact artifact provenance. [Official Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

Constrain the privacy output to a small JSON schema and independently validate it. Grammar-constrained output improves structural reliability but does not establish that classifications are correct. llama.cpp supports a subset of JSON Schema for this purpose. [llama.cpp grammar documentation](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)

Keep ASR ready during dictation. On low-memory systems, finalize transcription, release ASR resources if necessary, then load the privacy model. Process training jobs when idle; new dictation takes priority and may pause or restart a privacy job. Bound context length and process long transcripts in overlapping sentence windows, with a final context review where feasible. Never mark unprocessed text as clean because it exceeded a context limit.

Target peak application memory below 4 GB on the 8 GB baseline, including workers and UI; measure this rather than deriving it from weight-file size. If no candidate meets both resources and privacy quality, retain local dictation and disable automatic contribution on that configuration. Do not substitute an unvalidated smaller classifier or cloud inference silently.

Initial performance targets, to be confirmed on named reference machines:

- Recording feedback within 150 ms of shortcut activation.
- Warm partial transcript updates within approximately 1 second; final insertion within 2 seconds of release for a typical short utterance.
- ASR processes audio faster than real time under sustained operation without an unbounded backlog.
- Background privacy processing does not block typing or microphone capture; show queue progress when slow.
- Cold model loading is visibly distinct from recording readiness. Capture early speech into a bounded buffer during warm-up, and report failure instead of dropping it silently.

## 5. Recording experience and cross-platform behavior

Offer remappable shortcuts rather than assuming one chord is free on every keyboard layout. Define three actions: hold-to-record, toggle-recording, and lock-current-recording. While holding, the lock action transitions to toggle recording; releasing the held key then does not stop it. Pressing the toggle action again stops it. Escape cancels without insertion or contribution.

Use a nonactivating overlay and tray indicator showing recording, locked recording, finalizing, permission failure, and microphone failure. Include elapsed duration and an accessible text/status announcement; do not rely on color alone. The overlay must not steal focus. Background training work has a separate status.

Recording state machine:

```text
IDLE -> STARTING -> RECORDING_HELD -> FINALIZING -> DELIVERING -> IDLE
                        |
                     lock
                        v
                  RECORDING_LOCKED -> FINALIZING

IDLE -> STARTING -> RECORDING_LOCKED     (toggle start)
Any active state -> CANCELLED / ERROR -> IDLE
```

Ignore key auto-repeat, serialize conflicting commands, and make stop/cancel idempotent. Stop capture on screen lock, sleep, device removal, process shutdown, or permission revocation. A missed release event must not cause indefinite recording: provide a visible stop action and a provisional 10-minute maximum session duration. Notify before the limit and finalize on reaching it. Never insert text following cancellation or an ambiguous recovery after a crash.

At recording start, remember the target application and focused editable element where the OS permits. At completion, insert only if focus still matches that target and the field remains suitable. If it changed, preserve the result in a local result panel and let the user choose where to paste. Do not send text to a newly focused terminal or chat accidentally. Never synthesize an Enter key to submit the result.

Prefer native text insertion where reliable. A clipboard fallback needs explicit disclosure because clipboard history and synchronization can retain full dictation. Preserve existing clipboard contents where feasible, restore only if the clipboard still contains the app's own replacement, and do not promise erasure from external clipboard managers. Provide manual copy/paste when automatic insertion is unavailable. Exclude protected password fields and offer per-application exclusions; do not claim complete detection of every sensitive field.

| Platform | Initial integration and limits |
| --- | --- |
| Windows | Native shortcut press/release handling, microphone permission checks, and text injection adapter. Input injection is restricted by process integrity levels; provide fallback for elevated targets rather than elevating the whole app. |
| macOS | Microphone authorization and Accessibility trust checks; native shortcut handling and a nonactivating panel. Permissions can be denied or revoked; show actionable status. |
| Linux X11 | Native shortcut and injection adapter tested against supported desktop environments; document its security context. |
| Linux Wayland | Use available GlobalShortcuts and consent-based input mechanisms through portals/native compositor integrations. Probe capabilities; support varies. Where global activation or injection is unavailable, provide tray/in-app controls and manual paste. |

Cross-platform launch means usable releases on all three OS families, with a published capability matrix. It does not justify claiming universal injection into every application or compositor. Test the platform adapters early, before polishing UI. Tauri's shortcut plugin can be a starting point, but it does not replace platform-specific validation. [Tauri global shortcuts](https://v2.tauri.app/plugin/global-shortcut/), [Windows SendInput restrictions](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput), [Apple microphone authorization](https://developer.apple.com/documentation/bundleresources/requesting-authorization-for-media-capture-on-macos), [Apple Accessibility trust](https://developer.apple.com/documentation/applicationservices/1459186-axisprocesstrustedwithoptions), [Wayland GlobalShortcuts portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.GlobalShortcuts.html), [RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html)

## 6. Capture, transcription, and local storage

Capture microphone audio only, not system output. Normalize to a documented canonical representation, initially 16 kHz mono PCM with sample-index timing. Preserve the resampling/time mapping needed to cut the exact stored signal. Use bounded audio buffers and explicit overflow errors.

Produce revisable partial hypotheses from overlapping audio windows and reconcile them by segment identity. Finalize a canonical transcript after stop. Keep punctuation and display formatting separate from spoken-word alignment. A silence detector prevents empty sessions from becoming hallucinated text or training examples.

Use the OS application-data directory, not a shared temporary directory or the source repository. Logical subdirectories: `models/`, `sessions/`, `queue/`, and `metadata/`. Store audio, transcripts, model prompts, and sensitive offsets encrypted with per-session data keys protected through the OS credential store. Use vetted authenticated encryption. If secure key storage is unavailable, disable persistent contribution storage until configured; do not fall back to plaintext.

Raw active-session audio and text are temporary working data, not an indefinite history. Persist encrypted chunks only as needed for opted-in contribution/recovery; with contribution off, use bounded memory and discard after delivery. If a long session needs disk buffering, use the same encrypted temporary store and lifetime rules. An optional local history would require a separate setting and is outside the default scope.

Proposed retention defaults are engineering choices, not statutory deadlines:

| Data | Default lifetime |
| --- | --- |
| Raw audio, raw transcript, sensitive spans | Delete immediately after dataset construction or rejection; hard expiry 24 hours even if processing fails |
| Eligible local upload package | Delete after confirmed upload; expire after 7 days offline |
| Rejected/uncertain examples | Delete after the decision; no indefinite review bucket |
| Server training examples | Up to 30 days, or earlier once no longer required for the training job |
| Personalized model artifacts | While the customer enables personalization, subject to deletion and rollback policy |
| Operational metadata | Minimal fields with a documented purpose and bounded retention; never transcript content |

Run cleanup on startup and periodically. Use atomic state transitions so a crash cannot promote partial output to upload-ready. Exclude payload directories from backups where supported; account for database journals, crash dumps, swap, and OS backups in the threat model. Key destruction helps make ciphertext inaccessible but must not be advertised as guaranteed physical erasure from SSDs or third-party backups.

## 7. Privacy detection and synchronized audio/text removal

Detect direct identifiers, contact details, addresses, account/government identifiers, financial details, credentials and secrets, customer-confidential material, and contextual combinations that can identify a person. Include health, political/religious information, sexuality, and other sensitive personal narratives. User-defined private terms stay local and augment the detector. Names can be ambiguous; favor excluding questionable training material over retaining more examples.

Pipeline for an opted-in session:

1. Freeze the canonical transcript revision, word IDs, and sample timings. Record model and policy versions.
2. Run deterministic patterns for structured identifiers and secrets, plus the local instruction model for contextual spans. Take the union of detections; a model cannot override a rules-based exclusion.
3. Give the model immutable token IDs and bounded contextual windows. Require an output such as `{ "spans": [{ "start_word_id": 12, "end_word_id_exclusive": 17, "category": "contact", "action": "drop_sentence" }], "uncertain": false }`. The empty-span case still requires a complete, valid response.
4. Treat transcript text as untrusted data, including spoken instructions to ignore the classifier's rules. The worker has no tools, filesystem authority beyond its input/output channel, or network access. Validate IDs, enum values, ordering, bounds, window coverage, and schema independently.
5. Expand sensitive spans to their complete sentence or utterance by default. This removes surrounding clues and avoids training on unnatural fragments. Fine-grained word cuts are allowed only for validated alignment cases; never rely on the model to invent replacement text or timestamps.
6. Convert removal spans into audio intervals using the ASR alignment. Add conservative padding, initially 250 ms on each side, merge overlapping intervals, and remove any neighboring words whose audio intersects the expanded cuts. Tune padding through listening tests; 250 ms is not a safety guarantee.
7. If timings are missing, overlapping, nonmonotonic, low-confidence, or inconsistent with the transcript, discard the containing utterance or entire session. Do not introduce an unapproved third alignment model to hide a failure of the two-model design.
8. Build the complement of removed intervals. Produce retained contiguous audio clips and their exact matching spoken text. For a spliced playback/export artifact, concatenate those intervals and rebuild its timestamps; maintain a boundary manifest. Do not create new grammatical claims by joining unrelated sentences as though they were continuous speech.
9. Recheck remaining text for sensitive content and verify audio/text consistency. An optional re-transcription can reuse the same ASR model, but is not independent proof of privacy. Reject background conversations, uncertain language, poor-quality audio, and unexplained speech that is not aligned to text. Automatic single-speaker detection is imperfect; the product is for personal dictation, not meeting capture.
10. Mark the artifact eligible only if every required stage succeeds and consent is still valid. Otherwise delete it. No timeout, parse error, worker crash, or memory shortage may default to a clean result.

Example: “Send the parcel to Jane at 14 Oak Street. The meeting starts tomorrow.” Drop the first sentence's audio and text. Retain only the second sentence, assuming its context also passes. The full original text is still delivered to the user's focused application.

For retained source interval `[a, b)` following retained intervals with total length `L`, its destination interval is `[L, L + b - a)`. A retained word at sample `t` maps to `L + t - a`. Clip boundaries are half-open integer sample intervals. Regenerate text spacing/punctuation without changing spoken labels. Never upload the original sensitive text, removal map, or raw source timestamps merely to explain a deletion.

The canonical training dataset consists of contiguous examples, even when a spliced audio/text export is available. Do not use beeps, silence replacements, `[REDACTED]` labels, synthetic bridge audio, or rewritten prose as ordinary speech-recognition targets. Any playback-only fade must occur strictly inside retained samples and must not reintroduce removed audio.

## 8. Local real-time API

Provide a versioned loopback HTTP/WebSocket API for native clients. Bind explicitly to loopback, never all interfaces; an optional Unix socket/named pipe transport may follow. Disable the API until the user enables integration access. Loopback alone is not authentication.

Pair clients through the desktop UI, issue random revocable tokens with separate scopes, and store only token hashes on the server side. Initial scopes: `transcript:live`, `transcript:final`, `session:control`, and `status:read`. Reading captions must not grant permission to start a microphone or retrieve recordings. Do not expose raw audio/history over the first API version.

Proposed contract:

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/status` | Authenticated capabilities and coarse recording state; no transcript |
| `GET /v1/events` | Authenticated WebSocket upgrade for permitted event subscriptions |
| `POST /v1/sessions` | Start visible recording with control scope; return conflict if already active |
| `POST /v1/sessions/{id}/stop` | Idempotent finalize request for the active authorized session |
| `POST /v1/sessions/{id}/cancel` | Idempotent cancellation with no insertion or contribution |

Example event:

```json
{
  "version": 1,
  "event": "transcript.partial",
  "session_id": "random-session-id",
  "seq": 21,
  "segment_id": "segment-3",
  "revision": 4,
  "start_ms": 2600,
  "end_ms": 4100,
  "text": "The meeting starts tomorrow",
  "is_final": false,
  "privacy": "unredacted"
}
```

A later revision replaces the earlier text for that segment; clients must not blindly append partials. Events include `session.started`, `transcript.partial`, `transcript.final`, `session.stopped`, `session.cancelled`, and `error`. `transcript.final` contains the complete canonical transcript and supersedes provisional segments. Session stop follows finalization; cancellation invalidates provisional output but cannot erase content a client already observed.

Caption clients receive unredacted text only after a clear local grant. Explain that the app cannot control the receiving client's logging or remote forwarding. Streaming privacy redaction is not promised: later words can reveal that earlier text was sensitive, and post-session filtering cannot undo disclosure. A future filtered stream must have distinct permissions, delay semantics, and evaluation.

Reject unauthorized HTTP and WebSocket requests, validate Host and Origin headers, deny arbitrary browser origins, avoid credentials in URL query strings, and rate-limit pairing/authentication attempts. Browser integrations require a separately designed handshake rather than permissive CORS. Bound event buffers; disconnect slow clients with a clear resynchronization error, and expose only an authorized active-session snapshot for reconnect. Do not retain a transcript replay archive by default. API errors and logs must not contain dictated text.

## 9. Consent, automatic uploads, and server boundary

Contribution is off by default. Explain the destination, data types, training purpose, retention, identifiable nature of voice, automatic upload behavior, and withdrawal process before opt-in. Refusing contribution leaves local dictation functional. Keep customer-specific personalization separate from any future shared-model contribution. Consent must be freely given, specific, informed, and unambiguous; employment deployments need additional care because of power imbalance. [European Commission consent guidance](https://commission.europa.eu/law/law-topic/data-protection/information-individuals_en)

Record consent version, purposes, timestamp, and revocation status. Provide one-click pause, withdrawal, “delete my training data and personalized model,” contribution history without raw sensitive text, and an optional per-session “do not contribute” control. No per-recording review is required for eligible clips; uncertain clips are discarded automatically. Do not upload earlier sessions retroactively when consent is later enabled.

Training job states:

```text
LOCAL_PENDING -> ANALYZING -> BUILDING -> ELIGIBLE -> UPLOADING -> ACKNOWLEDGED
       |             |           |          |           |
       +-------------+-----------+----------+-----------+-> DELETED / REJECTED
```

Recheck consent and expiry at enqueue, before transfer, at server admission, and before training. Withdrawal must cancel queued/in-flight work, stop new training, and initiate deletion of already received artifacts. Handle races explicitly: a server receipt arriving after local withdrawal does not make the sample eligible again.

Upload packages contain only retained audio/text pairs, random sample ID, sample rate, language, duration, quality metrics, model/policy versions, consent reference, and integrity checksums. Authentication determines customer ownership; never trust a tenant ID supplied in the package. Strip filenames, device usernames, precise location, application titles, raw prompts, and sensitive audit spans. Customer linkage remains personal data and is required for personalization/deletion.

Use authenticated TLS, tenant-scoped authorization, encrypted object storage, scoped job credentials, and separation between account identity and dataset objects. The upload worker must not have read access to the raw-session store. Use idempotency keys, bounded retry with backoff, checksum validation, and explicit upload receipts. Neither transport failure nor server rejection may cause fallback to uploading the original recording.

The server validates package format, consent, size limits, integrity, and eligibility version. Client checks are not a security boundary against malformed uploads. Keep rejected server material quarantined only for the short time necessary to reject/delete it; do not log its contents. Automated secondary checks can reduce residual risk but do not turn uploaded data into anonymous data.

## 10. Personalization and deletion-aware training

Start with customer-specific fine-tuning or adapters, with no mixing between customers and no promotion into a global model. Keep the base model unchanged. The training implementation must prove that its chosen adaptation method can be exported, quantized, and loaded by the desktop ASR runtime before large-scale collection starts.

Use filtered contiguous audio/text pairs, quality thresholds, duplicate detection, and a held-out evaluation split by session/time. ASR-generated labels can reinforce the recognizer's mistakes: accept only sufficiently reliable examples, offer optional corrections of retained nonsensitive clips, and support prompted readings. Do not inspect arbitrary edits in other applications to infer user corrections.

Compare each candidate against the base model on customer-held-out speech and a general regression set. Define an improvement threshold and regression ceiling before training; evaluate word/character error rate as appropriate, latency, model size, and memory. If there is too little reliable data or no measured improvement, keep the base model and report that outcome honestly.

Maintain a lineage graph from sample IDs to dataset versions, jobs, checkpoints, adapters/merged weights, quantized builds, and delivered artifacts. Sign model manifests, verify downloads, install atomically, and retain a compatible rollback path. Never activate a model that exceeds the customer's device budget. Model delivery is independent of contribution consent changes and must obey deletion/revocation policy.

Deletion must cover queued uploads, object storage, derived datasets, checkpoints, caches, and affected customer model artifacts. The first implementation can discard the entire customer adapter and fall back to the base model; retrain only from remaining authorized samples if requested. Deleting examples alone does not remove their influence from trained weights. Do not claim reliable machine unlearning without evidence. Document backup expiry and prevent restored backups from reintroducing deleted samples. Copies exported outside the service cannot be technically recalled; disclose this boundary. Erasure duties have conditions and exceptions requiring a documented response process. [European Commission erasure guidance](https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/dealing-requests-individuals_en)

## 11. GDPR and deployment readiness

This plan supports privacy engineering; it is not a certification of GDPR compliance. Redaction is only one safeguard. Identify controllers/processors, document a lawful basis for each purpose, apply minimization and retention, handle data-subject rights, arrange processor contracts, and assess international access/transfers. Voice is not automatically special-category biometric data merely because it is recorded; the definition and identification purpose matter. Spoken content can independently contain special-category data. Ordinary contributor consent does not authorize processing every third party mentioned in a recording. Review Articles 4–9, 12–22, 25, 28, 32–35, and Chapter V with qualified counsel before production. [GDPR legal text](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng)

Complete a DPIA screening before collecting production training data, and a DPIA where the processing is likely to create high risk. Make its risk findings actionable in collection rules, testing, access controls, and retention. [European Commission DPIA guidance](https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/obligations_en)

Release evidence should include a data-flow inventory, privacy notice, consent screenshots, deletion exercise, processor/subprocessor list, hosting/access locations, incident procedure, and a threat model. Explicitly assess accidental bystanders, ASR omissions, prompt injection, compromised caption clients, clipboard history, stolen laptops, model supply-chain changes, and cross-tenant access. Do not promise protection against an already compromised OS.

Keep training disabled until the privacy evaluation and governance requirements are met. A local-only release may proceed independently if its own recording and storage safeguards are complete. Production upload credentials must not be enabled merely because the UI prototype works.

## 12. Implementation milestones and acceptance criteria

### Milestone 1: platform and hardware proof

Prove capture, shortcut down/up, toggle/lock, nonactivating status, and text insertion/fallback on Windows, macOS, Linux X11, and representative GNOME/KDE Wayland sessions. Benchmark the two model roles on an 8 GB CPU-only machine and a 16 GB reference machine. Record exact OS, CPU, memory, model, quantization, and timing. Decide supported versions and capabilities from evidence.

Exit: a documented platform matrix and resource budget, plus an end-to-end model-export experiment. No production data collection.

### Milestone 2: local dictation and integration API

Implement the recording state machine, audio capture, streaming reconciliation, final insertion, encrypted temporary storage, cancellation, cleanup, pairing, API scopes, and caption example client. Keep contribution off.

Exit: offline dictation after model installation; no unexpected network traffic; API transcript access denied without permission; focus changes never misdirect insertion; no duplicated text on repeated stop requests.

### Milestone 3: privacy dataset construction

Implement rules, structured local classification, conservative sentence removal, interval mapping, retained clips, spliced export, quality gates, and expiry. Use synthetic or explicitly authorized evaluation recordings with human-annotated sensitive audio intervals, including identifiers ASR gets wrong.

Exit: invalid outputs, uncertain language/alignment, worker failures, and queue overflow cannot yield upload-ready artifacts. Audio/text labels match retained speech, and no removed interval appears in an output file or metadata.

### Milestone 4: automatic contribution and rights controls

Implement explicit opt-in, consent versioning, upload queue, server admission, tenant isolation, receipts, pause/withdrawal, and deletion. Test against an isolated nonproduction server before enabling production uploads.

Exit: automatic upload works for eligible opted-in sessions without manual review; opted-out or rejected sessions never upload; withdrawal during transfer/training is correctly resolved; deletion is demonstrated through lineage and backup-restoration handling.

### Milestone 5: personalized training and distribution

Implement dataset snapshots, isolated training jobs, evaluation, compatible export, signed delivery, rollout/rollback, and model deletion. Cap storage and compute per customer.

Exit: a candidate shows measured improvement on held-out speech without unacceptable regressions; a customer model can be revoked/deleted and replaced by the base model.

### Milestone 6: release validation

Run packaging/signing, accessibility, permission-denial, crash-recovery, retention, and security tests across the supported platform matrix. Complete privacy/governance review and publish capability limitations.

Critical evaluation requirements:

- Measure sensitive-span recall, false negatives, over-redaction, and retained useful minutes by category and language. Include names, spoken numbers, accents, noise, code-switching, indirect identifiers, and adversarial spoken instructions.
- Evaluate residual sensitive speech in the retained audio by human listening, not just by searching retained transcripts. Include alignment boundary errors and omitted background speech.
- Provisional automatic-upload gate: at least 99% recall on the curated high-risk span benchmark, zero misses in a mandatory secrets/identifier regression suite, and reported sample sizes/confidence intervals. These are engineering targets, not an anonymity or GDPR guarantee; also assess session-level leakage. Disable automatic upload for unvalidated languages/configurations.
- Test Unicode, punctuation, repeated phrases, interval merging, timestamp remapping, fully redacted sessions, and clips too short to train on.
- Test denied permissions, shortcut conflicts, lost key-up, device removal, sleep/lock, cancellation, focus movement, clipboard changes, disk full, low memory, and interrupted model downloads.
- Test malicious local web pages, unauthorized WebSocket subscriptions, token revocation, slow consumers, replay/reconnect behavior, wrong-tenant requests, upload retry duplication, and withdrawal races.
- Audit filesystem/network/log outputs to ensure raw recordings, transcripts, prompts, and API tokens do not escape their designated boundaries.

## 13. Instructions for the implementation model

Treat confirmed product decisions as requirements. Preserve the separation between full local dictation and privacy-filtered training material. Build platform capability probes and model benchmarks first; then implement the milestones in order with tests proportionate to their privacy and data-loss risks.

Do not replace local inference with a remote service, require a dedicated GPU, add mandatory per-session upload review, or claim anonymous voice collection. Do not treat every operating system as having identical shortcut/insertion capabilities. Do not silently add learned model roles beyond the two specified.

Resolve routine engineering choices within this plan. Keep English, the hardware baseline, retention periods, latency targets, model candidates, and benchmark thresholds explicitly labeled as provisional until measured or confirmed. Obtain a product decision before expanding to shared-model training, meeting capture, minors, voice synthesis, or materially different collection purposes. The next implementation handoff should state what works, the tested platform matrix, measured model performance, and which upload gates remain unmet.
