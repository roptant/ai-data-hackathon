# Threat model

Written for: reviewers of the privacy and security posture, and for the release
evidence the plan asks for (section 11).

Scope: the desktop application, its local API, the upload path and the training
service skeleton. Out of scope: an already compromised operating system. If
another process on the machine has the user's privileges, it can read the
microphone, the clipboard and this application's memory, and nothing here
changes that.

## Assets

| Asset | Where it lives | Lifetime |
| --- | --- | --- |
| Live microphone audio | bounded in-memory buffer | the session |
| Canonical transcript | memory; the user's target application | the session |
| Encrypted raw session copy (opted in only) | `sessions/`, AES-256-GCM | hard 24 h expiry |
| Detected sensitive spans | never persisted unencrypted; never uploaded | the decision |
| Upload package (retained clips + text) | `queue/`, encrypted | until acknowledged, 7 days offline |
| Payload master key | OS credential store | until deleted |
| API tokens | hashes only, SQLite | until revoked |
| Upload credentials | deployment secret, outside this repo | - |
| Server samples and adapters | per-tenant storage | 30 days / while enabled |

## Threats and what is done about them

### Accidental bystanders

Someone else speaks during dictation. **Partly mitigated.** Quality gates reject
clips that are mostly noise and clips whose words explain far less speech than
the VAD found (`check_audio_text_consistency`). Single-speaker detection is
imperfect and is not claimed; the product is personal dictation, not meeting
capture. Residual risk stays, and the listening pass in
[EVALUATION.md](EVALUATION.md) is how it gets measured.

### ASR omissions and mistranscriptions

The privacy model only sees ASR output, so an identifier the recogniser mangled
is never classified. **Not fully mitigable by design.** Deterministic rules catch
some spoken forms the model might miss (digit-word runs, spelled addresses),
sentence-level removal takes the surrounding context, and uncertainty rejects
the session. The residual case is demonstrated by a test rather than hidden.

### Prompt injection through dictated speech

A user (or someone dictating at them) says "ignore all previous instructions".
**Mitigated structurally.** The prompt states the transcript is untrusted data;
the worker has no tools, no network and no filesystem authority beyond its own
channel; its answer is validated against the frozen transcript; rule hits are
unioned in afterwards and cannot be overridden; and a detected attempt both
drops the utterance and makes the analysis uncertain, so the session is
rejected. An injection can therefore reduce what is collected, never widen it.

### A compromised or careless caption client

A paired client receives unredacted live text. **Bounded, not prevented.** Tokens
are paired through the desktop UI, scoped (`transcript:live` does not grant
`session:control`), hashed at rest and revocable; raw audio and history are not
exposed at all. What the client then logs or forwards is outside this
application's control, and the consent text says so. Streaming redaction is not
offered, because later words can reveal that earlier text was sensitive.

### A malicious local web page

A page in the user's browser tries to reach the loopback API. **Mitigated.** The
API is disabled by default, binds only to loopback, requires a bearer token,
rejects any request carrying an `Origin` header, rejects a `Host` header that is
not the loopback address it bound to (DNS rebinding), refuses credentials in the
query string, and rate-limits authentication attempts.

### Clipboard history and synchronisation

The clipboard fallback can leave a full dictation in a clipboard manager or sync
to another device. **Disclosed, off by default.** It must be enabled explicitly;
previous contents are restored only if the clipboard still holds our
replacement; no claim is made about erasing it from external managers.

### Text inserted into the wrong window

Focus moves between recording and delivery. **Mitigated.** The target is
remembered at recording start and insertion proceeds only if focus still
matches; otherwise the text waits in a local result panel. Password fields and
excluded applications receive nothing, and Enter is never synthesised. Where the
platform cannot report focus at all (Wayland), insertion is refused rather than
guessed.

### A stolen or lost laptop

**Partly mitigated.** Payloads are encrypted with keys derived from a master key
in the OS credential store, so they are not readable without unlocking the
account. An unlocked, logged-in machine offers no protection. Key destruction
and file overwriting make ciphertext inaccessible in practice; they are not
advertised as physical erasure from SSD wear-levelling, swap, hibernation files,
crash dumps or third-party backups. Payload directories carry a `CACHEDIR.TAG`
and, on macOS, a no-index marker; both are advisory.

### Model supply chain

A tampered or substituted weight file. **Mitigated.** URL, digest, size, licence
and revision are pinned; downloads are HTTPS-only, size-capped, verified before
being renamed into place, recorded in a manifest and re-verified at startup;
only inert weight-file extensions are accepted; nothing downloaded is executed.

### Cross-tenant access on the server

**Mitigated in the skeleton.** Authentication resolves the tenant; no tenant
field is read from the package, and a package containing one is refused.
Storage is per tenant, sample caps are per tenant, and lineage nodes carry a
tenant ID. A real deployment adds tenant-scoped authorization at the storage
layer, which this skeleton does not have.

### Upload duplication and withdrawal races

**Mitigated.** A deterministic archive yields a stable idempotency key, so a
retry returns the first receipt instead of creating a second sample. Consent is
rechecked at enqueue, before eligibility, before transfer and after a slow
transfer; withdrawal cancels queued and in-flight work and deletes payloads; a
receipt arriving afterwards is recorded as not accepted and opens a deletion
request.

### Logs, crash reports and telemetry

**Mitigated.** The event log accepts only numbers, booleans, enums and
token-shaped strings and refuses content-shaped field names; subprocess stderr
is never logged because it echoes transcript text; the HTTP access log is
suppressed; API errors carry reason codes.

### Backups reintroducing deleted data

**Partly mitigated.** The lineage graph records deletions and the server refuses
to re-admit a deleted sample ID, so a restored backup or replayed upload cannot
quietly bring one back. Backup expiry itself is a deployment matter and has to
be documented separately.

## Deletion, honestly

Deleting examples does not remove their influence from trained weights. The
first implementation discards the whole customer adapter and falls back to the
base model, and retrains only from remaining authorized samples if asked.
Reliable machine unlearning is not claimed. Copies exported outside the service
cannot be technically recalled, and that boundary is disclosed rather than
papered over.

## Residual risks a reviewer should weigh

- The privacy detectors have not been measured on real speech. Everything above
  about detection is a mechanism, not a result.
- Padding of 250 ms is a guess until listening tests are run.
- The retained-audio consistency check is coarse; bystander speech that aligns
  with the transcript would pass it.
- Contextual identifiability - a session that names nobody but describes a
  situation only one person is in - is the weakest area of any span-based
  approach, including this one.
