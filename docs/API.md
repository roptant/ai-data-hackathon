# Local real-time API, version 1

The dictation application exposes a loopback HTTP and WebSocket API for native
clients on the same machine — live caption overlays, stream-deck style
controls, note-taking integrations. This document is the contract.

Every request and response shape below was taken from a running server; the
reason codes are the exact strings the API returns.

**Read this first:** the API is **disabled by default**, every request needs a
token that the user paired through the desktop application, and the text it
carries is the **unredacted** dictation. Loopback is not authentication, and a
client that receives transcript text is trusted with everything the user
dictated.

---

## 1. Enabling and binding

The API is off until the user enables integration access. In
`metadata/settings.json`:

```json
{
  "api": {
    "enabled": true,
    "port": 8765,
    "client_queue_limit": 256,
    "auth_attempts_per_minute": 10,
    "pairing_window_seconds": 120
  }
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Starting the server while false raises `ApiError`. |
| `host` | `127.0.0.1` | Only `127.0.0.1`, `localhost` or `::1` are accepted. Binding to `0.0.0.0` is refused outright. |
| `port` | `8765` | Port `0` binds an ephemeral port; read it back from `LocalApiServer.port`. |
| `client_queue_limit` | `256` | Events buffered per client before it is disconnected as a slow consumer. |
| `auth_attempts_per_minute` | `10` | Shared limit across pairing and authentication attempts. |
| `pairing_window_seconds` | `120` | Lifetime of a pairing code. |

There is no `dictation api serve` command yet (see
[DEVIATIONS.md](DEVIATIONS.md)); the server is started by the host application:

```python
from dictation.api.events import EventBus
from dictation.api.server import LocalApiServer
from dictation.api.tokens import TokenManager
from dictation.app import Application

app = Application.open()
# build_coordinator refuses unless both model roles are resolved and installed;
# pass allow_stub=True for a demo with scripted stand-ins.
coordinator = app.build_coordinator()          # implements SessionController
server = LocalApiServer(app.tokens, coordinator.bus, coordinator, app.settings.api)
port = server.start()                          # raises ApiError if disabled
...
server.stop()
```

`LocalApiServer` is also a context manager, and `wait_for_port(port)` blocks
until it accepts connections.

---

## 2. Pairing and tokens

Pairing is **not** part of the HTTP surface. A client cannot authorise itself:
the user approves it in the desktop application, and only then does the client
receive a token. This is why there is no `POST /v1/pair` endpoint.

Today, issuing a token is a CLI operation:

```bash
$ dictation api pair "Caption overlay" --scope transcript:live --scope status:read
token id: token-9f1c2e7d4a6b8c05
scopes:   status:read, transcript:live
token:    kJ8v2Qx...        # shown once
```

```bash
$ dictation api tokens          # list, with state and last use
$ dictation api revoke token-9f1c2e7d4a6b8c05
```

Properties a client should rely on:

- **The plaintext token is shown once.** Only a SHA-256 hash is stored, so a
  lost token is re-issued, never recovered.
- **Revocation is immediate.** The next request returns `401`
  `unknown_or_revoked_token`.
- **Failed authentication is rate-limited** (10 attempts per minute by
  default, shared with pairing). Exceeding it returns `401`
  `too_many_authentication_attempts__try_again_shortly` — back off rather than
  retrying in a loop.

Store the token where the client stores its own secrets. Do not put it in a
shell history, a log file, or a URL.

In-process pairing, for a host application driving the flow itself:

```python
from dictation.api.tokens import Scope

request = app.tokens.begin_pairing("Caption overlay", {Scope.TRANSCRIPT_LIVE})
# show request.code to the user; they confirm in the UI
record, token = app.tokens.approve_pairing(request.code)
```

---

## 3. Scopes

| Scope | Grants |
| --- | --- |
| `status:read` | Capabilities and coarse recording state. Session lifecycle events. **No transcript text.** |
| `transcript:live` | `transcript.partial` events — revisable hypotheses while recording. |
| `transcript:final` | `transcript.final` events — the complete canonical transcript. |
| `session:control` | Start, stop and cancel recording. |

Scopes are independent by design: **reading captions does not grant permission
to start a microphone**, and controlling sessions does not grant transcript
text. Request the narrowest set that makes the client work.

`status:read` is required for the event stream itself, because the WebSocket
upgrade is authenticated with it. A caption client therefore holds
`status:read` + `transcript:live`.

Raw audio, recording history and stored artifacts are **not exposed at all** in
version 1, under any scope.

---

## 4. Request rules

Every request must:

- carry `Authorization: Bearer <token>`,
- send a `Host` header matching the loopback address the server bound to —
  `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`,
- send **no** `Origin` header,
- keep credentials out of the query string.

| Rule broken | Status | `reason` |
| --- | --- | --- |
| No `Authorization` header | 401 | `missing_bearer_token` |
| Unknown or revoked token | 401 | `unknown_or_revoked_token` |
| Too many attempts | 401 | `too_many_authentication_attempts__try_again_shortly` |
| Token lacks the scope | 403 | `scope_denied` |
| Any `Origin` header present | 403 | `browser_origin_denied` |
| `Host` is not the bound loopback address | 403 | `host_not_allowed` |
| `token` or `access_token` in the query string | 400 | `credentials_in_query_string` |

The `Host` check defends against DNS rebinding, where a page on a public
domain resolves to `127.0.0.1` and then talks to local services. The `Origin`
check means **browsers are not supported clients**: they cannot set
`Authorization` on a WebSocket handshake and always send `Origin`. A browser
integration needs its own handshake design, distinct permissions and its own
review — permissive CORS is not it.

### Error shape

Every failure returns the same body:

```json
{ "error": true, "reason": "scope_denied" }
```

`reason` is a stable, token-shaped code — safe to branch on, safe to log. It
never contains dictated text. Neither do the server's own logs: request lines
are not logged at all, and the structured event log refuses content-shaped
fields.

---

## 5. Endpoints

### `GET /v1/status`

Scope: `status:read`.

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/v1/status
```

```json
{
  "version": 1,
  "capabilities": ["session:control", "status:read"],
  "state": {
    "state": "idle",
    "recording": false,
    "locked": false,
    "elapsed_ms": 0,
    "session_id": "",
    "max_session_seconds": 600,
    "contribute": false,
    "duration_ms": 0,
    "asr_model": "whisper-base-q5_1",
    "privacy_model": "qwen3-4b-instruct-2507-q4",
    "partials_available": true
  }
}
```

| Field | Meaning |
| --- | --- |
| `version` | API contract version. `1` today. |
| `capabilities` | The scopes *this* token holds — use it to disable UI a client cannot drive. |
| `state.state` | `idle`, `starting`, `recording_held`, `recording_locked`, `finalizing`, `delivering`, `cancelled`, `error`. |
| `state.recording` / `locked` | Coarse booleans for an indicator. |
| `state.elapsed_ms` | Time since recording started. |
| `state.session_id` | Active session, or `""`. |
| `state.max_session_seconds` | The duration cap after which the app finalises by itself. |
| `state.contribute` | Whether this session is a contribution candidate. Never whether anything was uploaded. |
| `state.partials_available` | `false` when the ASR backend cannot stream; do not wait for `transcript.partial` events. |

There is no transcript text in `/v1/status`, in any state.

### `POST /v1/sessions`

Scope: `session:control`. Starts a **visible** recording — the indicator and
tray state change exactly as they would for a shortcut press.

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/v1/sessions
```

```json
{ "session_id": "session-4f2a91c0e3b7d854" }
```

- `201` on success.
- `409` `a_recording_is_already_active` when a session is already running. One
  session at a time; the API never interrupts a session the user started.
- `409` `recording_could_not_be_started` when capture failed — a denied
  microphone permission, a missing device, no capture backend.

Session IDs are random and non-sequential. They are not reused.

### `POST /v1/sessions/{id}/stop`

Scope: `session:control`. Requests finalisation: the transcript is completed
and delivered to whatever the user had focused when recording began.

```json
{ "session_id": "session-4f2a91c0e3b7d854", "accepted": true }
```

- Always `200`. `accepted` is `false` when that ID is not the active session —
  already finished, already cancelled, or never existed.
- **Idempotent.** Repeating the call does not deliver text twice.

### `POST /v1/sessions/{id}/cancel`

Scope: `session:control`. Cancels: no text is inserted anywhere, and no
training copy is built.

```json
{ "session_id": "session-4f2a91c0e3b7d854", "accepted": true }
```

- Always `200`, `accepted` as above. Idempotent.
- Cancellation is final. A finalisation already in flight still produces no
  insertion, and no artifact survives.
- Text a client already received over the event stream cannot be un-sent.
  Cancellation invalidates provisional output; it does not erase what was
  observed.

### Unknown routes

| Request | Status | `reason` |
| --- | --- | --- |
| `POST /v1/sessions/{id}/pause` | 404 | `unknown_action` |
| Anything else | 404 | `unknown_endpoint` |

---

## 6. Event stream: `GET /v1/events`

A WebSocket upgrade. Scope `status:read` authenticates the connection; each
event is then filtered by the scope it requires, so one connection delivers
exactly what the token permits.

### Handshake

```
GET /v1/events HTTP/1.1
Host: 127.0.0.1:8765
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: <16 random bytes, base64>
Sec-WebSocket-Version: 13
Authorization: Bearer <token>
```

| Problem | Status | `reason` |
| --- | --- | --- |
| Missing `Upgrade: websocket` | 400 | `websocket_upgrade_required` |
| Missing `Sec-WebSocket-Key` | 400 | `missing_websocket_key` |
| Missing or bad token | 401 | see §4 |
| Token without `status:read` | 403 | `scope_denied` |

On success: `101 Switching Protocols` with the usual
`Sec-WebSocket-Accept`. Frames are unmasked text frames from the server,
masked frames from the client, per RFC 6455.

### Envelope

```json
{
  "version": 1,
  "event": "transcript.partial",
  "session_id": "session-4f2a91c0e3b7d854",
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

| Field | Present on | Notes |
| --- | --- | --- |
| `version` | all | Contract version. |
| `event` | all | See the table below. |
| `session_id` | all | The session the event belongs to. |
| `seq` | all | Monotonic per connection. Gaps mean events were filtered by scope — not that anything was lost. |
| `segment_id`, `revision` | events that carry a segment | Identity plus revision of the hypothesis. |
| `start_ms`, `end_ms`, `text`, `is_final` | `transcript.*` | Timings are relative to the start of the recording. |
| `privacy` | `transcript.*` | Always `"unredacted"` in version 1. |
| `reason` | `error`, cancellations | A token-shaped code, never free text. |

### Events

| Event | Required scope | Meaning |
| --- | --- | --- |
| `session.started` | `status:read` | Recording began. |
| `transcript.partial` | `transcript:live` | A revisable hypothesis for one segment. |
| `transcript.final` | `transcript:final` | The complete canonical transcript. Supersedes all provisional segments. |
| `session.stopped` | `status:read` | The session finished. |
| `session.cancelled` | `status:read` | The session was cancelled; provisional output is invalid. |
| `error` | `status:read` | Something failed; see `reason`. |

### Handling revisions correctly

**Do not append partials.** A later `revision` for the same `segment_id`
*replaces* the earlier text for that segment. Keep a map from `segment_id` to
the highest revision seen, drop anything lower, and render segments ordered by
`start_ms`:

```python
segments: dict[str, tuple[int, int, str]] = {}   # id -> (revision, start_ms, text)

def apply(event):
    if event["event"] == "transcript.final":
        segments.clear()
        return event["text"]                      # supersedes everything
    if event["event"] != "transcript.partial":
        return None
    seen = segments.get(event["segment_id"])
    if seen and seen[0] >= event["revision"]:
        return None                               # stale, out-of-order delivery
    segments[event["segment_id"]] = (event["revision"], event["start_ms"], event["text"])
    return " ".join(text for _, _, text in sorted(segments.values(), key=lambda s: s[1]))
```

Out-of-order delivery is normal on a loaded machine. Applying a stale revision
makes displayed text visibly regress.

### Flow control and disconnects

Each connection has a bounded queue (`client_queue_limit`, 256 by default). A
client that stops reading is **disconnected**, not buffered indefinitely:

| Close code | Meaning | What to do |
| --- | --- | --- |
| `1008` | `slow_consumer_resync_required` | Reconnect and call `GET /v1/status`. The backlog is gone. |
| `1009` | `frame_too_large` | The client sent a frame over 4096 bytes. Send only close and pong frames. |
| `1000` | Normal close | Either side finished. |

Ping frames are answered with pong. There is **no transcript replay archive**:
a reconnecting client receives the current authorised snapshot from
`/v1/status` and events from that point on, never history.

### Minimal client

Any WebSocket library that can set request headers works. With
[`websocket-client`](https://pypi.org/project/websocket-client/):

```python
import json
import websocket

ws = websocket.create_connection(
    "ws://127.0.0.1:8765/v1/events",
    header=[f"Authorization: Bearer {TOKEN}"],
    suppress_origin=True,              # the server rejects any Origin header
)
try:
    while True:
        event = json.loads(ws.recv())
        if event["event"] == "transcript.final":
            print(event["text"])
finally:
    ws.close()
```

---

## 7. What this API deliberately does not do

- **No redacted stream.** `privacy` is always `unredacted`. Streaming
  redaction is not offered because later words can reveal that earlier text was
  sensitive, and post-session filtering cannot undo a disclosure already sent.
  A filtered stream would need distinct permissions, delay semantics and its
  own evaluation.
- **No raw audio, no history, no stored artifacts.** Not under any scope.
- **No control over contribution.** A client cannot opt in, opt out, or learn
  whether a session was uploaded. That lives in the desktop UI and
  `dictation consent`.
- **No promises about the receiving client.** Once text is delivered to a
  paired client, this application cannot control whether that client logs it,
  syncs it or forwards it to a remote service. The pairing screen says so, and
  so should any client asking for `transcript:live`.
- **No browser support.** See §4.

---

## 8. Versioning

`version` appears in the status response and in every event. Within version 1:

- new event types and new fields may be added — ignore what you do not know,
- existing field meanings, reason codes and status codes will not change,
- a new scope may appear; tokens keep the scopes they were issued with.

A breaking change becomes `/v2` with its own `version` value.

---

## 9. Embedding the core instead

If you are writing a host application in Python rather than a client over the
socket, the useful surfaces are:

| Class | Module | Purpose |
| --- | --- | --- |
| `Application` | `dictation.app` | Assembles paths, settings, store, consent, queue, tokens, registry |
| `CaptureCoordinator` | `dictation.capture.coordinator` | Sessions, delivery, the training copy; implements `SessionController` |
| `RecordingStateMachine` | `dictation.capture.state_machine` | Pure state machine, if you drive shortcuts yourself |
| `analyze` | `dictation.privacy.pipeline` | Rules + model union over a frozen transcript |
| `build_dataset` | `dictation.dataset.builder` | The privacy-filtered training copy |
| `UploadQueue`, `UploadWorker` | `dictation.upload` | Consent-gated queue and transfer |
| `TokenManager`, `EventBus` | `dictation.api` | Pairing and event fan-out without the HTTP server |

A coordinator only needs to satisfy four methods to be driven by this API —
`status`, `start_session`, `stop_session`, `cancel_session` — so a different
capture implementation can reuse the whole API layer. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the module map.
