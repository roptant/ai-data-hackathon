"""Local API: pairing, scopes, host and origin checks, events (plan section 8)."""

from __future__ import annotations

import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request

import pytest

from dictation.api import websocket as ws
from dictation.api.events import ApiEvent, EventBus, EventName
from dictation.api.server import LocalApiServer, wait_for_port
from dictation.api.tokens import Scope, TokenManager, hash_token, parse_scopes
from dictation.config import ApiSettings
from dictation.errors import ApiError, ScopeDenied, SlowConsumer, Unauthorized
from dictation.store.db import Database


@pytest.fixture
def tokens() -> TokenManager:
    return TokenManager(Database.in_memory())


class FakeController:
    """Minimal SessionController for API tests."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.cancelled: list[str] = []
        self.active = "session-1"
        self.conflict = False

    def status(self) -> dict[str, object]:
        return {"state": "idle", "recording": False, "elapsed_ms": 0}

    def start_session(self, *, source: str) -> str:
        if self.conflict:
            raise ApiError("a recording is already active")
        self.started.append(source)
        return self.active

    def stop_session(self, session_id: str) -> bool:
        self.stopped.append(session_id)
        return session_id == self.active

    def cancel_session(self, session_id: str) -> bool:
        self.cancelled.append(session_id)
        return session_id == self.active


# -- tokens ------------------------------------------------------------------


def test_only_the_hash_is_stored(tokens: TokenManager) -> None:
    record, token = tokens.issue("captions", frozenset({Scope.TRANSCRIPT_LIVE}))
    rows = tokens.db.tokens()
    assert rows[0]["token_hash"] == hash_token(token)
    assert token not in json.dumps([dict(row) for row in rows])
    assert record.token_id


def test_authentication_resolves_scopes(tokens: TokenManager) -> None:
    _, token = tokens.issue("captions", frozenset({Scope.TRANSCRIPT_LIVE, Scope.STATUS_READ}))
    record = tokens.authenticate(token)
    assert record.scopes == {Scope.TRANSCRIPT_LIVE, Scope.STATUS_READ}


def test_unknown_token_is_refused(tokens: TokenManager) -> None:
    with pytest.raises(Unauthorized):
        tokens.authenticate("not-a-token")


def test_missing_token_is_refused(tokens: TokenManager) -> None:
    with pytest.raises(Unauthorized):
        tokens.authenticate(None)


def test_revoked_token_stops_working(tokens: TokenManager) -> None:
    record, token = tokens.issue("captions", frozenset({Scope.STATUS_READ}))
    assert tokens.revoke(record.token_id)
    with pytest.raises(Unauthorized):
        tokens.authenticate(token)
    assert not tokens.revoke(record.token_id)


def test_reading_captions_does_not_grant_control(tokens: TokenManager) -> None:
    _, token = tokens.issue("captions", frozenset({Scope.TRANSCRIPT_LIVE}))
    record = tokens.authenticate(token)
    record.require(Scope.TRANSCRIPT_LIVE)
    with pytest.raises(ScopeDenied):
        record.require(Scope.SESSION_CONTROL)


def test_authentication_attempts_are_rate_limited() -> None:
    manager = TokenManager(Database.in_memory(), auth_attempts_per_minute=3)
    for _ in range(3):
        with pytest.raises(Unauthorized):
            manager.authenticate("wrong")
    with pytest.raises(Unauthorized, match="too many"):
        manager.authenticate("wrong")


def test_pairing_requires_user_approval(tokens: TokenManager) -> None:
    request = tokens.begin_pairing("captions", frozenset({Scope.TRANSCRIPT_LIVE}))
    assert request.code.isdigit() and len(request.code) == 6
    assert tokens.pending_pairings()
    record, token = tokens.approve_pairing(request.code)
    assert tokens.authenticate(token).token_id == record.token_id


def test_expired_pairing_code_is_refused(tokens: TokenManager) -> None:
    request = tokens.begin_pairing("captions", frozenset({Scope.STATUS_READ}), now=0.0)
    with pytest.raises(Unauthorized):
        tokens.approve_pairing(request.code, now=request.expires_at + 1)


def test_denied_pairing_issues_nothing(tokens: TokenManager) -> None:
    request = tokens.begin_pairing("captions", frozenset({Scope.STATUS_READ}))
    tokens.deny_pairing(request.code)
    with pytest.raises(Unauthorized):
        tokens.approve_pairing(request.code)


def test_unknown_scope_is_refused(tokens: TokenManager) -> None:
    assert parse_scopes("status:read,nonsense") == frozenset({Scope.STATUS_READ})


# -- event bus ---------------------------------------------------------------


def make_token(tokens: TokenManager, *scopes: Scope):
    record, _ = tokens.issue("client", frozenset(scopes))
    return record


def test_events_are_filtered_by_scope(tokens: TokenManager) -> None:
    bus = EventBus()
    status_only = bus.subscribe("a", make_token(tokens, Scope.STATUS_READ))
    live = bus.subscribe("b", make_token(tokens, Scope.STATUS_READ, Scope.TRANSCRIPT_LIVE))
    bus.emit(EventName.TRANSCRIPT_PARTIAL, "s1", text="hello", segment_id="seg-1", revision=1)
    assert status_only.drain() == ()
    assert len(live.drain()) == 1


def test_final_transcript_needs_its_own_scope(tokens: TokenManager) -> None:
    bus = EventBus()
    live = bus.subscribe("b", make_token(tokens, Scope.TRANSCRIPT_LIVE))
    bus.emit(EventName.TRANSCRIPT_FINAL, "s1", text="hello", is_final=True)
    assert live.drain() == ()


def test_sequence_numbers_increase(tokens: TokenManager) -> None:
    bus = EventBus()
    subscriber = bus.subscribe("a", make_token(tokens, Scope.STATUS_READ))
    bus.emit(EventName.SESSION_STARTED, "s1")
    bus.emit(EventName.SESSION_STOPPED, "s1")
    seqs = [event.seq for event in subscriber.drain()]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 2


def test_slow_consumer_is_disconnected_with_a_resync_error(tokens: TokenManager) -> None:
    bus = EventBus(queue_limit=2)
    subscriber = bus.subscribe("a", make_token(tokens, Scope.STATUS_READ))
    for _ in range(5):
        bus.emit(EventName.SESSION_STARTED, "s1")
    assert subscriber.disconnected
    subscriber.queue.clear()
    with pytest.raises(SlowConsumer):
        subscriber.next_event()
    assert bus.disconnect_stale() == ("a",)


def test_event_payload_matches_the_documented_envelope() -> None:
    event = ApiEvent(
        event=EventName.TRANSCRIPT_PARTIAL,
        session_id="random-session-id",
        seq=21,
        segment_id="segment-3",
        revision=4,
        start_ms=2600,
        end_ms=4100,
        text="The meeting starts tomorrow",
        is_final=False,
    )
    payload = event.payload()
    assert payload == {
        "version": 1,
        "event": "transcript.partial",
        "session_id": "random-session-id",
        "seq": 21,
        "segment_id": "segment-3",
        "revision": 4,
        "start_ms": 2600,
        "end_ms": 4100,
        "text": "The meeting starts tomorrow",
        "is_final": False,
        "privacy": "unredacted",
    }


def test_status_events_carry_no_text() -> None:
    event = ApiEvent(event=EventName.SESSION_STARTED, session_id="s1", seq=1)
    assert "text" not in event.payload()


# -- HTTP surface ------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def running_api(tokens: TokenManager):
    controller = FakeController()
    settings = ApiSettings(enabled=True, port=free_port(), client_queue_limit=8)
    server = LocalApiServer(tokens, EventBus(queue_limit=8), controller, settings)
    server.start()
    assert wait_for_port(server.port)
    try:
        yield server, controller
    finally:
        server.stop()


def request(
    server: LocalApiServer,
    path: str,
    *,
    token: str | None = None,
    method: str = "GET",
    host: str | None = None,
    origin: str | None = None,
):
    url = f"http://127.0.0.1:{server.port}{path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(url, headers=headers, method=method)
    if host:
        req.add_header("Host", host)
    if method == "POST":
        req.data = b"{}"
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode() or "{}")


def test_api_refuses_to_start_when_disabled(tokens: TokenManager) -> None:
    server = LocalApiServer(tokens, EventBus(), FakeController(), ApiSettings(enabled=False))
    with pytest.raises(ApiError):
        server.start()


def test_api_refuses_to_bind_all_interfaces(tokens: TokenManager) -> None:
    server = LocalApiServer(
        tokens, EventBus(), FakeController(), ApiSettings(enabled=True, host="0.0.0.0")
    )
    with pytest.raises(ApiError, match="refusing to bind"):
        server.start()


def test_status_requires_a_token(running_api) -> None:
    server, _ = running_api
    status, payload = request(server, "/v1/status")
    assert status == 401
    assert payload["reason"]


def test_status_returns_state_without_transcript(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("client", frozenset({Scope.STATUS_READ}))
    status, payload = request(server, "/v1/status", token=token)
    assert status == 200
    assert payload["version"] == 1
    assert "text" not in json.dumps(payload)


def test_browser_origin_is_denied(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("client", frozenset({Scope.STATUS_READ}))
    status, payload = request(
        server, "/v1/status", token=token, origin="https://evil.example.com"
    )
    assert status == 403
    assert payload["reason"] == "browser_origin_denied"


def test_foreign_host_header_is_denied(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("client", frozenset({Scope.STATUS_READ}))
    status, payload = request(server, "/v1/status", token=token, host="dictation.example.com")
    assert status == 403
    assert payload["reason"] == "host_not_allowed"


def test_credentials_in_the_query_string_are_refused(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("client", frozenset({Scope.STATUS_READ}))
    status, payload = request(server, f"/v1/status?token={token}", token=token)
    assert status == 400
    assert payload["reason"] == "credentials_in_query_string"


def test_starting_a_session_requires_control_scope(running_api, tokens: TokenManager) -> None:
    server, controller = running_api
    _, caption_token = tokens.issue("captions", frozenset({Scope.TRANSCRIPT_LIVE}))
    status, payload = request(server, "/v1/sessions", token=caption_token, method="POST")
    assert status == 403
    assert payload["reason"] == "scope_denied"
    assert controller.started == []


def test_control_scope_can_start_stop_and_cancel(running_api, tokens: TokenManager) -> None:
    server, controller = running_api
    _, token = tokens.issue("remote", frozenset({Scope.SESSION_CONTROL, Scope.STATUS_READ}))
    status, payload = request(server, "/v1/sessions", token=token, method="POST")
    assert status == 201
    session_id = payload["session_id"]
    status, payload = request(
        server, f"/v1/sessions/{session_id}/stop", token=token, method="POST"
    )
    assert status == 200 and payload["accepted"] is True
    status, payload = request(
        server, f"/v1/sessions/{session_id}/cancel", token=token, method="POST"
    )
    assert status == 200
    assert controller.stopped == [session_id]
    assert controller.cancelled == [session_id]


def test_repeated_stop_is_idempotent(running_api, tokens: TokenManager) -> None:
    server, controller = running_api
    _, token = tokens.issue("remote", frozenset({Scope.SESSION_CONTROL}))
    request(server, "/v1/sessions", token=token, method="POST")
    first = request(server, "/v1/sessions/session-1/stop", token=token, method="POST")
    second = request(server, "/v1/sessions/session-1/stop", token=token, method="POST")
    assert first[0] == second[0] == 200


def test_second_session_conflicts(running_api, tokens: TokenManager) -> None:
    server, controller = running_api
    controller.conflict = True
    _, token = tokens.issue("remote", frozenset({Scope.SESSION_CONTROL}))
    status, _ = request(server, "/v1/sessions", token=token, method="POST")
    assert status == 409


def test_unknown_endpoint_is_not_found(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("client", frozenset({Scope.STATUS_READ}))
    status, _ = request(server, "/v1/recordings", token=token)
    assert status == 404


def test_events_endpoint_requires_a_websocket_upgrade(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("client", frozenset({Scope.STATUS_READ}))
    status, payload = request(server, "/v1/events", token=token)
    assert status == 400
    assert payload["reason"] == "websocket_upgrade_required"


# -- websocket ---------------------------------------------------------------


def test_accept_key_matches_the_rfc_example() -> None:
    assert ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_frames_round_trip() -> None:
    payload = ws.encode_text("hello")
    masked = _mask(payload)
    frame = ws.parse_frame(masked)
    assert frame is not None
    assert frame.payload == b"hello"
    assert frame.opcode == ws.OPCODE_TEXT


def test_partial_frame_returns_none() -> None:
    assert ws.parse_frame(b"\x81") is None


def test_oversized_client_frame_is_refused() -> None:
    header = bytes([0x81, 0xFE]) + (ws.MAX_CLIENT_FRAME_BYTES + 1).to_bytes(2, "big")
    with pytest.raises(ws.FrameTooLarge):
        ws.parse_frame(header + b"\x00\x00\x00\x00")


def test_websocket_stream_delivers_permitted_events(running_api, tokens: TokenManager) -> None:
    server, _ = running_api
    _, token = tokens.issue("captions", frozenset({Scope.STATUS_READ, Scope.TRANSCRIPT_LIVE}))
    key = base64.b64encode(os.urandom(16)).decode()
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as client:
        client.sendall(
            (
                "GET /v1/events HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{server.port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                f"Authorization: Bearer {token}\r\n"
                "\r\n"
            ).encode()
        )
        handshake = client.recv(4096)
        assert b"101 Switching Protocols" in handshake
        assert ws.accept_key(key).encode() in handshake

        deadline = time.monotonic() + 5
        while server.context.bus.subscriber_count == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        server.emit(
            EventName.TRANSCRIPT_PARTIAL,
            "s1",
            segment_id="seg-1",
            revision=1,
            text="the meeting starts tomorrow",
        )
        client.settimeout(5)
        frame = ws.parse_frame(client.recv(4096))
        assert frame is not None
        message = json.loads(frame.payload.decode())
        assert message["event"] == "transcript.partial"
        assert message["privacy"] == "unredacted"
        client.sendall(_mask(ws.encode_close()))


def test_websocket_refuses_an_unauthenticated_subscriber(running_api) -> None:
    server, _ = running_api
    key = base64.b64encode(os.urandom(16)).decode()
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as client:
        client.sendall(
            (
                "GET /v1/events HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{server.port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "\r\n"
            ).encode()
        )
        response = client.recv(4096)
    assert b"401" in response
    assert b"101" not in response


def _mask(frame: bytes) -> bytes:
    """Re-encode a server frame as a masked client frame."""
    parsed = ws.parse_frame(frame)
    assert parsed is not None
    mask = os.urandom(4)
    payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(parsed.payload))
    header = bytearray([0x80 | parsed.opcode])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    else:
        header.append(0x80 | 126)
        header.extend(length.to_bytes(2, "big"))
    return bytes(header) + mask + payload
