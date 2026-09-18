"""Local real-time API (plan section 8).

A versioned loopback HTTP and WebSocket API for native clients.  It is disabled
until the user enables integration access, binds explicitly to the loopback
address, and authenticates every request with a paired token.

Rejections it performs on purpose:

* any ``Host`` header that is not the loopback address it bound to - a defence
  against DNS rebinding from a local web page,
* any ``Origin`` header at all, because browser integrations need a separately
  designed handshake rather than permissive CORS,
* credentials in the query string,
* requests without a bearer token, or with one lacking the scope,
* a second session while one is active (``409``),
* client frames larger than a control frame.

Errors and logs carry reason codes, never dictated text.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from dictation.api import websocket as ws
from dictation.api.events import ApiEvent, EventBus, EventName
from dictation.api.tokens import Scope, TokenManager, TokenRecord
from dictation.config import ApiSettings
from dictation.errors import ApiError, ScopeDenied, SlowConsumer, Unauthorized
from dictation.logging_ import events as event_log
from dictation.types import new_id
from dictation.version import API_VERSION

SERVER_NAME = "local-dictation"


@runtime_checkable
class SessionController(Protocol):
    """What the API is allowed to ask of the capture coordinator."""

    def status(self) -> dict[str, object]: ...

    def start_session(self, *, source: str) -> str:
        """Start a visible recording; returns the session ID."""

    def stop_session(self, session_id: str) -> bool:
        """Idempotent finalise request.  False when the ID is not active."""

    def cancel_session(self, session_id: str) -> bool:
        """Idempotent cancellation with no insertion or contribution."""


@dataclass(slots=True)
class ApiContext:
    """Everything the handler needs, injected rather than global."""

    tokens: TokenManager
    bus: EventBus
    controller: SessionController
    settings: ApiSettings
    bound_port: int = 0

    def allowed_hosts(self) -> frozenset[str]:
        port = self.bound_port or self.settings.port
        return frozenset(
            {
                f"{self.settings.host}:{port}",
                f"localhost:{port}",
                f"127.0.0.1:{port}",
                f"[::1]:{port}",
            }
        )


class _ThreadingApiServer(ThreadingHTTPServer):
    """Threading server that does not print tracebacks for dropped clients.

    A client disconnecting mid-frame is ordinary, and the default handler would
    print a traceback that may include the request line.
    """

    daemon_threads = True
    context: ApiContext

    def handle_error(self, request: object, client_address: object) -> None:
        event_log.warn("api.connection_error")


class _Handler(BaseHTTPRequestHandler):
    server_version = SERVER_NAME
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def context(self) -> ApiContext:
        return self.server.context  # type: ignore[attr-defined]

    # -- logging -------------------------------------------------------------

    def log_message(self, format: str, *args: object) -> None:
        """Suppress the default access log.

        Request lines can contain identifiers, and the structured event log is
        the only channel that is checked for content leakage.
        """
        return None

    # -- helpers -------------------------------------------------------------

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, reason: str) -> None:
        event_log.warn("api.request_refused", status=int(status), reason=reason)
        self._json(status, {"error": True, "reason": reason})

    def _check_origin(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        if host not in self.context.allowed_hosts():
            self._error(HTTPStatus.FORBIDDEN, "host_not_allowed")
            return False
        origin = self.headers.get("Origin")
        if origin:
            # No browser origin is trusted; a browser integration needs its own
            # handshake design (plan section 8).
            self._error(HTTPStatus.FORBIDDEN, "browser_origin_denied")
            return False
        return True

    def _authenticate(self, scope: Scope) -> TokenRecord | None:
        parsed = urlparse(self.path)
        if "token" in parsed.query or "access_token" in parsed.query:
            self._error(HTTPStatus.BAD_REQUEST, "credentials_in_query_string")
            return None
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        try:
            record = self.context.tokens.authenticate(token)
            record.require(scope)
        except ScopeDenied:
            self._error(HTTPStatus.FORBIDDEN, "scope_denied")
            return None
        except Unauthorized as error:
            self._error(HTTPStatus.UNAUTHORIZED, _reason(str(error)))
            return None
        return record

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if not self._check_origin():
            return
        path = urlparse(self.path).path
        if path == "/v1/status":
            token = self._authenticate(Scope.STATUS_READ)
            if token is None:
                return
            status = dict(self.context.controller.status())
            status.pop("transcript", None)  # never in status
            self._json(
                HTTPStatus.OK,
                {
                    "version": API_VERSION,
                    "capabilities": sorted(str(scope) for scope in token.scopes),
                    "state": status,
                },
            )
            return
        if path == "/v1/events":
            self._handle_events()
            return
        self._error(HTTPStatus.NOT_FOUND, "unknown_endpoint")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if not self._check_origin():
            return
        path = urlparse(self.path).path.rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(min(length, 4096))

        if path == "/v1/sessions":
            token = self._authenticate(Scope.SESSION_CONTROL)
            if token is None:
                return
            try:
                session_id = self.context.controller.start_session(source="api")
            except ApiError as error:
                self._error(HTTPStatus.CONFLICT, _reason(str(error)))
                return
            self._json(HTTPStatus.CREATED, {"session_id": session_id})
            return

        parts = path.split("/")
        if len(parts) == 5 and parts[1] == "v1" and parts[2] == "sessions":
            session_id, action = parts[3], parts[4]
            token = self._authenticate(Scope.SESSION_CONTROL)
            if token is None:
                return
            if action == "stop":
                accepted = self.context.controller.stop_session(session_id)
            elif action == "cancel":
                accepted = self.context.controller.cancel_session(session_id)
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown_action")
                return
            # Idempotent: repeating the request is not an error, and repeating
            # a stop must not deliver text twice.
            self._json(HTTPStatus.OK, {"session_id": session_id, "accepted": accepted})
            return

        self._error(HTTPStatus.NOT_FOUND, "unknown_endpoint")

    # -- event stream --------------------------------------------------------

    def _handle_events(self) -> None:
        if (self.headers.get("Upgrade") or "").lower() != "websocket":
            self._error(HTTPStatus.BAD_REQUEST, "websocket_upgrade_required")
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._error(HTTPStatus.BAD_REQUEST, "missing_websocket_key")
            return
        token = self._authenticate(Scope.STATUS_READ)
        if token is None:
            return

        self.wfile.write(ws.handshake_response(key))
        self.wfile.flush()
        subscriber_id = new_id("subscriber")
        subscriber = self.context.bus.subscribe(subscriber_id, token)
        event_log.emit(
            "api.subscriber_connected",
            subscriber=subscriber_id,
            token_id=token.token_id,
            scopes=sorted(str(scope) for scope in token.scopes),
        )
        try:
            self._pump(subscriber_id)
        finally:
            self.context.bus.unsubscribe(subscriber_id)
            event_log.emit("api.subscriber_disconnected", subscriber=subscriber_id)

    def _pump(self, subscriber_id: str) -> None:
        connection: socket.socket = self.connection
        connection.settimeout(0.2)
        bus = self.context.bus
        pending = bytearray()
        while True:
            subscriber = bus.subscriber(subscriber_id)
            if subscriber is None:
                return
            try:
                event = subscriber.next_event()
            except SlowConsumer as error:
                self._send(ws.encode_close(ws.CLOSE_POLICY_VIOLATION, _reason(str(error))))
                return
            if event is not None:
                if not self._send(ws.encode_text(event.encode())):
                    return
                continue
            # Nothing queued: read for a close frame, then idle briefly.
            try:
                chunk = connection.recv(1024)
                if not chunk:
                    return
                pending.extend(chunk)
                try:
                    frame = ws.parse_frame(bytes(pending))
                except ws.FrameTooLarge:
                    self._send(ws.encode_close(ws.CLOSE_TOO_LARGE, "frame_too_large"))
                    return
                if frame is not None:
                    del pending[: frame.consumed]
                    if frame.is_close:
                        self._send(ws.encode_close())
                        return
                    if frame.opcode == ws.OPCODE_PING:
                        self._send(ws.encode_frame(frame.payload, ws.OPCODE_PONG))
            except TimeoutError:
                pass
            except OSError:
                return

    def _send(self, payload: bytes) -> bool:
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except OSError:
            return False
        return True


def _reason(message: str) -> str:
    """Reduce an exception message to a token-shaped reason code."""
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in message.strip().lower()
    )
    return cleaned[:64] or "error"


class LocalApiServer:
    """Threaded loopback server, started only when integration is enabled."""

    def __init__(
        self,
        tokens: TokenManager,
        bus: EventBus,
        controller: SessionController,
        settings: ApiSettings | None = None,
    ) -> None:
        self.settings = settings or ApiSettings()
        self.context = ApiContext(
            tokens=tokens, bus=bus, controller=controller, settings=self.settings
        )
        self._httpd: _ThreadingApiServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.context.bound_port

    @property
    def running(self) -> bool:
        return self._httpd is not None

    def start(self) -> int:
        """Bind and serve.  Refuses unless integration access is enabled."""
        if not self.settings.enabled:
            raise ApiError(
                "the local API is disabled; enable integration access in settings first"
            )
        if self.settings.host not in {"127.0.0.1", "localhost", "::1"}:
            # Never all interfaces: the API would be reachable from the network.
            raise ApiError(f"refusing to bind the local API to {self.settings.host}")
        if self._httpd is not None:
            return self.context.bound_port

        httpd = _ThreadingApiServer((self.settings.host, self.settings.port), _Handler)
        httpd.context = self.context
        self.context.bound_port = httpd.server_address[1]
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, name="dictation-api", daemon=True)
        self._thread.start()
        event_log.emit("api.started", port=self.context.bound_port)
        return self.context.bound_port

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        event_log.emit("api.stopped")

    # -- publishing ----------------------------------------------------------

    def publish(self, event: ApiEvent) -> int:
        return self.context.bus.publish(event)

    def emit(self, event: EventName, session_id: str, **fields: object) -> ApiEvent:
        return self.context.bus.emit(event, session_id, **fields)

    def __enter__(self) -> LocalApiServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def wait_for_port(port: int, *, host: str = "127.0.0.1", timeout: float = 5.0) -> bool:
    """Poll until the API accepts connections.  Used by the CLI and tests."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False
