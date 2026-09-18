"""Event envelopes and bounded per-client queues (plan section 8).

The envelope is the plan's: version, event, session id, sequence number,
segment id, revision, timings, text, finality and a ``privacy`` field that says
plainly that the text is unredacted.  A later revision *replaces* the earlier
text for that segment; nothing here lets a client append partials blindly and
be correct.

Each subscriber gets a bounded queue.  A client that cannot keep up is
disconnected with a resynchronisation error rather than being allowed to grow
the buffer, and there is no transcript replay archive: a reconnecting client
gets the current authorised snapshot, not history.

Streaming redaction is not offered.  Later words can reveal that earlier text
was sensitive, so a filtered stream would need distinct permissions, delay
semantics and its own evaluation.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from dictation.api.tokens import Scope, TokenRecord
from dictation.errors import SlowConsumer
from dictation.types import CANONICAL_SAMPLE_RATE
from dictation.version import API_VERSION


class EventName(StrEnum):
    SESSION_STARTED = "session.started"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    SESSION_STOPPED = "session.stopped"
    SESSION_CANCELLED = "session.cancelled"
    ERROR = "error"


#: Which scope each event requires.  Captions need no control permission.
EVENT_SCOPES: dict[EventName, Scope] = {
    EventName.SESSION_STARTED: Scope.STATUS_READ,
    EventName.TRANSCRIPT_PARTIAL: Scope.TRANSCRIPT_LIVE,
    EventName.TRANSCRIPT_FINAL: Scope.TRANSCRIPT_FINAL,
    EventName.SESSION_STOPPED: Scope.STATUS_READ,
    EventName.SESSION_CANCELLED: Scope.STATUS_READ,
    EventName.ERROR: Scope.STATUS_READ,
}


@dataclass(frozen=True, slots=True)
class ApiEvent:
    """One event as delivered to a permitted client."""

    event: EventName
    session_id: str
    seq: int
    version: int = API_VERSION
    segment_id: str = ""
    revision: int = 0
    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    is_final: bool = False
    #: Always ``unredacted`` in this version, stated rather than implied.
    privacy: str = "unredacted"
    reason: str = ""

    def payload(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": self.version,
            "event": str(self.event),
            "session_id": self.session_id,
            "seq": self.seq,
        }
        if self.segment_id:
            data["segment_id"] = self.segment_id
            data["revision"] = self.revision
        if self.event in {EventName.TRANSCRIPT_PARTIAL, EventName.TRANSCRIPT_FINAL}:
            data.update(
                {
                    "start_ms": self.start_ms,
                    "end_ms": self.end_ms,
                    "text": self.text,
                    "is_final": self.is_final,
                    "privacy": self.privacy,
                }
            )
        if self.reason:
            data["reason"] = self.reason
        return data

    def encode(self) -> str:
        return json.dumps(self.payload(), sort_keys=True)

    @property
    def required_scope(self) -> Scope:
        return EVENT_SCOPES[self.event]


def samples_to_ms(samples: int, sample_rate: int = CANONICAL_SAMPLE_RATE) -> int:
    return int(round(samples * 1000 / sample_rate))


@dataclass(slots=True)
class Subscriber:
    """One connected client with its own bounded queue."""

    subscriber_id: str
    token: TokenRecord
    limit: int = 256
    queue: deque[ApiEvent] = field(default_factory=deque)
    dropped: int = 0
    disconnected: bool = False
    reason: str = ""

    def permitted(self, event: ApiEvent) -> bool:
        return event.required_scope in self.token.scopes

    def offer(self, event: ApiEvent) -> bool:
        """Enqueue an event.  Overflow marks the subscriber for disconnection."""
        if self.disconnected:
            return False
        if not self.permitted(event):
            return False
        if len(self.queue) >= self.limit:
            self.dropped += 1
            self.disconnected = True
            self.reason = "slow_consumer_resync_required"
            return False
        self.queue.append(event)
        return True

    def drain(self) -> tuple[ApiEvent, ...]:
        drained = tuple(self.queue)
        self.queue.clear()
        return drained

    def next_event(self) -> ApiEvent | None:
        if self.queue:
            return self.queue.popleft()
        if self.disconnected:
            raise SlowConsumer(self.reason)
        return None


class EventBus:
    """Fan-out to subscribers, filtered by scope."""

    def __init__(self, *, queue_limit: int = 256) -> None:
        self.queue_limit = queue_limit
        self._subscribers: dict[str, Subscriber] = {}
        self._lock = threading.Lock()
        self._seq = 0

    def subscribe(self, subscriber_id: str, token: TokenRecord) -> Subscriber:
        subscriber = Subscriber(subscriber_id=subscriber_id, token=token, limit=self.queue_limit)
        with self._lock:
            self._subscribers[subscriber_id] = subscriber
        return subscriber

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscriber(self, subscriber_id: str) -> Subscriber | None:
        """Look up a live subscriber, or ``None`` once it has gone."""
        with self._lock:
            return self._subscribers.get(subscriber_id)

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def publish(self, event: ApiEvent) -> int:
        """Deliver to every permitted subscriber.  Returns the delivery count."""
        delivered = 0
        with self._lock:
            targets = list(self._subscribers.values())
        for subscriber in targets:
            if subscriber.offer(event):
                delivered += 1
        return delivered

    def emit(
        self,
        event: EventName,
        session_id: str,
        **fields: object,
    ) -> ApiEvent:
        api_event = ApiEvent(event=event, session_id=session_id, seq=self.next_seq(), **fields)  # type: ignore[arg-type]
        self.publish(api_event)
        return api_event

    def disconnect_stale(self, *, now: float | None = None) -> tuple[str, ...]:
        """Drop subscribers already marked as slow consumers."""
        _ = time.time() if now is None else now
        with self._lock:
            stale = [key for key, value in self._subscribers.items() if value.disconnected]
            for key in stale:
                del self._subscribers[key]
        return tuple(stale)
