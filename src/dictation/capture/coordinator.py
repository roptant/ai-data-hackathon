"""Capture coordinator (plan sections 3, 5 and 6).

This is where the two paths of the plan meet and stay separate:

* **The user's path** runs first and is never blocked by privacy work.  Audio
  is captured into a bounded buffer, transcribed, and the *full* transcript is
  delivered to the focused application.
* **The training path** runs afterwards, on a separate copy, only for a session
  that consent allows.  Its failure never affects the first path, and it can be
  paused or preempted by a new dictation.

The coordinator owns the state machine, the audio device lifecycle, session
IDs, cancellation and the resource limits, and it is the object the local API
controls - through the narrow
:class:`~dictation.api.server.SessionController` surface, not by reaching into
these internals.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from dictation.api.events import EventBus, EventName, samples_to_ms
from dictation.asr.base import AsrWorker
from dictation.asr.reconciler import PartialReconciler
from dictation.capture.audio_buffer import BoundedPcmBuffer
from dictation.capture.devices import AudioSource, MemorySource
from dictation.capture.state_machine import (
    Effect,
    Event,
    Outcome,
    RecordingStateMachine,
    State,
)
from dictation.capture.vad import EnergyVad
from dictation.config import Settings
from dictation.consent.consent import ConsentManager
from dictation.dataset.builder import BuildResult, build_dataset, make_recheck
from dictation.dataset.package import build_package
from dictation.errors import (
    ApiError,
    AsrUnavailable,
    BufferOverflow,
    CaptureError,
    CapabilityUnavailable,
    DictationError,
)
from dictation.logging_ import events as event_log
from dictation.platform_.base import FocusTarget, PlatformAdapter
from dictation.platform_.insertion import DeliveryResult, TextDelivery
from dictation.privacy import rules
from dictation.privacy.classifier.base import PrivacyClassifier
from dictation.privacy.pipeline import analyze
from dictation.store.retention import RetentionPolicy
from dictation.store.session_store import ArtifactKind, SessionStore
from dictation.types import CANONICAL_SAMPLE_WIDTH, Decision, Transcript, new_id
from dictation.upload.queue import UploadQueue

SourceFactory = Callable[[], AudioSource]

#: Samples read per device poll.  20 ms at 16 kHz.
CHUNK_SAMPLES = 320


@dataclass(slots=True)
class SessionRuntime:
    """Live state for one recording."""

    session_id: str
    started_at: float
    source: AudioSource
    buffer: BoundedPcmBuffer
    target: FocusTarget
    contribute: bool
    reconciler: PartialReconciler = field(default_factory=PartialReconciler)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: str = ""
    opted_out: bool = False

    @property
    def duration_ms(self) -> int:
        return self.buffer.duration_ms


@dataclass(slots=True)
class SessionOutcome:
    """What happened to one finished session."""

    session_id: str
    transcript: Transcript | None = None
    delivery: DeliveryResult | None = None
    training: BuildResult | None = None
    training_reason: str = ""
    error: str = ""

    @property
    def text(self) -> str:
        return self.transcript.display_text() if self.transcript else ""


class CaptureCoordinator:
    """Drives recording, delivery and the asynchronous training copy."""

    def __init__(
        self,
        *,
        settings: Settings,
        asr: AsrWorker,
        delivery: TextDelivery,
        adapter: PlatformAdapter,
        classifier: PrivacyClassifier | None = None,
        bus: EventBus | None = None,
        store: SessionStore | None = None,
        queue: UploadQueue | None = None,
        consent: ConsentManager | None = None,
        source_factory: SourceFactory | None = None,
        retention: RetentionPolicy | None = None,
        process_training_inline: bool = True,
    ) -> None:
        self.settings = settings
        self.asr = asr
        self.delivery = delivery
        self.adapter = adapter
        self.classifier = classifier
        self.bus = bus or EventBus(queue_limit=settings.api.client_queue_limit)
        self.store = store
        self.queue = queue
        self.consent = consent
        self.retention = retention or RetentionPolicy(settings.retention)
        self.source_factory = source_factory or (lambda: MemorySource(b""))
        self.process_training_inline = process_training_inline
        self.machine = RecordingStateMachine(
            max_session_seconds=settings.capture.max_session_seconds,
            warn_before_limit_seconds=settings.capture.warn_before_limit_seconds,
        )
        self.vad = EnergyVad(sample_rate=settings.capture.sample_rate)
        self._lock = threading.RLock()
        self._runtime: SessionRuntime | None = None
        self._last_outcome: SessionOutcome | None = None

    # -- status --------------------------------------------------------------

    def status(self) -> dict[str, object]:
        """Coarse state for the indicator and the local API.  No transcript."""
        with self._lock:
            snapshot = self.machine.snapshot()
            runtime = self._runtime
            snapshot.update(
                {
                    "contribute": bool(runtime.contribute) if runtime else False,
                    "duration_ms": runtime.duration_ms if runtime else 0,
                    "asr_model": self.asr.capabilities.model_id,
                    "privacy_model": (
                        self.classifier.capabilities.model_id if self.classifier else ""
                    ),
                    "partials_available": self.asr.capabilities.streaming,
                }
            )
            return snapshot

    @property
    def last_outcome(self) -> SessionOutcome | None:
        return self._last_outcome

    # -- shortcut entry points ----------------------------------------------

    def on_hold_down(self) -> str | None:
        return self._begin(Event.HOLD_DOWN, source="shortcut_hold")

    def on_toggle(self) -> str | None:
        with self._lock:
            recording = self.machine.is_recording
            session_id = self.machine.session_id
            outcome = self.machine.handle(Event.TOGGLE) if recording else None
        if outcome is not None:
            self._handle(outcome)
            return session_id
        return self._begin(Event.TOGGLE, source="shortcut_toggle")

    def on_hold_up(self) -> None:
        self._dispatch(Event.HOLD_UP)

    def on_lock(self) -> None:
        self._dispatch(Event.LOCK)

    def on_cancel(self) -> None:
        self._dispatch(Event.CANCEL)

    def _dispatch(self, event: Event) -> Outcome:
        """Feed one event under the lock, then run its effects without it.

        Effects must not hold the lock: finalisation joins the capture thread,
        and that thread takes the lock on every duration-limit tick.
        """
        with self._lock:
            outcome = self.machine.handle(event)
        self._handle(outcome)
        return outcome

    # -- SessionController surface ------------------------------------------

    def start_session(self, *, source: str = "api", locked: bool = True) -> str:
        """Start a *visible* recording.  Conflicts if one is already active."""
        with self._lock:
            if self.machine.is_active:
                raise ApiError("a recording is already active")
        session_id = self._begin(Event.TOGGLE if locked else Event.HOLD_DOWN, source=source)
        if session_id is None:
            raise ApiError("recording could not be started")
        return session_id

    def stop_session(self, session_id: str) -> bool:
        """Idempotent finalise.  False when that session is not active."""
        with self._lock:
            if self.machine.session_id != session_id:
                return False
            outcome = self.machine.handle(Event.STOP)
        if outcome.ignored:
            return False
        self._handle(outcome)
        return True

    def cancel_session(self, session_id: str) -> bool:
        with self._lock:
            if self.machine.session_id != session_id:
                return False
            outcome = self.machine.handle(Event.CANCEL)
        if outcome.ignored:
            return False
        self._handle(outcome)
        return True

    # -- lifecycle -----------------------------------------------------------

    def _begin(self, event: Event, *, source: str) -> str | None:
        session_id = new_id("session")
        with self._lock:
            outcome = self.machine.handle(event, session_id=session_id)
            if outcome.ignored:
                event_log.emit("capture.command_ignored", reason=outcome.ignored_reason)
                return None
        try:
            self._open_session(session_id, source=source)
        except (CaptureError, CapabilityUnavailable) as error:
            with self._lock:
                self._runtime = None
                failed = self.machine.handle(Event.FAILED)
            self._handle(failed)
            with self._lock:
                self.machine.handle(Event.RESET)
            event_log.error("capture.device_failed", session=session_id, reason=_code(error))
            self.bus.emit(
                EventName.ERROR, session_id, reason=_code(error)
            )
            return None
        self._dispatch(Event.CAPTURE_STARTED)
        return session_id

    def _open_session(self, session_id: str, *, source: str) -> None:
        capture_settings = self.settings.capture
        audio_source = self.source_factory()
        audio_source.open()
        target = self.adapter.current_focus()
        contribute = self._contribution_allowed()
        runtime = SessionRuntime(
            session_id=session_id,
            started_at=time.time(),
            source=audio_source,
            buffer=BoundedPcmBuffer.for_seconds(
                capture_settings.max_session_seconds, capture_settings.sample_rate
            ),
            target=target,
            contribute=contribute,
        )
        with self._lock:
            self._runtime = runtime
        if self.store is not None:
            now = runtime.started_at
            self.store.db.insert_session(
                session_id,
                started_at=now,
                expires_at=self.retention.session_expiry(now=now),
                language=self.settings.language,
                contribute=contribute,
            )
        self.bus.emit(EventName.SESSION_STARTED, session_id)
        event_log.emit(
            "capture.started",
            session=session_id,
            source=source,
            contribute=contribute,
        )
        runtime.thread = threading.Thread(
            target=self._capture_loop, args=(runtime,), name="dictation-capture", daemon=True
        )
        runtime.thread.start()

    def _capture_loop(self, runtime: SessionRuntime) -> None:
        """Read the device until stop, cancellation or the duration limit."""
        while not runtime.stop_requested.is_set():
            try:
                chunk = runtime.source.read_chunk(CHUNK_SAMPLES)
            except CaptureError as error:
                runtime.error = _code(error)
                break
            if chunk:
                try:
                    runtime.buffer.append(chunk)
                except BufferOverflow:
                    # Explicit failure: never silently drop captured audio.
                    runtime.error = "buffer_overflow"
                    break
            else:
                time.sleep(0.005)
            with self._lock:
                tick = self.machine.tick()
            if Effect.WARN_DURATION_LIMIT in tick.effects:
                event_log.warn("capture.duration_warning", session=runtime.session_id)
            if tick.accepted and Effect.FINALIZE_TRANSCRIPT in tick.effects:
                # The duration limit fired: finalise rather than record forever.
                threading.Thread(
                    target=self._finalize, args=(runtime,), name="dictation-finalize", daemon=True
                ).start()
                return
        if runtime.error:
            self._dispatch(Event.FAILED)

    # -- effects -------------------------------------------------------------

    def _handle(self, outcome: Outcome) -> None:
        for effect in outcome.effects:
            match effect:
                case Effect.STOP_CAPTURE:
                    self._stop_capture()
                case Effect.FINALIZE_TRANSCRIPT:
                    runtime = self._runtime
                    if runtime is not None:
                        self._finalize(runtime)
                case Effect.DISCARD_SESSION:
                    self._discard()
                case Effect.RELEASE_RESOURCES:
                    self._release()
                case _:
                    pass

    def _stop_capture(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.stop_requested.set()
        if runtime.thread is not None and runtime.thread is not threading.current_thread():
            runtime.thread.join(timeout=2.0)
        try:
            runtime.source.close()
        except Exception:  # noqa: BLE001 - closing must not mask the outcome
            pass

    def _discard(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.buffer.clear()
        if self.store is not None:
            self.store.delete_session(runtime.session_id)
        self.bus.emit(EventName.SESSION_CANCELLED, runtime.session_id)
        event_log.emit("capture.cancelled", session=runtime.session_id)
        self._runtime = None

    def _release(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            runtime.buffer.clear()
        self._runtime = None
        with self._lock:
            if self.machine.state in (State.CANCELLED, State.ERROR):
                self.machine.handle(Event.RESET)

    # -- finalisation --------------------------------------------------------

    def _finalize(self, runtime: SessionRuntime) -> SessionOutcome:
        """Transcribe, deliver, then hand the copy to the training path."""
        self._stop_capture()
        outcome = SessionOutcome(session_id=runtime.session_id)
        pcm = runtime.buffer.read()

        if self.vad.is_effectively_silent(pcm, min_ratio=self.settings.capture.min_speech_ratio):
            # A silent session must not become hallucinated text or an example.
            event_log.emit("capture.silent_session", session=runtime.session_id)
            outcome.training_reason = "silent_session"
            self._complete(runtime, outcome, delivered=False)
            return outcome

        try:
            transcript = self.asr.transcribe(
                pcm,
                session_id=runtime.session_id,
                sample_rate=self.settings.capture.sample_rate,
                language=self.settings.language,
            )
        except AsrUnavailable as error:
            outcome.error = _code(error)
            self.bus.emit(EventName.ERROR, runtime.session_id, reason=outcome.error)
            self._dispatch(Event.FAILED)
            with self._lock:
                if self.machine.state in (State.CANCELLED, State.ERROR):
                    self.machine.handle(Event.RESET)
            self._last_outcome = outcome
            return outcome

        outcome.transcript = transcript
        with self._lock:
            finalized = self.machine.handle(Event.FINALIZED)
        cancelled = not self.machine.insertion_permitted

        self.bus.emit(
            EventName.TRANSCRIPT_FINAL,
            runtime.session_id,
            segment_id="final",
            revision=transcript.revision,
            start_ms=0,
            end_ms=samples_to_ms(transcript.span.end, transcript.sample_rate),
            text=transcript.display_text(),
            is_final=True,
        )

        if Effect.DELIVER_TEXT in finalized.effects and not cancelled:
            outcome.delivery = self.delivery.deliver(
                transcript.display_text(),
                runtime.target,
                session_id=runtime.session_id,
                cancelled=False,
            )
            with self._lock:
                self._handle(self.machine.handle(Event.DELIVERED))
        else:
            self._discard()

        if runtime.contribute and not cancelled and not runtime.opted_out:
            outcome.training = self._training_copy(runtime, transcript, pcm)
            if outcome.training is not None:
                outcome.training_reason = outcome.training.reason
        else:
            outcome.training_reason = "contribution_off"

        self._complete(runtime, outcome, delivered=True)
        return outcome

    def _complete(self, runtime: SessionRuntime, outcome: SessionOutcome, *, delivered: bool) -> None:
        if self.store is not None:
            self.store.db.finish_session(
                runtime.session_id,
                finished_at=time.time(),
                duration_ms=runtime.duration_ms,
                word_count=len(outcome.transcript) if outcome.transcript else 0,
            )
        self.bus.emit(EventName.SESSION_STOPPED, runtime.session_id)
        runtime.buffer.clear()
        with self._lock:
            if self.machine.state in (State.CANCELLED, State.ERROR):
                self.machine.handle(Event.RESET)
            if self.machine.state is State.DELIVERING:
                self.machine.handle(Event.DELIVERED)
            self._runtime = None
        self._last_outcome = outcome
        event_log.emit(
            "capture.finished",
            session=runtime.session_id,
            duration_ms=runtime.duration_ms,
            delivered=delivered,
            training=outcome.training_reason or "none",
        )

    # -- training path -------------------------------------------------------

    def _contribution_allowed(self) -> bool:
        """Whether this session may even be considered for contribution."""
        if self.consent is None or self.queue is None or self.store is None:
            return False
        decision = self.consent.check_session(
            session_started_at=time.time(), language=self.settings.language
        )
        return bool(decision)

    def _training_copy(
        self,
        runtime: SessionRuntime,
        transcript: Transcript,
        pcm: bytes,
    ) -> BuildResult | None:
        """Build the privacy-filtered training copy for one session.

        Runs after delivery.  Every failure path here ends with the artifact
        deleted and the job rejected; none of them can affect the text the user
        already received.
        """
        if self.queue is None or self.store is None or self.consent is None:
            return None
        if self.classifier is None:
            event_log.warn("training.no_classifier", session=runtime.session_id)
            return None

        enqueued = self.queue.enqueue(
            runtime.session_id,
            session_started_at=runtime.started_at,
            session_opted_out=runtime.opted_out,
            language=transcript.language,
        )
        if not enqueued.accepted or enqueued.job is None:
            return None
        job_id = enqueued.job.job_id

        try:
            self.store.put(
                runtime.session_id,
                ArtifactKind.RAW_AUDIO,
                pcm,
                expires_at=self.retention.expiry_for(ArtifactKind.RAW_AUDIO, now=time.time()),
            )
            self.queue.start_analysis(job_id)
            analysis = analyze(
                transcript,
                self.classifier,
                settings=self.settings.privacy,
                private_terms=self.settings.private_terms,
            )
            self.queue.start_build(job_id)
            result = build_dataset(
                analysis,
                pcm,
                quality_settings=self.settings.quality,
                privacy_settings=self.settings.privacy,
                seen_hashes=self.store.db.contributed_hashes(),
                vad=self.vad,
                recheck=make_recheck(self._recheck_clean),
            )
            if result.decision is not Decision.ELIGIBLE:
                self.queue.reject(job_id, result.reason or "rejected")
                return result

            consent_record = self.consent.active()
            if consent_record is None:
                self.queue.reject(job_id, "no_active_consent")
                return result
            package = build_package(
                result,
                consent_version=consent_record.version,
                consent_reference=consent_record.consent_id,
            )
            self.queue.mark_eligible(job_id, package, content_hash=result.content_hash)
            return result
        except DictationError as error:
            self.queue.reject(job_id, _code(error))
            return None
        except Exception:  # noqa: BLE001 - nothing may fall through as clean
            self.queue.reject(job_id, "training_pipeline_error")
            raise

    def _recheck_clean(self, transcript: Transcript) -> tuple[bool, str]:
        """Post-removal recheck: rules always, the model when it is loaded."""
        findings = rules.detect(transcript, private_terms=self.settings.private_terms)
        if findings.injection_suspected:
            return False, "injection_after_removal"
        if findings.spans:
            return False, "sensitive_after_removal"
        if self.classifier is None:
            return True, ""
        analysis = analyze(
            transcript,
            self.classifier,
            settings=self.settings.privacy,
            private_terms=self.settings.private_terms,
        )
        if analysis.rejected_reason:
            return False, analysis.rejected_reason
        if analysis.uncertain:
            return False, "uncertain_after_removal"
        if analysis.spans:
            return False, "sensitive_after_removal"
        return True, ""

    # -- synchronous helper for scripted use --------------------------------

    def run_session(
        self,
        source: AudioSource,
        *,
        locked: bool = True,
        opted_out: bool = False,
        timeout_s: float = 30.0,
    ) -> SessionOutcome:
        """Record one whole source to completion.

        Used by the CLI demo, the evaluation harness and the tests: it starts a
        session, waits for the source to drain, then stops and returns the
        outcome.
        """
        self.source_factory = lambda: source
        session_id = self.start_session(source="scripted", locked=locked)
        runtime = self._runtime
        if runtime is not None:
            runtime.opted_out = opted_out
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = self._runtime
            if current is None:
                break
            if current.error:
                break
            # The scripted sources are finite; wait until the buffer stops growing.
            before = current.buffer.sample_count
            time.sleep(0.02)
            if current.buffer.sample_count == before and before > 0:
                break
        self.stop_session(session_id)
        outcome = self._last_outcome
        if outcome is None:  # pragma: no cover - defensive
            return SessionOutcome(session_id=session_id, error="no_outcome")
        return outcome


def _code(error: Exception) -> str:
    """Token-shaped reason code from an exception, safe for the event log."""
    name = type(error).__name__
    return "".join(
        character if character.isalnum() or character == "_" else "_" for character in name
    ).lower()


def pcm_seconds(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / CANONICAL_SAMPLE_WIDTH / sample_rate
