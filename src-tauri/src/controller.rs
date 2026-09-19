//! Serialized recording control (plan §5).
//!
//! Every input — shortcuts, UI buttons, API calls, lifecycle signals, and
//! worker results — arrives as a [`Command`] on one channel, so conflicting
//! commands are applied one at a time. The pure state machine decides; this
//! loop performs its effects. ASR results carry an epoch, and a result for an
//! older or cancelled session is discarded.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        mpsc::{Receiver, RecvTimeoutError, Sender},
    },
    time::{Duration, Instant},
};

use dictation_api::{ApiEvent, ControlError, EventKind};
use dictation_core::{
    asr::{LiveReconciler, LiveSettings, RecognizedSegment, display_text, is_hallucination_risk},
    platform::{LifecycleEvent, lifecycle_action},
    recording::{Effect, Event, RecordingStateMachine, State},
    shortcuts::{Interpreted, ShortcutEvent, ShortcutInterpreter},
    vad::{SessionActivity, VadSettings},
};
use dictation_platform::{desktop::FocusSnapshot, microphone::MicrophoneCapture};
use tauri::Emitter;

use crate::{
    asr_thread::{AsrHandle, AsrRequest, AsrResult},
    delivery, state::Shared, training_copy,
};

pub enum UiAction {
    Toggle,
    Stop,
    Cancel,
    Lock,
}

pub enum ApiRequest {
    Start,
    Stop(String),
    Cancel(String),
}

pub enum Command {
    Shortcut(ShortcutEvent),
    Ui(UiAction),
    Api(ApiRequest, Sender<Result<String, ControlError>>),
    Lifecycle(LifecycleEvent),
    AsrPartial { epoch: u64, result: AsrResult, offset: u64, audio_end: u64 },
    AsrFinal { epoch: u64, result: AsrResult },
    Shutdown,
}

/// Partial passes start once this much new audio has arrived.
const PARTIAL_INTERVAL_SAMPLES: u64 = 16_000;
/// How long an error stays visible before returning to idle.
const ERROR_DISPLAY_MS: u64 = 2_500;

struct Active {
    session_id: String,
    epoch: u64,
    started_unix: i64,
    focus: Option<FocusSnapshot>,
    capture: Option<MicrophoneCapture>,
    pcm: Vec<i16>,
    live: LiveReconciler,
    partial_in_flight: bool,
    last_pass_end: u64,
    final_text: Option<String>,
    final_segments: Vec<RecognizedSegment>,
    language: String,
}

pub struct Controller {
    shared: Arc<Shared>,
    asr: AsrHandle,
    machine: RecordingStateMachine,
    interpreter: ShortcutInterpreter,
    active: Option<Active>,
    epoch: u64,
    clock: Instant,
    error_since: Option<u64>,
    last_publish_ms: u64,
    seq: u64,
}

impl Controller {
    pub fn new(shared: Arc<Shared>, asr: AsrHandle) -> Self {
        let limit = shared.settings().max_session_seconds;
        Self {
            shared,
            asr,
            machine: RecordingStateMachine::new(limit, 30),
            interpreter: ShortcutInterpreter::default(),
            active: None,
            epoch: 0,
            clock: Instant::now(),
            error_since: None,
            last_publish_ms: 0,
            seq: 0,
        }
    }

    fn now_ms(&self) -> u64 {
        u64::try_from(self.clock.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    pub fn run(mut self, receiver: &Receiver<Command>) {
        loop {
            match receiver.recv_timeout(Duration::from_millis(40)) {
                Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                    self.apply(Event::Cancel);
                    return;
                }
                Ok(command) => self.handle(command),
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.tick();
        }
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Shortcut(event) => match self.interpreter.handle(event, self.now_ms()) {
                Interpreted::Apply(event) => self.apply(event),
                Interpreted::DeferHoldUp { .. } | Interpreted::Ignore => {}
            },
            Command::Ui(action) => self.apply(match action {
                UiAction::Toggle => Event::Toggle,
                UiAction::Stop => Event::Stop,
                UiAction::Cancel => Event::Cancel,
                UiAction::Lock => Event::Lock,
            }),
            Command::Api(request, reply) => {
                let _ = reply.send(self.api(request));
            }
            Command::Lifecycle(event) => {
                let dictation_core::platform::LifecycleAction::CancelWithoutDelivery = lifecycle_action(event);
                if self.machine.state().is_active() {
                    self.notice(match event {
                        LifecycleEvent::ScreenLocked => "cancelled_screen_locked",
                        LifecycleEvent::SystemSleeping => "cancelled_system_sleep",
                        LifecycleEvent::InputDeviceRemoved => "cancelled_device_removed",
                        LifecycleEvent::MicrophonePermissionRevoked => "cancelled_permission_revoked",
                        LifecycleEvent::ProcessShuttingDown => "cancelled_shutdown",
                    });
                }
                self.apply(Event::Cancel);
            }
            Command::AsrPartial { epoch, result, offset, audio_end } => self.on_partial(epoch, result, offset, audio_end),
            Command::AsrFinal { epoch, result } => self.on_final(epoch, result),
            Command::Shutdown => {}
        }
    }

    fn api(&mut self, request: ApiRequest) -> Result<String, ControlError> {
        let active_id = self.active.as_ref().map(|active| active.session_id.clone());
        match request {
            ApiRequest::Start => {
                if self.machine.state().is_active() {
                    return Err(ControlError::Conflict);
                }
                self.apply(Event::Toggle);
                self.active
                    .as_ref()
                    .map(|active| active.session_id.clone())
                    .ok_or(ControlError::Unavailable("microphone_unavailable"))
            }
            ApiRequest::Stop(id) | ApiRequest::Cancel(id) if active_id.as_deref() != Some(id.as_str()) => {
                // Idempotent: repeating a stop/cancel for the session that just
                // ended succeeds; an unknown session does not.
                if self.shared.status().session_id.as_deref() == Some(id.as_str()) {
                    Ok(id)
                } else {
                    Err(ControlError::NotFound)
                }
            }
            ApiRequest::Stop(id) => {
                self.apply(Event::Stop);
                Ok(id)
            }
            ApiRequest::Cancel(id) => {
                self.apply(Event::Cancel);
                Ok(id)
            }
        }
    }

    fn tick(&mut self) {
        let now = self.now_ms();
        if let Some(event) = self.interpreter.poll(now) {
            self.apply(event);
        }
        if let Ok(outcome) = self.machine.tick(now) {
            if outcome.effects.contains(&Effect::WarnDurationLimit) {
                self.notice("duration_limit_soon");
            } else if outcome.accepted {
                for event in self.perform(outcome.effects) {
                    self.apply(event);
                }
            }
        }
        // Device removal or permission revocation surfaces as a stream error.
        let capture_failed = self
            .active
            .as_ref()
            .and_then(|active| active.capture.as_ref())
            .is_some_and(|capture| capture.check_runtime().is_err());
        if capture_failed && self.machine.state().is_recording() {
            self.notice("microphone_failure");
            self.apply(Event::Cancel);
        }
        self.schedule_partial();
        if let Some(since) = self.error_since {
            if now.saturating_sub(since) >= ERROR_DISPLAY_MS {
                self.error_since = None;
                self.apply(Event::Reset);
            }
        }
        if self.machine.state().is_recording() && now.saturating_sub(self.last_publish_ms) >= 250 {
            self.last_publish_ms = now;
            let elapsed = self.machine.snapshot(now).elapsed_ms;
            self.shared.publish(|status| status.elapsed_ms = elapsed);
        }
    }

    fn notice(&self, code: &str) {
        self.shared.publish(|status| status.notice = Some(code.to_owned()));
    }

    fn apply(&mut self, event: Event) {
        let mut queue = VecDeque::from([event]);
        while let Some(event) = queue.pop_front() {
            let session = matches!(event, Event::HoldDown | Event::Toggle).then(|| format!("session-{}", dictation_engine::training::random_hex(16)));
            let Ok(outcome) = self.machine.handle(event, session.as_deref(), self.now_ms()) else {
                continue;
            };
            if !outcome.accepted {
                continue;
            }
            queue.extend(self.perform(outcome.effects));
        }
    }

    /// Performs effects; returns follow-up events produced synchronously.
    #[allow(clippy::too_many_lines)]
    fn perform(&mut self, effects: Vec<Effect>) -> Vec<Event> {
        let mut follow = Vec::new();
        for effect in effects {
            match effect {
                Effect::StartCapture => follow.push(self.start_capture()),
                Effect::StopCapture => {
                    if let Some(active) = self.active.as_mut() {
                        if let Some(capture) = active.capture.take() {
                            match capture.stop() {
                                Ok(audio) => active.pcm = audio.samples,
                                Err(_) => {
                                    active.pcm.clear();
                                    self.shared.publish(|status| status.notice = Some("microphone_failure".to_owned()));
                                    if self.machine.state() == State::Finalizing {
                                        follow.push(Event::Failed);
                                    }
                                }
                            }
                        }
                    }
                }
                Effect::FinalizeTranscript => follow.extend(self.finalize()),
                Effect::DeliverText => {
                    self.deliver();
                    follow.push(Event::Delivered);
                }
                Effect::DiscardSession => {
                    if let Some(active) = self.active.as_mut() {
                        active.pcm = Vec::new();
                        active.final_text = None;
                        active.final_segments.clear();
                        active.epoch = u64::MAX; // any in-flight result is now stale
                    }
                }
                Effect::AnnounceState => follow.extend(self.announce()),
                Effect::WarnDurationLimit => self.notice("duration_limit_soon"),
                Effect::ReleaseResources => {
                    if let Some(active) = self.active.take() {
                        if self.machine.state() == State::Idle && active.final_text.is_some() {
                            training_copy::maybe_enqueue(&self.shared, &active.session_id, active.started_unix, &active.language, &active.final_segments, active.pcm);
                        }
                    }
                    self.shared.preemption.set(false);
                    let _ = self.shared.background_wake.send(());
                }
            }
        }
        follow
    }

    fn start_capture(&mut self) -> Event {
        self.epoch += 1;
        let session_id = self.machine.session_id().unwrap_or_default().to_owned();
        // Remember the target before anything else can take focus.
        let focus = self.shared.desktop.focus();
        self.shared.preemption.set(true);
        let limit = self.shared.settings().max_session_seconds + 5;
        match MicrophoneCapture::start_default(limit) {
            Ok(capture) => {
                self.active = Some(Active {
                    session_id,
                    epoch: self.epoch,
                    started_unix: crate::unix_now(),
                    focus,
                    capture: Some(capture),
                    pcm: Vec::new(),
                    live: LiveReconciler::new(LiveSettings::default()),
                    partial_in_flight: false,
                    last_pass_end: 0,
                    final_text: None,
                    final_segments: Vec::new(),
                    language: String::new(),
                });
                self.shared.publish(|status| status.notice = None);
                self.asr.send(AsrRequest::Warm);
                Event::CaptureStarted
            }
            Err(error) => {
                let code = match error {
                    dictation_platform::microphone::CaptureError::NoInputDevice => "no_microphone",
                    dictation_platform::microphone::CaptureError::DefaultConfig(_) => "microphone_permission_or_config",
                    _ => "microphone_failure",
                };
                self.notice(code);
                Event::Failed
            }
        }
    }

    fn schedule_partial(&mut self) {
        if !self.machine.state().is_recording() {
            return;
        }
        let Some(active) = self.active.as_mut() else { return };
        if active.partial_in_flight {
            return;
        }
        let Some(capture) = active.capture.as_ref() else { return };
        let captured = capture.captured_samples() as u64;
        let from = active.live.committed_until();
        if captured.saturating_sub(active.last_pass_end) < PARTIAL_INTERVAL_SAMPLES {
            return;
        }
        active.last_pass_end = captured;
        let Ok(pcm) = capture.snapshot_since(usize::try_from(from).unwrap_or(usize::MAX)) else { return };
        active.partial_in_flight = true;
        self.asr.send(AsrRequest::Partial { epoch: active.epoch, pcm, offset: from, audio_end: captured });
    }

    fn on_partial(&mut self, epoch: u64, result: AsrResult, offset: u64, audio_end: u64) {
        let Some(active) = self.active.as_mut() else { return };
        if active.epoch != epoch {
            return;
        }
        active.partial_in_flight = false;
        if !self.machine.state().is_recording() {
            return;
        }
        let Ok((_, segments)) = result else { return };
        let segments: Vec<RecognizedSegment> = segments
            .into_iter()
            .map(|segment| segment.offset(offset))
            .filter(|segment| segment.no_speech_probability < 0.6)
            .collect();
        let updates = active.live.apply_pass(&segments, audio_end);
        let session_id = active.session_id.clone();
        let caption = active.live.text();
        for update in updates {
            let seq = self.next_seq();
            self.shared.bus.publish(ApiEvent::partial(
                &session_id,
                seq,
                &update.segment_id,
                update.revision,
                update.start_sample / 16,
                update.end_sample / 16,
                &update.text,
            ));
        }
        let _ = self.shared.app.emit_to("indicator", "caption", tail(&caption, 90));
    }

    fn finalize(&mut self) -> Vec<Event> {
        let Some(active) = self.active.as_mut() else { return vec![Event::Failed] };
        if active.pcm.is_empty() {
            active.final_text = Some(String::new());
            return vec![Event::Finalized];
        }
        let activity = SessionActivity::analyze(&active.pcm, 16_000, VadSettings::default());
        if activity.is_effectively_silent() {
            // Silence never becomes (hallucinated) text.
            active.final_text = Some(String::new());
            self.shared.publish(|status| status.notice = Some("no_speech_detected".to_owned()));
            return vec![Event::Finalized];
        }
        self.asr.send(AsrRequest::Final { epoch: active.epoch, pcm: active.pcm.clone() });
        Vec::new()
    }

    fn on_final(&mut self, epoch: u64, result: AsrResult) {
        let Some(active) = self.active.as_mut() else { return };
        if active.epoch != epoch || self.machine.state() != State::Finalizing {
            return; // stale: cancelled or superseded
        }
        match result {
            Ok((language, segments)) => {
                let activity = SessionActivity::analyze(&active.pcm, 16_000, VadSettings::default());
                let kept: Vec<RecognizedSegment> = segments
                    .into_iter()
                    .filter(|segment| !is_hallucination_risk(segment, &activity, 0.6))
                    .collect();
                active.final_text = Some(display_text(&kept));
                active.final_segments = kept;
                active.language = language;
                self.apply(Event::Finalized);
            }
            Err(code) => {
                let notice = if code.contains("not installed") || code.contains("missing") {
                    "asr_model_missing"
                } else {
                    "transcription_failed"
                };
                self.notice(notice);
                self.apply(Event::Failed);
            }
        }
    }

    fn deliver(&mut self) {
        let Some(active) = self.active.as_ref() else { return };
        let text = active.final_text.clone().unwrap_or_default();
        let session_id = active.session_id.clone();
        let end_ms = active.pcm.len() as u64 / 16;
        let focus = active.focus.clone();
        if !text.is_empty() {
            let seq = self.next_seq();
            self.shared.bus.publish(ApiEvent::final_transcript(&session_id, seq, end_ms, &text));
            delivery::deliver(&self.shared, &session_id, focus.as_ref(), &text);
        }
    }

    /// Publishes the new state and derives API lifecycle events from the
    /// transition that just happened, so a cancelled session is never also
    /// reported as stopped.
    fn announce(&mut self) -> Option<Event> {
        let now = self.now_ms();
        let snapshot = self.machine.snapshot(now);
        let state = snapshot.state;
        let name = state_name(state);
        if state == State::Error {
            self.error_since = Some(now);
        }
        let session_id = self.active.as_ref().map(|active| active.session_id.clone()).or(snapshot.session_id);
        let limit = self.shared.settings().max_session_seconds;
        self.shared.publish(|status| {
            status.recording_state = name;
            status.locked = snapshot.locked;
            status.elapsed_ms = snapshot.elapsed_ms;
            status.max_session_seconds = limit;
            if session_id.is_some() {
                status.session_id.clone_from(&session_id);
            }
        });
        let _ = self.shared.app.emit_to("indicator", "indicator-state", name);
        crate::indicator::update(&self.shared.app, state);
        let last = self.machine.history().last().map(|transition| transition.event);
        let kind = match last {
            Some(Event::CaptureStarted) => Some(EventKind::SessionStarted),
            Some(Event::Delivered) => Some(EventKind::SessionStopped),
            Some(Event::Cancel) => Some(EventKind::SessionCancelled),
            Some(Event::Failed) => Some(EventKind::Error),
            _ => None,
        };
        if let (Some(kind), Some(session_id)) = (kind, session_id) {
            let seq = self.next_seq();
            let code = (kind == EventKind::Error).then(|| self.shared.status().notice.unwrap_or_else(|| "error".to_owned()));
            self.shared.bus.publish(ApiEvent::session(kind, &session_id, seq, code.as_deref()));
        }
        // Cancelled sessions return to idle immediately; nothing is kept.
        (state == State::Cancelled).then_some(Event::Reset)
    }
}

fn tail(text: &str, characters: usize) -> String {
    let count = text.chars().count();
    if count <= characters {
        text.to_owned()
    } else {
        format!("…{}", text.chars().skip(count - characters).collect::<String>())
    }
}

#[must_use]
pub const fn state_name(state: State) -> &'static str {
    match state {
        State::Idle => "idle",
        State::Starting => "starting",
        State::RecordingHeld => "recording",
        State::RecordingLocked => "recording_locked",
        State::Finalizing => "finalizing",
        State::Delivering => "delivering",
        State::Cancelled => "cancelled",
        State::Error => "error",
    }
}
