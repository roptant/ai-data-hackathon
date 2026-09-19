//! Pure recording state machine from implementation-plan section 5.
//!
//! I/O is expressed as effects. The desktop coordinator performs those effects
//! without holding the state lock, keeping stop/cancel idempotent and making it
//! impossible for a late finalization to authorize delivery after cancellation.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Starting,
    RecordingHeld,
    RecordingLocked,
    Finalizing,
    Delivering,
    Cancelled,
    Error,
}

impl State {
    #[must_use]
    pub const fn is_recording(self) -> bool {
        matches!(self, Self::RecordingHeld | Self::RecordingLocked)
    }

    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(
            self,
            Self::Starting
                | Self::RecordingHeld
                | Self::RecordingLocked
                | Self::Finalizing
                | Self::Delivering
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    HoldDown,
    HoldUp,
    Toggle,
    Lock,
    Stop,
    Cancel,
    CaptureStarted,
    Finalized,
    Delivered,
    Failed,
    DurationLimit,
    Reset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    StartCapture,
    StopCapture,
    FinalizeTranscript,
    DeliverText,
    DiscardSession,
    AnnounceState,
    WarnDurationLimit,
    ReleaseResources,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub accepted: bool,
    pub state: State,
    pub effects: Vec<Effect>,
    pub ignored_reason: Option<&'static str>,
}

impl Outcome {
    fn accepted(state: State, effects: impl IntoIterator<Item = Effect>) -> Self {
        Self {
            accepted: true,
            state,
            effects: effects.into_iter().collect(),
            ignored_reason: None,
        }
    }

    fn ignored(state: State, reason: &'static str) -> Self {
        Self {
            accepted: false,
            state,
            effects: Vec::new(),
            ignored_reason: Some(reason),
        }
    }

    #[must_use]
    pub const fn is_ignored(&self) -> bool {
        !self.accepted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidTransition {
    pub state: State,
    pub event: Event,
}

impl fmt::Display for InvalidTransition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "event {:?} is invalid in state {:?}",
            self.event, self.state
        )
    }
}

impl std::error::Error for InvalidTransition {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub at_ms: u64,
    pub before: State,
    pub event: Event,
    pub after: State,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub state: State,
    pub recording: bool,
    pub locked: bool,
    pub elapsed_ms: u64,
    pub session_id: Option<String>,
    pub max_session_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct RecordingStateMachine {
    state: State,
    session_id: Option<String>,
    started_at_ms: Option<u64>,
    max_session_seconds: u64,
    warn_before_limit_seconds: u64,
    flags: u8,
    history: Vec<Transition>,
}

const CANCEL_REQUESTED: u8 = 1 << 0;
const PENDING_LOCKED: u8 = 1 << 1;
const PENDING_STOP: u8 = 1 << 2;
const WARNED: u8 = 1 << 3;

impl Default for RecordingStateMachine {
    fn default() -> Self {
        Self::new(600, 30)
    }
}

impl RecordingStateMachine {
    #[must_use]
    pub const fn new(max_session_seconds: u64, warn_before_limit_seconds: u64) -> Self {
        Self {
            state: State::Idle,
            session_id: None,
            started_at_ms: None,
            max_session_seconds,
            warn_before_limit_seconds,
            flags: 0,
            history: Vec::new(),
        }
    }

    #[must_use]
    pub const fn state(&self) -> State {
        self.state
    }

    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    #[must_use]
    pub const fn insertion_permitted(&self) -> bool {
        !self.has_flag(CANCEL_REQUESTED)
    }

    #[must_use]
    pub fn history(&self) -> &[Transition] {
        &self.history
    }

    #[must_use]
    pub fn snapshot(&self, now_ms: u64) -> Snapshot {
        Snapshot {
            state: self.state,
            recording: self.state.is_recording(),
            locked: self.state == State::RecordingLocked,
            elapsed_ms: self
                .started_at_ms
                .map_or(0, |started| now_ms.saturating_sub(started)),
            session_id: self.session_id.clone(),
            max_session_seconds: self.max_session_seconds,
        }
    }

    /// Apply one serialized command or lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidTransition`] for commands that indicate a programming
    /// error rather than an expected idempotent or conflicting command.
    pub fn handle(
        &mut self,
        event: Event,
        session_id: Option<&str>,
        now_ms: u64,
    ) -> Result<Outcome, InvalidTransition> {
        let before = self.state;
        let outcome = match event {
            Event::HoldDown => self.start(false, session_id, Event::HoldDown, now_ms)?,
            Event::Toggle => self.toggle(session_id, now_ms)?,
            Event::HoldUp => self.hold_up(),
            Event::Lock => self.lock(),
            Event::Stop => self.stop(),
            Event::Cancel => self.cancel(),
            Event::CaptureStarted => self.capture_started(),
            Event::Finalized => self.finalized(),
            Event::Delivered => self.delivered(),
            Event::Failed => self.failed(),
            Event::DurationLimit => self.duration_limit(),
            Event::Reset => self.reset()?,
        };
        if outcome.accepted {
            self.history.push(Transition {
                at_ms: now_ms,
                before,
                event,
                after: self.state,
            });
        }
        Ok(outcome)
    }

    /// Enforce the warning and maximum recording duration.
    ///
    /// # Errors
    ///
    /// Propagates an invalid duration-limit transition if the internal state is
    /// inconsistent. Expected calls outside recording are ignored instead.
    pub fn tick(&mut self, now_ms: u64) -> Result<Outcome, InvalidTransition> {
        if !self.state.is_recording() {
            return Ok(Outcome::ignored(self.state, "not_recording"));
        }
        let Some(started_at_ms) = self.started_at_ms else {
            return Ok(Outcome::ignored(self.state, "not_recording"));
        };
        let elapsed_ms = now_ms.saturating_sub(started_at_ms);
        let limit_ms = self.max_session_seconds.saturating_mul(1_000);
        if elapsed_ms >= limit_ms {
            return self.handle(Event::DurationLimit, None, now_ms);
        }
        let warning_ms = self
            .max_session_seconds
            .saturating_sub(self.warn_before_limit_seconds)
            .saturating_mul(1_000);
        if !self.has_flag(WARNED) && elapsed_ms >= warning_ms {
            self.set_flag(WARNED, true);
            return Ok(Outcome::accepted(self.state, [Effect::WarnDurationLimit]));
        }
        Ok(Outcome::ignored(self.state, "within_limit"))
    }

    fn start(
        &mut self,
        locked: bool,
        session_id: Option<&str>,
        source: Event,
        now_ms: u64,
    ) -> Result<Outcome, InvalidTransition> {
        match self.state {
            State::Starting | State::RecordingHeld | State::RecordingLocked => {
                return Ok(Outcome::ignored(self.state, "auto_repeat"));
            }
            State::Finalizing | State::Delivering => {
                return Ok(Outcome::ignored(self.state, "busy_finalizing"));
            }
            State::Idle => {}
            _ => {
                return Err(InvalidTransition {
                    state: self.state,
                    event: source,
                });
            }
        }
        let Some(session_id) = session_id.filter(|value| !value.is_empty()) else {
            return Err(InvalidTransition {
                state: self.state,
                event: source,
            });
        };
        self.state = State::Starting;
        self.session_id = Some(session_id.to_owned());
        self.started_at_ms = Some(now_ms);
        self.flags = 0;
        self.set_flag(PENDING_LOCKED, locked);
        Ok(Outcome::accepted(
            self.state,
            [Effect::StartCapture, Effect::AnnounceState],
        ))
    }

    fn toggle(
        &mut self,
        session_id: Option<&str>,
        now_ms: u64,
    ) -> Result<Outcome, InvalidTransition> {
        match self.state {
            State::Idle => self.start(true, session_id, Event::Toggle, now_ms),
            State::Starting => {
                self.set_flag(PENDING_STOP, true);
                Ok(Outcome::accepted(self.state, []))
            }
            State::RecordingHeld | State::RecordingLocked => Ok(self.begin_finalize()),
            _ => Ok(Outcome::ignored(self.state, "not_recording")),
        }
    }

    fn hold_up(&mut self) -> Outcome {
        match self.state {
            State::RecordingHeld => self.begin_finalize(),
            State::RecordingLocked => Outcome::ignored(self.state, "locked"),
            State::Starting if !self.has_flag(PENDING_LOCKED) => {
                self.set_flag(PENDING_STOP, true);
                Outcome::accepted(self.state, [])
            }
            _ => Outcome::ignored(self.state, "not_held"),
        }
    }

    fn lock(&mut self) -> Outcome {
        match self.state {
            State::RecordingHeld => {
                self.state = State::RecordingLocked;
                Outcome::accepted(self.state, [Effect::AnnounceState])
            }
            State::Starting => {
                self.set_flag(PENDING_LOCKED, true);
                Outcome::accepted(self.state, [])
            }
            State::RecordingLocked => Outcome::ignored(self.state, "already_locked"),
            _ => Outcome::ignored(self.state, "not_recording"),
        }
    }

    fn stop(&mut self) -> Outcome {
        match self.state {
            State::RecordingHeld | State::RecordingLocked => self.begin_finalize(),
            State::Starting => {
                self.set_flag(PENDING_STOP, true);
                Outcome::accepted(self.state, [])
            }
            _ => Outcome::ignored(self.state, "already_stopping"),
        }
    }

    fn cancel(&mut self) -> Outcome {
        if matches!(self.state, State::Idle | State::Cancelled) {
            return Outcome::ignored(self.state, "nothing_to_cancel");
        }
        self.set_flag(CANCEL_REQUESTED, true);
        self.state = State::Cancelled;
        Outcome::accepted(
            self.state,
            [
                Effect::StopCapture,
                Effect::DiscardSession,
                Effect::AnnounceState,
                Effect::ReleaseResources,
            ],
        )
    }

    fn capture_started(&mut self) -> Outcome {
        if self.state != State::Starting {
            return Outcome::ignored(self.state, "not_starting");
        }
        self.state = if self.has_flag(PENDING_LOCKED) {
            State::RecordingLocked
        } else {
            State::RecordingHeld
        };
        if self.has_flag(PENDING_STOP) {
            self.set_flag(PENDING_STOP, false);
            return self.begin_finalize();
        }
        Outcome::accepted(self.state, [Effect::AnnounceState])
    }

    fn begin_finalize(&mut self) -> Outcome {
        self.state = State::Finalizing;
        Outcome::accepted(
            self.state,
            [
                Effect::StopCapture,
                Effect::FinalizeTranscript,
                Effect::AnnounceState,
            ],
        )
    }

    fn finalized(&mut self) -> Outcome {
        if self.state != State::Finalizing {
            return Outcome::ignored(self.state, "not_finalizing");
        }
        if self.has_flag(CANCEL_REQUESTED) {
            self.state = State::Cancelled;
            return Outcome::accepted(self.state, [Effect::DiscardSession]);
        }
        self.state = State::Delivering;
        Outcome::accepted(self.state, [Effect::DeliverText])
    }

    fn delivered(&mut self) -> Outcome {
        if self.state != State::Delivering {
            return Outcome::ignored(self.state, "not_delivering");
        }
        self.state = State::Idle;
        self.session_id = None;
        self.started_at_ms = None;
        Outcome::accepted(
            self.state,
            [Effect::AnnounceState, Effect::ReleaseResources],
        )
    }

    fn failed(&mut self) -> Outcome {
        if self.state == State::Idle {
            return Outcome::ignored(self.state, "idle");
        }
        self.state = State::Error;
        Outcome::accepted(
            self.state,
            [
                Effect::StopCapture,
                Effect::DiscardSession,
                Effect::AnnounceState,
                Effect::ReleaseResources,
            ],
        )
    }

    fn duration_limit(&mut self) -> Outcome {
        if !self.state.is_recording() {
            return Outcome::ignored(self.state, "not_recording");
        }
        self.begin_finalize()
    }

    fn reset(&mut self) -> Result<Outcome, InvalidTransition> {
        match self.state {
            State::Cancelled | State::Error => {
                self.state = State::Idle;
                self.session_id = None;
                self.started_at_ms = None;
                self.flags = 0;
                Ok(Outcome::accepted(self.state, [Effect::AnnounceState]))
            }
            State::Idle => Ok(Outcome::ignored(self.state, "already_idle")),
            _ => Err(InvalidTransition {
                state: self.state,
                event: Event::Reset,
            }),
        }
    }

    const fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    fn set_flag(&mut self, flag: u8, enabled: bool) {
        if enabled {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn started() -> RecordingStateMachine {
        let mut machine = RecordingStateMachine::default();
        machine
            .handle(Event::HoldDown, Some("session-1"), 0)
            .expect("start");
        machine
            .handle(Event::CaptureStarted, None, 100)
            .expect("capture started");
        machine
    }

    #[test]
    fn hold_to_record_round_trip() {
        let mut machine = started();
        let stop = machine.handle(Event::HoldUp, None, 500).expect("stop");
        assert_eq!(machine.state(), State::Finalizing);
        assert!(stop.effects.contains(&Effect::FinalizeTranscript));
        machine
            .handle(Event::Finalized, None, 600)
            .expect("finalized");
        assert_eq!(machine.state(), State::Delivering);
        machine
            .handle(Event::Delivered, None, 700)
            .expect("delivered");
        assert_eq!(machine.state(), State::Idle);
    }

    #[test]
    fn toggle_starts_locked_and_second_toggle_stops() {
        let mut machine = RecordingStateMachine::default();
        machine
            .handle(Event::Toggle, Some("session-1"), 0)
            .expect("toggle start");
        machine
            .handle(Event::CaptureStarted, None, 100)
            .expect("capture started");
        assert_eq!(machine.state(), State::RecordingLocked);
        machine
            .handle(Event::Toggle, None, 200)
            .expect("toggle stop");
        assert_eq!(machine.state(), State::Finalizing);
    }

    #[test]
    fn auto_repeat_does_not_replace_the_session() {
        let mut machine = started();
        let repeat = machine
            .handle(Event::HoldDown, Some("session-2"), 200)
            .expect("ignored repeat");
        assert!(repeat.is_ignored());
        assert_eq!(repeat.ignored_reason, Some("auto_repeat"));
        assert_eq!(machine.session_id(), Some("session-1"));
    }

    #[test]
    fn locking_survives_hold_release() {
        let mut machine = started();
        machine.handle(Event::Lock, None, 200).expect("lock");
        let released = machine
            .handle(Event::HoldUp, None, 300)
            .expect("ignored release");
        assert_eq!(released.ignored_reason, Some("locked"));
        assert_eq!(machine.state(), State::RecordingLocked);
    }

    #[test]
    fn release_before_capture_confirmation_still_finalizes() {
        let mut machine = RecordingStateMachine::default();
        machine
            .handle(Event::HoldDown, Some("session-1"), 0)
            .expect("start");
        machine.handle(Event::HoldUp, None, 1).expect("release");
        machine
            .handle(Event::CaptureStarted, None, 2)
            .expect("capture started");
        assert_eq!(machine.state(), State::Finalizing);
    }

    #[test]
    fn cancellation_is_idempotent_and_blocks_late_delivery() {
        let mut machine = started();
        machine.handle(Event::Cancel, None, 200).expect("cancel");
        assert!(!machine.insertion_permitted());
        assert!(
            machine
                .handle(Event::Cancel, None, 300)
                .expect("second cancel")
                .is_ignored()
        );
        let late = machine
            .handle(Event::Finalized, None, 400)
            .expect("late finalization");
        assert!(!late.effects.contains(&Effect::DeliverText));
    }

    #[test]
    fn duration_limit_warns_then_finalizes() {
        let mut machine = RecordingStateMachine::new(600, 30);
        machine
            .handle(Event::HoldDown, Some("session-1"), 0)
            .expect("start");
        machine
            .handle(Event::CaptureStarted, None, 100)
            .expect("capture started");
        let warning = machine.tick(575_000).expect("warning");
        assert!(warning.effects.contains(&Effect::WarnDurationLimit));
        let limit = machine.tick(601_000).expect("limit");
        assert!(limit.effects.contains(&Effect::FinalizeTranscript));
        assert_eq!(machine.state(), State::Finalizing);
    }

    #[test]
    fn reset_while_recording_is_invalid() {
        let mut machine = started();
        assert!(machine.handle(Event::Reset, None, 200).is_err());
    }
}
