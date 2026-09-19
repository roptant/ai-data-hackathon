//! Completion-token coordinator for non-blocking capture and delivery.
//!
//! Slow capture, transcription, and insertion operations execute outside this
//! type. Their completions carry an epoch token; cancellation or replacement
//! makes old completions stale. Training work is produced only after delivery
//! and is owned by a separate supervisor.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
    thread,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SessionToken {
    session_id: String,
    epoch: u64,
}

impl SessionToken {
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LivePhase {
    Capturing,
    Finalizing,
    Delivering,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinationError {
    Busy,
    NoActiveSession,
    StaleCompletion,
    WrongPhase {
        expected: LivePhase,
        actual: LivePhase,
    },
}

impl fmt::Display for CoordinationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => write!(formatter, "a live session already exists"),
            Self::NoActiveSession => write!(formatter, "no live session exists"),
            Self::StaleCompletion => write!(formatter, "completion belongs to an obsolete session"),
            Self::WrongPhase { expected, actual } => {
                write!(formatter, "expected {expected:?}, found {actual:?}")
            }
        }
    }
}

impl std::error::Error for CoordinationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveSession {
    token: SessionToken,
    phase: LivePhase,
}

#[derive(Debug, Default)]
pub struct CaptureCoordinator {
    next_epoch: u64,
    active: Option<ActiveSession>,
}

impl CaptureCoordinator {
    #[must_use]
    pub fn active(&self) -> Option<(&SessionToken, LivePhase)> {
        self.active
            .as_ref()
            .map(|active| (&active.token, active.phase))
    }

    /// Reserves a new capture epoch.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinationError::Busy`] while another session is active.
    pub fn start(
        &mut self,
        session_id: impl Into<String>,
    ) -> Result<SessionToken, CoordinationError> {
        if self.active.is_some() {
            return Err(CoordinationError::Busy);
        }
        self.next_epoch = self.next_epoch.saturating_add(1);
        let token = SessionToken {
            session_id: session_id.into(),
            epoch: self.next_epoch,
        };
        self.active = Some(ActiveSession {
            token: token.clone(),
            phase: LivePhase::Capturing,
        });
        Ok(token)
    }

    /// Marks capture stopping without waiting for capture or ASR I/O.
    ///
    /// # Errors
    ///
    /// Rejects stale tokens and calls made outside the capture phase.
    pub fn begin_finalization(&mut self, token: &SessionToken) -> Result<(), CoordinationError> {
        let active = self.current_mut(token)?;
        require_phase(active.phase, LivePhase::Capturing)?;
        active.phase = LivePhase::Finalizing;
        Ok(())
    }

    /// Accepts an ASR completion only for the current finalizing epoch.
    ///
    /// # Errors
    ///
    /// Rejects late or out-of-order completions.
    pub fn finalization_completed(
        &mut self,
        token: &SessionToken,
    ) -> Result<(), CoordinationError> {
        let active = self.current_mut(token)?;
        require_phase(active.phase, LivePhase::Finalizing)?;
        active.phase = LivePhase::Delivering;
        Ok(())
    }

    /// Completes user delivery and releases the live session.
    ///
    /// The returned token can seed separately owned background training work.
    ///
    /// # Errors
    ///
    /// Rejects late or out-of-order delivery completions.
    pub fn delivery_completed(
        &mut self,
        token: &SessionToken,
    ) -> Result<SessionToken, CoordinationError> {
        let active = self.current(token)?;
        require_phase(active.phase, LivePhase::Delivering)?;
        let completed = active.token.clone();
        self.active = None;
        Ok(completed)
    }

    /// Invalidates all outstanding work for the current epoch.
    ///
    /// # Errors
    ///
    /// Rejects a cancellation token belonging to another session.
    pub fn cancel(&mut self, token: &SessionToken) -> Result<(), CoordinationError> {
        self.current(token)?;
        self.active = None;
        Ok(())
    }

    /// Fails the current session without allowing delivery or training.
    ///
    /// # Errors
    ///
    /// Rejects a failure notification belonging to another session.
    pub fn fail(&mut self, token: &SessionToken) -> Result<(), CoordinationError> {
        self.cancel(token)
    }

    fn current(&self, token: &SessionToken) -> Result<&ActiveSession, CoordinationError> {
        let active = self
            .active
            .as_ref()
            .ok_or(CoordinationError::NoActiveSession)?;
        if active.token == *token {
            Ok(active)
        } else {
            Err(CoordinationError::StaleCompletion)
        }
    }

    fn current_mut(
        &mut self,
        token: &SessionToken,
    ) -> Result<&mut ActiveSession, CoordinationError> {
        let active = self
            .active
            .as_mut()
            .ok_or(CoordinationError::NoActiveSession)?;
        if active.token == *token {
            Ok(active)
        } else {
            Err(CoordinationError::StaleCompletion)
        }
    }
}

fn require_phase(actual: LivePhase, expected: LivePhase) -> Result<(), CoordinationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CoordinationError::WrongPhase { expected, actual })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingJobState {
    Pending,
    Running,
    Eligible,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingOutcome {
    Eligible,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct TrainingSupervisor {
    states: Arc<Mutex<BTreeMap<SessionToken, TrainingJobState>>>,
}

impl Default for TrainingSupervisor {
    fn default() -> Self {
        Self {
            states: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }
}

impl TrainingSupervisor {
    /// Spawns a separately owned training job after successful delivery.
    ///
    /// The worker receives its input by value and has no reference to the live
    /// capture coordinator.
    pub fn spawn<T, F>(&self, token: SessionToken, input: T, worker: F)
    where
        T: Send + 'static,
        F: FnOnce(T) -> Result<TrainingOutcome, ()> + Send + 'static,
    {
        if let Ok(mut states) = self.states.lock() {
            states.insert(token.clone(), TrainingJobState::Pending);
        }
        let states = Arc::clone(&self.states);
        thread::spawn(move || {
            if let Ok(mut guard) = states.lock() {
                guard.insert(token.clone(), TrainingJobState::Running);
            }
            let terminal = match worker(input) {
                Ok(TrainingOutcome::Eligible) => TrainingJobState::Eligible,
                Ok(TrainingOutcome::Rejected) => TrainingJobState::Rejected,
                Err(()) => TrainingJobState::Failed,
            };
            if let Ok(mut guard) = states.lock() {
                guard.insert(token, terminal);
            }
        });
    }

    #[must_use]
    pub fn state(&self, token: &SessionToken) -> Option<TrainingJobState> {
        self.states.lock().ok()?.get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, time::Duration};

    use super::*;

    #[test]
    fn cancellation_invalidates_late_finalization() {
        let mut coordinator = CaptureCoordinator::default();
        let old = coordinator.start("old").unwrap();
        coordinator.begin_finalization(&old).unwrap();
        coordinator.cancel(&old).unwrap();
        let current = coordinator.start("current").unwrap();
        assert_eq!(
            coordinator.finalization_completed(&old),
            Err(CoordinationError::StaleCompletion)
        );
        assert_eq!(coordinator.active(), Some((&current, LivePhase::Capturing)));
    }

    #[test]
    fn training_is_released_only_after_delivery() {
        let mut coordinator = CaptureCoordinator::default();
        let token = coordinator.start("session").unwrap();
        coordinator.begin_finalization(&token).unwrap();
        coordinator.finalization_completed(&token).unwrap();
        let training_token = coordinator.delivery_completed(&token).unwrap();
        assert_eq!(training_token, token);
        assert!(coordinator.active().is_none());
    }

    #[test]
    fn background_completion_cannot_mutate_new_live_session() {
        let mut coordinator = CaptureCoordinator::default();
        let old = coordinator.start("old").unwrap();
        coordinator.begin_finalization(&old).unwrap();
        coordinator.finalization_completed(&old).unwrap();
        let training_token = coordinator.delivery_completed(&old).unwrap();

        let supervisor = TrainingSupervisor::default();
        let (release_tx, release_rx) = mpsc::channel();
        supervisor.spawn(training_token.clone(), (), move |()| {
            release_rx.recv().unwrap();
            Ok(TrainingOutcome::Eligible)
        });
        let current = coordinator.start("current").unwrap();
        release_tx.send(()).unwrap();

        for _ in 0..100 {
            if supervisor.state(&training_token) == Some(TrainingJobState::Eligible) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            supervisor.state(&training_token),
            Some(TrainingJobState::Eligible)
        );
        assert_eq!(coordinator.active(), Some((&current, LivePhase::Capturing)));
    }
}
