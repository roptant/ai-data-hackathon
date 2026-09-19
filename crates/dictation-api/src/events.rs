//! Event envelopes and the bounded broadcast bus.
//!
//! A later revision replaces earlier text for the same segment; clients must
//! not append partials. `privacy` is always `"unredacted"` in this version:
//! streaming redaction is not offered, because later words can reveal that
//! earlier text was sensitive.

use dictation_storage::api_clients::Scope;
use serde::Serialize;
use tokio::sync::broadcast;

use crate::API_VERSION;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EventKind {
    #[serde(rename = "session.started")]
    SessionStarted,
    #[serde(rename = "transcript.partial")]
    TranscriptPartial,
    #[serde(rename = "transcript.final")]
    TranscriptFinal,
    #[serde(rename = "session.stopped")]
    SessionStopped,
    #[serde(rename = "session.cancelled")]
    SessionCancelled,
    #[serde(rename = "error")]
    Error,
}

impl EventKind {
    /// Scope required to receive this event. Captions need no control scope.
    #[must_use]
    pub const fn scope(self) -> Scope {
        match self {
            Self::TranscriptPartial => Scope::TranscriptLive,
            Self::TranscriptFinal => Scope::TranscriptFinal,
            Self::SessionStarted | Self::SessionStopped | Self::SessionCancelled | Self::Error => {
                Scope::StatusRead
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApiEvent {
    pub version: u32,
    pub event: EventKind,
    pub session_id: String,
    pub seq: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_final: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy: Option<&'static str>,
    /// Content-free error or cancellation code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ApiEvent {
    fn base(event: EventKind, session_id: &str, seq: u64) -> Self {
        Self {
            version: API_VERSION,
            event,
            session_id: session_id.to_owned(),
            seq,
            segment_id: None,
            revision: None,
            start_ms: None,
            end_ms: None,
            text: None,
            is_final: None,
            privacy: None,
            code: None,
        }
    }

    #[must_use]
    pub fn session(event: EventKind, session_id: &str, seq: u64, code: Option<&str>) -> Self {
        Self {
            code: code.map(str::to_owned),
            ..Self::base(event, session_id, seq)
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn partial(
        session_id: &str,
        seq: u64,
        segment_id: &str,
        revision: u64,
        start_ms: u64,
        end_ms: u64,
        text: &str,
    ) -> Self {
        Self {
            segment_id: Some(segment_id.to_owned()),
            revision: Some(revision),
            start_ms: Some(start_ms),
            end_ms: Some(end_ms),
            text: Some(text.to_owned()),
            is_final: Some(false),
            privacy: Some("unredacted"),
            ..Self::base(EventKind::TranscriptPartial, session_id, seq)
        }
    }

    /// The complete canonical transcript; supersedes every partial.
    #[must_use]
    pub fn final_transcript(session_id: &str, seq: u64, end_ms: u64, text: &str) -> Self {
        Self {
            start_ms: Some(0),
            end_ms: Some(end_ms),
            text: Some(text.to_owned()),
            is_final: Some(true),
            privacy: Some("unredacted"),
            ..Self::base(EventKind::TranscriptFinal, session_id, seq)
        }
    }
}

/// Bounded fan-out. A subscriber that lags past `capacity` events is told to
/// resynchronize and is disconnected by the server.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<ApiEvent>,
}

impl EventBus {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self { sender }
    }

    /// Publishes to current subscribers; nothing is retained for later ones.
    pub fn publish(&self, event: ApiEvent) {
        let _ = self.sender.send(event);
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<ApiEvent> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}
