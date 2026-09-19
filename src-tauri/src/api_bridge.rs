//! Connects the local API to the app: paired clients live in the encrypted
//! metadata store; control requests go through the serialized controller.

use std::{
    sync::{Arc, Weak, atomic::Ordering, mpsc::channel},
    time::Duration,
};

use dictation_api::{
    ApiConfig, ApiEvent, ControlError, Controller, StatusSnapshot,
    server::{AuthorizedClient, ClientDirectory},
};

use crate::{
    controller::{ApiRequest, Command},
    state::Shared,
};

pub struct Directory(pub Weak<Shared>);

impl ClientDirectory for Directory {
    fn authenticate(&self, token: &str) -> Option<AuthorizedClient> {
        let shared = self.0.upgrade()?;
        let stores = shared.stores.as_ref()?;
        let client = stores.metadata.lock().ok()?.authenticate_api_client(token, crate::unix_now()).ok()??;
        Some(AuthorizedClient { client_id: client.client_id.clone(), scopes: client.scopes })
    }

    fn revocation_generation(&self) -> u64 {
        self.0.upgrade().map_or(u64::MAX, |shared| shared.revocations.load(Ordering::SeqCst))
    }
}

pub struct Control(pub Weak<Shared>);

impl Control {
    fn request(&self, request: ApiRequest) -> Result<String, ControlError> {
        let shared = self.0.upgrade().ok_or(ControlError::Unavailable("shutting_down"))?;
        let (reply, receive) = channel();
        shared
            .commands
            .send(Command::Api(request, reply))
            .map_err(|_| ControlError::Unavailable("controller_unavailable"))?;
        receive
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| ControlError::Unavailable("controller_timeout"))?
    }
}

impl Controller for Control {
    fn start(&self) -> Result<String, ControlError> {
        self.request(ApiRequest::Start)
    }

    fn stop(&self, session_id: &str) -> Result<(), ControlError> {
        self.request(ApiRequest::Stop(session_id.to_owned())).map(|_| ())
    }

    fn cancel(&self, session_id: &str) -> Result<(), ControlError> {
        self.request(ApiRequest::Cancel(session_id.to_owned())).map(|_| ())
    }

    fn status(&self) -> StatusSnapshot {
        let status = self.0.upgrade().map(|shared| (shared.status(), shared.settings()));
        let (status, settings) = status.unwrap_or_default();
        StatusSnapshot {
            api_version: dictation_api::API_VERSION,
            recording_state: status.recording_state.to_owned(),
            active_session: matches!(status.recording_state, "recording" | "recording_locked" | "starting" | "finalizing")
                .then_some(status.session_id)
                .flatten(),
            microphone: status.capabilities.map_or("unknown", |capabilities| capabilities.microphone).to_owned(),
            asr_ready: status.asr_ready,
            contribution_enabled: settings.contribution_target != crate::settings::ContributionTarget::Disabled,
        }
    }

    fn live_snapshot(&self) -> Option<Vec<ApiEvent>> {
        // No transcript replay archive: reconnecting clients resume from the
        // next revision of the active session's segments.
        None
    }
}

/// Starts or stops the listener to match settings.
pub fn apply(shared: &Arc<Shared>) {
    let settings = shared.settings();
    let Ok(mut slot) = shared.api.lock() else { return };
    let running_port = slot.as_ref().map(|handle| handle.address.port());
    if !settings.api_enabled || shared.stores.is_none() {
        if let Some(handle) = slot.take() {
            handle.stop();
        }
        drop(slot);
        shared.publish(|status| status.api_listening = None);
        return;
    }
    if running_port == Some(settings.api_port) {
        return;
    }
    if let Some(handle) = slot.take() {
        handle.stop();
    }
    let started = dictation_api::start(
        ApiConfig { port: settings.api_port, per_client_buffer: 256 },
        Arc::new(Directory(Arc::downgrade(shared))),
        Arc::new(Control(Arc::downgrade(shared))),
        shared.bus.clone(),
        Arc::clone(&shared.pairings),
    );
    let port = started.as_ref().ok().map(|handle| handle.address.port());
    *slot = started.ok();
    drop(slot);
    shared.publish(|status| status.api_listening = port);
}
