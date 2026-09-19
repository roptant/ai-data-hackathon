//! The loopback listener, request guards, and WebSocket event streams.

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use axum::{
    Json, Router,
    extract::{
        FromRequestParts, Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode, header, request::Parts},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dictation_storage::api_clients::Scope;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast::error::RecvError, oneshot, watch};

use crate::{
    API_VERSION,
    events::{ApiEvent, EventBus, EventKind},
    pairing::{Pairings, PollResult},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedClient {
    pub client_id: String,
    pub scopes: Vec<Scope>,
}

impl AuthorizedClient {
    #[must_use]
    pub fn has(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }
}

/// Paired-client lookup, implemented by the desktop app over its store.
pub trait ClientDirectory: Send + Sync + 'static {
    fn authenticate(&self, token: &str) -> Option<AuthorizedClient>;
    /// Bumped on every revocation so open streams recheck their token.
    fn revocation_generation(&self) -> u64;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlError {
    Conflict,
    NotFound,
    Unavailable(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatusSnapshot {
    pub api_version: u32,
    pub recording_state: String,
    pub active_session: Option<String>,
    pub microphone: String,
    pub asr_ready: bool,
    pub contribution_enabled: bool,
}

/// Session control and state, implemented by the desktop coordinator.
pub trait Controller: Send + Sync + 'static {
    /// Starts a visible recording; `Conflict` when one is already active.
    ///
    /// # Errors
    ///
    /// Returns a control error.
    fn start(&self) -> Result<String, ControlError>;
    /// Idempotent finalize request for the active session.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` for another session.
    fn stop(&self, session_id: &str) -> Result<(), ControlError>;
    /// Idempotent cancellation: no insertion or contribution.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` for another session.
    fn cancel(&self, session_id: &str) -> Result<(), ControlError>;
    fn status(&self) -> StatusSnapshot;
    /// Current provisional segments of the active session, for reconnects.
    fn live_snapshot(&self) -> Option<Vec<ApiEvent>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiConfig {
    /// 0 picks an ephemeral port.
    pub port: u16,
    pub per_client_buffer: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            port: 8765,
            per_client_buffer: 256,
        }
    }
}

#[derive(Clone)]
struct ApiState {
    directory: Arc<dyn ClientDirectory>,
    controller: Arc<dyn Controller>,
    bus: EventBus,
    pairings: Arc<Pairings>,
    port: u16,
    closing: watch::Receiver<bool>,
}

fn error(status: StatusCode, code: &str) -> Response {
    (status, Json(serde_json::json!({ "error": code }))).into_response()
}

/// Rejects DNS-rebinding and browser-originated requests.
#[allow(clippy::result_large_err)]
fn guard(headers: &HeaderMap, port: u16) -> Result<(), Response> {
    let host = headers.get(header::HOST).and_then(|value| value.to_str().ok()).unwrap_or_default();
    let allowed = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    if !allowed.iter().any(|candidate| candidate == host) {
        return Err(error(StatusCode::MISDIRECTED_REQUEST, "invalid_host"));
    }
    if headers.contains_key(header::ORIGIN) || headers.contains_key("sec-fetch-site") {
        return Err(error(StatusCode::FORBIDDEN, "browser_origin_denied"));
    }
    Ok(())
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .filter(|token| !token.is_empty() && token.len() <= 256)
}

/// Extractor: guarded and authenticated client.
struct Client(AuthorizedClient, String);

impl FromRequestParts<ApiState> for Client {
    type Rejection = Response;

    #[allow(clippy::unused_async_trait_impl)]
    async fn from_request_parts(parts: &mut Parts, state: &ApiState) -> Result<Self, Self::Rejection> {
        guard(&parts.headers, state.port)?;
        let token = bearer(&parts.headers).ok_or_else(|| error(StatusCode::UNAUTHORIZED, "unauthorized"))?;
        let client = state
            .directory
            .authenticate(token)
            .ok_or_else(|| error(StatusCode::UNAUTHORIZED, "unauthorized"))?;
        Ok(Self(client, token.to_owned()))
    }
}

#[allow(clippy::result_large_err)]
fn require(client: &AuthorizedClient, scope: Scope) -> Result<(), Response> {
    if client.has(scope) {
        Ok(())
    } else {
        Err(error(StatusCode::FORBIDDEN, "insufficient_scope"))
    }
}

async fn status(State(state): State<ApiState>, Client(client, _): Client) -> Response {
    if let Err(response) = require(&client, Scope::StatusRead) {
        return response;
    }
    let snapshot = state.controller.status();
    // Preserve the Rust snapshot fields while providing the v1 envelope used
    // by the standalone Live Caption client. Capabilities belong to this token.
    let mut value = serde_json::to_value(&snapshot).expect("status is serializable");
    value["version"] = serde_json::json!(snapshot.api_version);
    value["capabilities"] = serde_json::json!(client.scopes.iter().map(|scope| scope.as_str()).collect::<Vec<_>>());
    value["state"] = serde_json::json!({
        "state": snapshot.recording_state,
        "session_id": snapshot.active_session.as_deref().unwrap_or(""),
        "recording": matches!(snapshot.recording_state.as_str(), "recording" | "recording_held" | "recording_locked"),
        "locked": snapshot.recording_state == "recording_locked",
        "partials_available": true,
    });
    Json(value).into_response()
}

fn control_response(result: Result<serde_json::Value, ControlError>) -> Response {
    match result {
        Ok(body) => Json(body).into_response(),
        Err(ControlError::Conflict) => error(StatusCode::CONFLICT, "session_already_active"),
        Err(ControlError::NotFound) => error(StatusCode::NOT_FOUND, "no_such_active_session"),
        Err(ControlError::Unavailable(code)) => error(StatusCode::SERVICE_UNAVAILABLE, code),
    }
}

async fn start_session(State(state): State<ApiState>, Client(client, _): Client) -> Response {
    if let Err(response) = require(&client, Scope::SessionControl) {
        return response;
    }
    control_response(state.controller.start().map(|id| serde_json::json!({ "session_id": id })))
}

async fn stop_session(State(state): State<ApiState>, Client(client, _): Client, Path(id): Path<String>) -> Response {
    if let Err(response) = require(&client, Scope::SessionControl) {
        return response;
    }
    control_response(state.controller.stop(&id).map(|()| serde_json::json!({ "session_id": id, "stopping": true })))
}

async fn cancel_session(State(state): State<ApiState>, Client(client, _): Client, Path(id): Path<String>) -> Response {
    if let Err(response) = require(&client, Scope::SessionControl) {
        return response;
    }
    control_response(state.controller.cancel(&id).map(|()| serde_json::json!({ "session_id": id, "cancelled": true })))
}

#[derive(Debug, Deserialize)]
struct PairBody {
    client_name: String,
    scopes: Vec<String>,
}

async fn open_pairing(State(state): State<ApiState>, headers: HeaderMap, Json(body): Json<PairBody>) -> Response {
    if let Err(response) = guard(&headers, state.port) {
        return response;
    }
    let scopes: Option<Vec<Scope>> = body.scopes.iter().map(|scope| Scope::parse(scope)).collect();
    let Some(scopes) = scopes.filter(|scopes| !scopes.is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "invalid_scopes");
    };
    match state.pairings.open(&body.client_name, scopes) {
        Some((request, secret)) => Json(serde_json::json!({
            "pairing_id": request.pairing_id,
            "verification_code": request.verification_code,
            "poll_secret": secret,
            "instructions": "Approve this client in Local Dictation; compare the verification code.",
        }))
        .into_response(),
        None => error(StatusCode::TOO_MANY_REQUESTS, "pairing_rate_limited"),
    }
}

async fn poll_pairing(State(state): State<ApiState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(response) = guard(&headers, state.port) {
        return response;
    }
    let secret = headers.get("x-pairing-secret").and_then(|value| value.to_str().ok()).unwrap_or_default();
    match state.pairings.poll(&id, secret) {
        PollResult::Pending => Json(serde_json::json!({ "status": "pending" })).into_response(),
        PollResult::Approved { token } => Json(serde_json::json!({ "status": "approved", "token": token })).into_response(),
        PollResult::Denied => Json(serde_json::json!({ "status": "denied" })).into_response(),
        PollResult::Unknown => error(StatusCode::NOT_FOUND, "unknown_pairing"),
    }
}

#[derive(Debug, Deserialize)]
struct Subscription {
    /// Optional comma-separated event names; never credentials.
    events: Option<String>,
}

async fn events(
    State(state): State<ApiState>,
    Client(client, token): Client,
    Query(subscription): Query<Subscription>,
    upgrade: WebSocketUpgrade,
) -> Response {
    if !(client.has(Scope::StatusRead) || client.has(Scope::TranscriptLive) || client.has(Scope::TranscriptFinal)) {
        return error(StatusCode::FORBIDDEN, "insufficient_scope");
    }
    let wanted: Option<Vec<String>> = subscription
        .events
        .map(|list| list.split(',').map(str::trim).map(str::to_owned).collect());
    upgrade
        .max_message_size(4 * 1024)
        .on_upgrade(move |socket| stream(socket, state, client, token, wanted))
}

fn event_name(kind: EventKind) -> &'static str {
    match kind {
        EventKind::SessionStarted => "session.started",
        EventKind::TranscriptPartial => "transcript.partial",
        EventKind::TranscriptFinal => "transcript.final",
        EventKind::SessionStopped => "session.stopped",
        EventKind::SessionCancelled => "session.cancelled",
        EventKind::Error => "error",
    }
}

async fn send_json(socket: &mut WebSocket, value: &impl Serialize) -> bool {
    match serde_json::to_string(value) {
        Ok(text) => socket.send(Message::Text(text.into())).await.is_ok(),
        Err(_) => false,
    }
}

async fn stream(
    mut socket: WebSocket,
    state: ApiState,
    mut client: AuthorizedClient,
    token: String,
    wanted: Option<Vec<String>>,
) {
    let mut receiver = state.bus.subscribe();
    let mut generation = state.directory.revocation_generation();
    let permitted = |client: &AuthorizedClient, event: &ApiEvent| {
        client.has(event.event.scope())
            && wanted.as_ref().is_none_or(|names| names.iter().any(|name| name == event_name(event.event)))
    };
    // Reconnect: the authorized active-session snapshot, never history.
    if let Some(snapshot) = state.controller.live_snapshot() {
        for event in snapshot.into_iter().filter(|event| permitted(&client, event)) {
            if !send_json(&mut socket, &event).await {
                return;
            }
        }
    }
    let mut revocation_check = tokio::time::interval(Duration::from_secs(2));
    let mut closing = state.closing.clone();
    loop {
        tokio::select! {
            _ = closing.changed() => {
                let _ = socket.send(Message::Close(None)).await;
                return;
            }
            event_result = receiver.recv() => {
                let current = state.directory.revocation_generation();
                if current != generation {
                    generation = current;
                    if let Some(updated) = state.directory.authenticate(&token) {
                        client = updated;
                    } else {
                        let _ = send_json(&mut socket, &serde_json::json!({ "version": API_VERSION, "event": "error", "code": "token_revoked" })).await;
                        let _ = socket.send(Message::Close(None)).await;
                        return;
                    }
                }
                match event_result {
                    Ok(event) => {
                        if permitted(&client, &event) && !send_json(&mut socket, &event).await {
                            return;
                        }
                    }
                    Err(RecvError::Lagged(_)) => {
                        let _ = send_json(&mut socket, &serde_json::json!({ "version": API_VERSION, "event": "error", "code": "resync_required" })).await;
                        let _ = socket.send(Message::Close(None)).await;
                        return;
                    }
                    Err(RecvError::Closed) => return,
                }
            }
            _ = revocation_check.tick() => {
                let current = state.directory.revocation_generation();
                if current != generation {
                    generation = current;
                    if let Some(updated) = state.directory.authenticate(&token) {
                        client = updated;
                    } else {
                        let _ = send_json(&mut socket, &serde_json::json!({ "version": API_VERSION, "event": "error", "code": "token_revoked" })).await;
                        let _ = socket.send(Message::Close(None)).await;
                        return;
                    }
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Close(_)) | Err(_)) | None => return,
                    _ => {}
                }
            }
        }
    }
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/status", get(status))
        .route("/v1/events", get(events))
        .route("/v1/sessions", post(start_session))
        .route("/v1/sessions/{id}/stop", post(stop_session))
        .route("/v1/sessions/{id}/cancel", post(cancel_session))
        .route("/v1/pairings", post(open_pairing))
        .route("/v1/pairings/{id}", get(poll_pairing))
        .layer(axum::extract::DefaultBodyLimit::max(8 * 1024))
        .with_state(state)
}

pub struct ApiHandle {
    pub address: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ApiHandle {
    /// Stops the listener and closes every stream.
    pub fn stop(mut self) {
        self.shutdown_now();
    }

    fn shutdown_now(&mut self) {
        if let Some(sender) = self.shutdown.take() {
            let _ = sender.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for ApiHandle {
    fn drop(&mut self) {
        self.shutdown_now();
    }
}

/// Starts the API on its own runtime thread, bound to 127.0.0.1 only.
///
/// # Errors
///
/// Fails when the runtime or listener cannot start.
pub fn start(
    config: ApiConfig,
    directory: Arc<dyn ClientDirectory>,
    controller: Arc<dyn Controller>,
    bus: EventBus,
    pairings: Arc<Pairings>,
) -> std::io::Result<ApiHandle> {
    // Bind synchronously so callers inside an async runtime can start the API.
    let std_listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, config.port))?;
    std_listener.set_nonblocking(true)?;
    let address = std_listener.local_addr()?;
    let (closing_sender, closing) = watch::channel(false);
    let state = ApiState {
        directory,
        controller,
        bus,
        pairings,
        port: address.port(),
        closing,
    };
    let (shutdown, signal) = oneshot::channel::<()>();
    let thread = std::thread::Builder::new()
        .name("local-api".to_owned())
        .spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                let Ok(listener) = tokio::net::TcpListener::from_std(std_listener) else {
                    return;
                };
                let _ = axum::serve(listener, router(state))
                    .with_graceful_shutdown(async move {
                        let _ = signal.await;
                        let _ = closing_sender.send(true);
                    })
                    .await;
            });
        })?;
    Ok(ApiHandle {
        address,
        shutdown: Some(shutdown),
        thread: Some(thread),
    })
}
