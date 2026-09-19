//! Security and behavior of the local API against a real loopback listener.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use dictation_api::{
    ApiConfig, ApiEvent, ControlError, Controller, EventBus, EventKind, Pairings, StatusSnapshot,
    server::{AuthorizedClient, ClientDirectory},
    start,
};
use dictation_storage::api_clients::Scope;
use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue};

#[derive(Default)]
struct Directory {
    tokens: Mutex<HashMap<String, Vec<Scope>>>,
    generation: AtomicU64,
}

impl Directory {
    fn add(&self, token: &str, scopes: &[Scope]) {
        self.tokens.lock().unwrap().insert(token.to_owned(), scopes.to_vec());
    }
    fn revoke(&self, token: &str) {
        self.tokens.lock().unwrap().remove(token);
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl ClientDirectory for Directory {
    fn authenticate(&self, token: &str) -> Option<AuthorizedClient> {
        self.tokens.lock().unwrap().get(token).map(|scopes| AuthorizedClient {
            client_id: format!("client-{token}"),
            scopes: scopes.clone(),
        })
    }
    fn revocation_generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }
}

#[derive(Default)]
struct FakeController {
    active: Mutex<Option<String>>,
}

impl Controller for FakeController {
    fn start(&self) -> Result<String, ControlError> {
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            return Err(ControlError::Conflict);
        }
        *active = Some("session-1".to_owned());
        Ok("session-1".to_owned())
    }
    fn stop(&self, id: &str) -> Result<(), ControlError> {
        if self.active.lock().unwrap().as_deref() == Some(id) || id == "session-1" {
            Ok(())
        } else {
            Err(ControlError::NotFound)
        }
    }
    fn cancel(&self, id: &str) -> Result<(), ControlError> {
        self.stop(id)
    }
    fn status(&self) -> StatusSnapshot {
        StatusSnapshot {
            api_version: 1,
            recording_state: "idle".to_owned(),
            active_session: self.active.lock().unwrap().clone(),
            microphone: "available".to_owned(),
            asr_ready: true,
            contribution_enabled: false,
        }
    }
    fn live_snapshot(&self) -> Option<Vec<ApiEvent>> {
        None
    }
}

struct Fixture {
    base: String,
    port: u16,
    directory: Arc<Directory>,
    bus: EventBus,
    pairings: Arc<Pairings>,
    _handle: dictation_api::ApiHandle,
}

fn fixture(buffer: usize) -> Fixture {
    let directory = Arc::new(Directory::default());
    directory.add("caption", &[Scope::TranscriptLive]);
    directory.add("finals", &[Scope::TranscriptFinal, Scope::StatusRead]);
    directory.add("control", &[Scope::SessionControl, Scope::StatusRead]);
    let bus = EventBus::new(buffer);
    let pairings = Arc::new(Pairings::default());
    let handle = start(
        ApiConfig { port: 0, per_client_buffer: buffer },
        directory.clone(),
        Arc::new(FakeController::default()),
        bus.clone(),
        pairings.clone(),
    )
    .unwrap();
    Fixture {
        base: format!("http://127.0.0.1:{}", handle.address.port()),
        port: handle.address.port(),
        directory,
        bus,
        pairings,
        _handle: handle,
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

async fn connect(fixture: &Fixture, token: &str) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    let mut request = format!("ws://127.0.0.1:{}/v1/events", fixture.port).into_client_request().unwrap();
    request
        .headers_mut()
        .insert("authorization", HeaderValue::from_str(&format!("Bearer {token}")).unwrap());
    tokio_tungstenite::connect_async(request).await.unwrap().0
}

async fn next_json(
    socket: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
) -> Option<serde_json::Value> {
    let message = tokio::time::timeout(Duration::from_secs(5), socket.next()).await.ok()??.ok()?;
    serde_json::from_str(message.to_text().ok()?).ok()
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_need_loopback_host_no_origin_and_a_scoped_token() {
    let fixture = fixture(64);
    let http = client();
    let status = |response: reqwest::Response| response.status().as_u16();
    assert_eq!(status(http.get(format!("{}/v1/status", fixture.base)).send().await.unwrap()), 401);
    assert_eq!(
        status(http.get(format!("{}/v1/status", fixture.base)).bearer_auth("control").header("host", "evil.example:80").send().await.unwrap()),
        421
    );
    assert_eq!(
        status(http.get(format!("{}/v1/status", fixture.base)).bearer_auth("control").header("origin", "https://evil.example").send().await.unwrap()),
        403
    );
    // Token in a query string is not a credential.
    assert_eq!(status(http.get(format!("{}/v1/status?token=control", fixture.base)).send().await.unwrap()), 401);
    // Captions do not imply status or control.
    assert_eq!(status(http.get(format!("{}/v1/status", fixture.base)).bearer_auth("caption").send().await.unwrap()), 403);
    assert_eq!(status(http.post(format!("{}/v1/sessions", fixture.base)).bearer_auth("caption").send().await.unwrap()), 403);
    let body: serde_json::Value = http
        .get(format!("{}/v1/status", fixture.base))
        .bearer_auth("control")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(body.get("text").is_none());
    assert_eq!(status(http.post(format!("{}/v1/sessions", fixture.base)).bearer_auth("control").send().await.unwrap()), 200);
    assert_eq!(status(http.post(format!("{}/v1/sessions", fixture.base)).bearer_auth("control").send().await.unwrap()), 409);
    for _ in 0..2 {
        assert_eq!(
            status(http.post(format!("{}/v1/sessions/session-1/stop", fixture.base)).bearer_auth("control").send().await.unwrap()),
            200
        );
    }
    assert_eq!(
        status(http.post(format!("{}/v1/sessions/other/cancel", fixture.base)).bearer_auth("control").send().await.unwrap()),
        404
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn pairing_releases_the_approved_token_once() {
    let fixture = fixture(64);
    let http = client();
    let opened: serde_json::Value = http
        .post(format!("{}/v1/pairings", fixture.base))
        .json(&serde_json::json!({ "client_name": "Overlay", "scopes": ["transcript:live"] }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = opened["pairing_id"].as_str().unwrap().to_owned();
    let secret = opened["poll_secret"].as_str().unwrap().to_owned();
    let pending = fixture.pairings.pending();
    assert_eq!(pending[0].client_name, "Overlay");
    assert_eq!(pending[0].verification_code, opened["verification_code"].as_str().unwrap());
    fixture.directory.add("paired-token", &[Scope::TranscriptLive]);
    assert!(fixture.pairings.approve(&id, "paired-token".to_owned()));
    let poll = |secret: String| {
        let http = http.clone();
        let url = format!("{}/v1/pairings/{id}", fixture.base);
        async move { http.get(url).header("x-pairing-secret", secret).send().await.unwrap() }
    };
    assert_eq!(poll("wrong".to_owned()).await.status().as_u16(), 404);
    let approved: serde_json::Value = poll(secret.clone()).await.json().await.unwrap();
    assert_eq!(approved["token"], "paired-token");
    assert_eq!(poll(secret).await.status().as_u16(), 404);
    let invalid = http
        .post(format!("{}/v1/pairings", fixture.base))
        .json(&serde_json::json!({ "client_name": "x", "scopes": ["admin:all"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(invalid.status().as_u16(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_filter_by_scope_and_close_on_revocation() {
    let fixture = fixture(64);
    let mut captions = connect(&fixture, "caption").await;
    let mut finals = connect(&fixture, "finals").await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    fixture.bus.publish(ApiEvent::session(EventKind::SessionStarted, "s", 1, None));
    fixture.bus.publish(ApiEvent::partial("s", 2, "segment-0", 1, 0, 800, "hello"));
    fixture.bus.publish(ApiEvent::final_transcript("s", 3, 900, "Hello."));
    let first = next_json(&mut captions).await.unwrap();
    assert_eq!(first["event"], "transcript.partial");
    assert_eq!(first["privacy"], "unredacted");
    assert_eq!(first["revision"], 1);
    assert_eq!(next_json(&mut finals).await.unwrap()["event"], "session.started");
    assert_eq!(next_json(&mut finals).await.unwrap()["event"], "transcript.final");
    fixture.directory.revoke("caption");
    fixture.bus.publish(ApiEvent::partial("s", 4, "segment-0", 2, 0, 900, "hello there"));
    let revoked = next_json(&mut captions).await.unwrap();
    assert_eq!(revoked["code"], "token_revoked");
}

#[tokio::test(flavor = "multi_thread")]
async fn slow_consumers_are_told_to_resynchronize() {
    let fixture = fixture(4);
    let mut captions = connect(&fixture, "caption").await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    for seq in 0..200 {
        fixture.bus.publish(ApiEvent::partial("s", seq, "segment-0", seq, 0, 10, "x"));
    }
    let mut saw_resync = false;
    while let Some(message) = next_json(&mut captions).await {
        if message["code"] == "resync_required" {
            saw_resync = true;
            break;
        }
    }
    assert!(saw_resync);
}
