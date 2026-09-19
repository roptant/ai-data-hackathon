//! HTTP boundary. Deploy behind a TLS-terminating proxy with authenticated
//! TLS; this listener refuses non-loopback binds unless explicitly allowed.
//! Handlers log reason codes and identifiers only, never request bodies.

use std::{
    collections::BTreeMap,
    net::{IpAddr, SocketAddr},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{ConnectInfo, DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dictation_protocol::{
    CONTENT_SHA256_HEADER, ConsentAck, ConsentGrant, DeletionReport, DeletionRequest,
    DeletionScope, ErrorBody, IDEMPOTENCY_HEADER, MAX_ARCHIVE_BYTES, WithdrawalAck,
};
use serde_json::json;

use crate::{
    admission::admit,
    store::{ServerStore, Tenant},
};

#[must_use]
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX))
}

/// Failed-authentication throttle keyed by client address. Successful
/// requests never count, so valid clients are not locked out by traffic.
#[derive(Default)]
pub struct AuthThrottle {
    failures: Mutex<BTreeMap<IpAddr, (Instant, u32)>>,
}

const FAILURE_WINDOW: Duration = Duration::from_secs(60);
const MAX_FAILURES: u32 = 10;

impl AuthThrottle {
    fn blocked(&self, address: IpAddr) -> bool {
        self.failures.lock().is_ok_and(|failures| {
            failures
                .get(&address)
                .is_some_and(|(since, count)| since.elapsed() < FAILURE_WINDOW && *count >= MAX_FAILURES)
        })
    }

    fn record_failure(&self, address: IpAddr) {
        if let Ok(mut failures) = self.failures.lock() {
            let entry = failures.entry(address).or_insert((Instant::now(), 0));
            if entry.0.elapsed() >= FAILURE_WINDOW {
                *entry = (Instant::now(), 0);
            }
            entry.1 += 1;
            if failures.len() > 10_000 {
                failures.retain(|_, (since, _)| since.elapsed() < FAILURE_WINDOW);
            }
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<ServerStore>>,
    pub throttle: Arc<AuthThrottle>,
}

fn error(status: StatusCode, code: &str) -> Response {
    (status, Json(ErrorBody { error: code.to_owned() })).into_response()
}

fn internal() -> Response {
    error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
}

/// A valid credential always authenticates: throttling applies only to
/// failing attempts, so others' failures from a shared address (NAT, proxy)
/// never lock a valid client out.
async fn authenticate(state: &AppState, headers: &HeaderMap, address: SocketAddr) -> Result<Tenant, Response> {
    let refuse = || {
        let blocked = state.throttle.blocked(address.ip());
        state.throttle.record_failure(address.ip());
        if blocked {
            error(StatusCode::TOO_MANY_REQUESTS, "too_many_failed_attempts")
        } else {
            error(StatusCode::UNAUTHORIZED, "unauthorized")
        }
    };
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .filter(|token| token.len() <= 256)
        .map(str::to_owned);
    let Some(token) = token else {
        return Err(refuse());
    };
    let store = Arc::clone(&state.store);
    let tenant = tokio::task::spawn_blocking(move || {
        store.lock().ok().and_then(|store| store.authenticate(&token).ok().flatten())
    })
    .await
    .ok()
    .flatten();
    tenant.ok_or_else(refuse)
}

async fn blocking<T: Send + 'static>(
    state: &AppState,
    work: impl FnOnce(&mut ServerStore) -> Result<T, crate::store::ServerError> + Send + 'static,
) -> Result<T, Response> {
    let store = Arc::clone(&state.store);
    tokio::task::spawn_blocking(move || {
        let mut guard = store.lock().map_err(|_| ())?;
        work(&mut guard).map_err(|_| ())
    })
    .await
    .ok()
    .and_then(Result::ok)
    .ok_or_else(internal)
}

async fn post_sample(
    State(state): State<AppState>,
    ConnectInfo(address): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let tenant = match authenticate(&state, &headers, address).await {
        Ok(tenant) => tenant,
        Err(response) => return response,
    };
    let header_text = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned()
    };
    let key = header_text(IDEMPOTENCY_HEADER);
    let declared = header_text(CONTENT_SHA256_HEADER);
    match blocking(&state, move |store| admit(store, &tenant, &key, &declared, &body, unix_now())).await {
        Ok(receipt) => Json(receipt).into_response(),
        Err(response) => response,
    }
}

async fn post_consent(
    State(state): State<AppState>,
    ConnectInfo(address): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(grant): Json<ConsentGrant>,
) -> Response {
    let tenant = match authenticate(&state, &headers, address).await {
        Ok(tenant) => tenant,
        Err(response) => return response,
    };
    if grant.purposes.iter().any(|purpose| purpose != "customer_personalization")
        || grant.consent_id.len() > 128
    {
        return error(StatusCode::BAD_REQUEST, "purpose_not_grantable");
    }
    let tenant_id = tenant.tenant_id;
    let consent_id = grant.consent_id.clone();
    match blocking(&state, move |store| {
        store.record_consent(&tenant_id, &grant)?;
        store.consent_active(&tenant_id, &grant.consent_id, unix_now())
    })
    .await
    {
        Ok(active) => Json(ConsentAck { consent_id, active }).into_response(),
        Err(response) => response,
    }
}

async fn withdraw_consent(
    State(state): State<AppState>,
    ConnectInfo(address): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Path(consent_id): Path<String>,
) -> Response {
    let tenant = match authenticate(&state, &headers, address).await {
        Ok(tenant) => tenant,
        Err(response) => return response,
    };
    let tenant_id = tenant.tenant_id;
    let id = consent_id.clone();
    // Withdrawal stops new training and deletes everything received under it.
    match blocking(&state, move |store| {
        let now = unix_now();
        store.withdraw_consent(&tenant_id, &id, now)?;
        let cancelled = store.connection.execute(
            "UPDATE training_jobs SET state='cancelled', reason='consent_withdrawn', finished_at=?2
             WHERE tenant_id=?1 AND finished_at IS NULL",
            rusqlite::params![tenant_id, now],
        )?;
        let samples: Vec<String> = {
            let mut statement = store
                .connection
                .prepare("SELECT sample_id FROM samples WHERE tenant_id=?1 AND consent_id=?2")?;
            statement
                .query_map(rusqlite::params![tenant_id, id], |row| row.get(0))?
                .collect::<Result<_, _>>()?
        };
        let (deleted, _) = store.delete_samples(&tenant_id, &samples, now)?;
        Ok((u32::try_from(cancelled).unwrap_or(u32::MAX), deleted))
    })
    .await
    {
        Ok((cancelled, deleted)) => Json(WithdrawalAck {
            consent_id,
            cancelled_training_jobs: cancelled,
            deleted_samples: deleted,
        })
        .into_response(),
        Err(response) => response,
    }
}

async fn post_deletion(
    State(state): State<AppState>,
    ConnectInfo(address): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(request): Json<DeletionRequest>,
) -> Response {
    let tenant = match authenticate(&state, &headers, address).await {
        Ok(tenant) => tenant,
        Err(response) => return response,
    };
    let tenant_id = tenant.tenant_id;
    let request_id = request.request_id.clone();
    match blocking(&state, move |store| {
        let now = unix_now();
        let (scope_label, (samples, artifacts)) = match &request.scope {
            DeletionScope::Everything => ("everything".to_owned(), store.delete_everything(&tenant_id, now)?),
            DeletionScope::Sample { sample_id } => (
                "sample".to_owned(),
                store.delete_samples(&tenant_id, std::slice::from_ref(sample_id), now)?,
            ),
            DeletionScope::Receipt { receipt_id } => {
                let sample = store.sample_for_receipt(&tenant_id, receipt_id)?;
                (
                    "receipt".to_owned(),
                    match sample {
                        Some(sample_id) => store.delete_samples(&tenant_id, &[sample_id], now)?,
                        None => (0, 0),
                    },
                )
            }
        };
        store.log_deletion(&tenant_id, &request.request_id, &scope_label, samples, artifacts, now)?;
        Ok((samples, artifacts))
    })
    .await
    {
        Ok((samples, artifacts)) => Json(DeletionReport {
            request_id,
            completed: true,
            deleted_samples: samples,
            deleted_artifacts: artifacts,
            notes: vec![
                "Encrypted backups expire on their own schedule; tombstones prevent restored samples from re-entering training.".to_owned(),
                "Copies exported outside the service cannot be recalled.".to_owned(),
                "Deleting samples removes affected personalized models entirely; no unlearning is claimed.".to_owned(),
            ],
        })
        .into_response(),
        Err(response) => response,
    }
}

async fn latest_model(
    State(state): State<AppState>,
    ConnectInfo(address): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Response {
    let tenant = match authenticate(&state, &headers, address).await {
        Ok(tenant) => tenant,
        Err(response) => return response,
    };
    match blocking(&state, move |store| store.latest_delivery(&tenant.tenant_id)).await {
        Ok(Some((version, signed))) => Json(json!({ "model_version": version, "signed": signed })).into_response(),
        Ok(None) => Json(json!({ "model_version": null, "signed": null })).into_response(),
        Err(response) => response,
    }
}

async fn model_artifact(
    State(state): State<AppState>,
    ConnectInfo(address): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Path(version): Path<String>,
) -> Response {
    let tenant = match authenticate(&state, &headers, address).await {
        Ok(tenant) => tenant,
        Err(response) => return response,
    };
    match blocking(&state, move |store| store.delivery_artifact(&tenant.tenant_id, &version)).await {
        Ok(Some(bytes)) => ([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response(),
        Ok(None) => error(StatusCode::NOT_FOUND, "not_found"),
        Err(response) => response,
    }
}

async fn health() -> &'static str {
    "ok"
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/samples", post(post_sample))
        .route("/v1/consents", post(post_consent))
        .route("/v1/consents/{consent_id}/withdraw", post(withdraw_consent))
        .route("/v1/deletion-requests", post(post_deletion))
        .route("/v1/models/latest", get(latest_model))
        .route("/v1/models/{version}/artifact", get(model_artifact))
        .layer(DefaultBodyLimit::max(MAX_ARCHIVE_BYTES))
        .with_state(state)
}
