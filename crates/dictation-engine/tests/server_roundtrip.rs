//! End-to-end contribution flow against a real loopback training server.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use dictation_core::{
    contribution::{CONSENT_VERSION, ConsentRecord, JobState, Purpose, RetrySettings, UploadTarget, VolumeCaps},
    dataset::DatasetSettings,
    privacy::{Classifier, ClassifierFailure, POLICY_VERSION},
    transcript::{FrozenTranscript, Word},
    vad::VadSettings,
};
use dictation_engine::{
    training::{EnqueueOutcome, RawSession, Stores, TrainingSettings, enqueue_session, package_archive, process_next_job},
    upload::{HttpTransport, Transport, UploadOutcome, UploadPolicy, send_deletion_requests, upload_ready},
};
use dictation_protocol::{ConsentGrant, DeletionRequest, DeletionScope};
use dictation_server::{
    http::{AppState, AuthThrottle, router},
    store::ServerStore,
};
use dictation_storage::{EncryptedStore, MasterKeyProvider, contribution::DeletionRequest as StoredDeletion};
use sha2::{Digest, Sha256};

struct Keys;
impl MasterKeyProvider for Keys {
    fn load_or_create(&self) -> Result<Option<[u8; 32]>, String> {
        Ok(Some([8; 32]))
    }
}

struct Clean;
impl Classifier for Clean {
    fn model_id(&self) -> String {
        "clean-test".to_owned()
    }
    fn classify(&mut self, _: &str, _: &str, _: &str, _: u32) -> Result<String, ClassifierFailure> {
        Ok(r#"{"spans":[],"uncertain":false}"#.to_owned())
    }
}

fn now() -> i64 {
    i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()).unwrap()
}

fn session(id: &str, started_at: i64, text: &str) -> RawSession {
    let mut words = Vec::new();
    let mut pcm = vec![20_i16; 8_000];
    for (index, token) in text.split_whitespace().enumerate() {
        let start = pcm.len() as u64;
        pcm.extend((0..6_400).map(|n| if n % 2 == 0 { 9_000 } else { -9_000 }));
        let end = pcm.len() as u64;
        pcm.extend(std::iter::repeat_n(20_i16, 1_600));
        if token.ends_with('.') {
            pcm.extend(std::iter::repeat_n(20_i16, 12_000));
        }
        let spoken = token.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_owned();
        words.push(Word::new(index as u64, spoken, start, end, 0.95).unwrap().with_display(token));
    }
    pcm.extend(std::iter::repeat_n(20_i16, 8_000));
    let total = pcm.len() as u64;
    RawSession {
        session_id: id.to_owned(),
        started_at,
        opted_out: false,
        asr_model: "whisper-base-q5_1".to_owned(),
        asr_model_revision: "r".to_owned(),
        transcript: FrozenTranscript::new(id, 1, words, 16_000, total, "en").unwrap(),
        pcm,
    }
}

fn start_server(root: &std::path::Path) -> (SocketAddr, Arc<Mutex<ServerStore>>) {
    let store = Arc::new(Mutex::new(ServerStore::open(root).unwrap()));
    let state = AppState { store: Arc::clone(&store), throttle: Arc::new(AuthThrottle::default()) };
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, router(state).into_make_service_with_connect_info::<SocketAddr>())
                .await
                .unwrap();
        });
    });
    (receiver.recv().unwrap(), store)
}

fn local_consent(granted_at: i64) -> ConsentRecord {
    ConsentRecord {
        consent_id: "consent-e2e".to_owned(),
        version: CONSENT_VERSION.to_owned(),
        purposes: vec![Purpose::CustomerPersonalization],
        granted_at,
        expires_at: granted_at + 86_400,
        revoked_at: None,
        paused: false,
        policy_version: POLICY_VERSION.to_owned(),
    }
}

fn policy() -> UploadPolicy {
    UploadPolicy { target: UploadTarget::NonProduction, caps: VolumeCaps::default(), retry: RetrySettings::default() }
}

#[test]
#[allow(clippy::too_many_lines)]
fn contribution_round_trip_with_withdrawal_and_isolation() {
    let root = std::env::temp_dir().join(format!("ld-server-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (address, server) = start_server(&root);
    let base = format!("http://{address}");
    let (tenant_a, token_a) = server.lock().unwrap().create_tenant("A", now()).unwrap();
    let (_tenant_b, token_b) = server.lock().unwrap().create_tenant("B", now()).unwrap();

    let started = now() - 10;
    let mut sessions = EncryptedStore::open_in_memory(&Keys).unwrap();
    let mut queue = EncryptedStore::open_in_memory(&Keys).unwrap();
    queue.insert_consent(&local_consent(started - 5)).unwrap();
    let mut transport = HttpTransport::new(&base, &token_a, UploadTarget::NonProduction).unwrap();
    let ack = transport
        .register_consent(&ConsentGrant {
            consent_id: "consent-e2e".to_owned(),
            version: CONSENT_VERSION.to_owned(),
            purposes: vec!["customer_personalization".to_owned()],
            granted_at: started - 5,
            expires_at: started + 86_400,
        })
        .unwrap();
    assert!(ack.active);

    // Production transport is never allowed over plain HTTP.
    assert!(HttpTransport::new(&base, &token_a, UploadTarget::Production).is_err());

    let settings = TrainingSettings {
        target: UploadTarget::NonProduction,
        dataset: DatasetSettings::default(),
        vad: VadSettings::default(),
        private_terms: Vec::new(),
        privacy_model_revision: "r".to_owned(),
    };
    let text = "Send the parcel to Jane at 14 Oak Street. The meeting starts tomorrow.";
    let mut stores = Stores { sessions: &mut sessions, queue: &mut queue };
    let EnqueueOutcome::Enqueued { job_id } =
        enqueue_session(&mut stores, &session("s-1", started, text), UploadTarget::NonProduction, now()).unwrap()
    else {
        panic!("enqueue refused");
    };
    process_next_job(&mut stores, &mut Clean, &settings, || false, now).unwrap();
    let archive = package_archive(&queue, &job_id).unwrap().unwrap();

    let outcomes = upload_ready(&queue, &mut transport, &policy(), now).unwrap();
    let UploadOutcome::Acknowledged { receipt_id, .. } = &outcomes[0] else {
        panic!("{outcomes:?}");
    };
    assert_eq!(queue.job(&job_id).unwrap().unwrap().state, JobState::Acknowledged);
    assert_eq!(server.lock().unwrap().tenant_sample_ids(&tenant_a.tenant_id).unwrap().len(), 1);

    // An idempotent retry returns the first receipt without a second copy.
    let job = queue.job(&job_id).unwrap().unwrap();
    let digest = hex::encode(Sha256::digest(&archive));
    let retry = transport.upload(&archive, &digest, &job.idempotency_key).unwrap();
    assert!(retry.duplicate && retry.accepted);
    assert_eq!(&retry.receipt_id, receipt_id);

    // Another tenant's key space is separate, and its credential cannot
    // resolve tenant A's receipt for deletion.
    let mut other = HttpTransport::new(&base, &token_b, UploadTarget::NonProduction).unwrap();
    let foreign = other.upload(&archive, &digest, &job.idempotency_key).unwrap();
    assert!(!foreign.accepted, "tenant B has no consent and must be refused");
    assert_ne!(foreign.receipt_id, retry.receipt_id);
    let report = other
        .request_deletion(&DeletionRequest {
            request_id: "d-foreign".to_owned(),
            scope: DeletionScope::Receipt { receipt_id: receipt_id.clone() },
        })
        .unwrap();
    assert_eq!(report.deleted_samples, 0);
    assert_eq!(server.lock().unwrap().tenant_sample_ids(&tenant_a.tenant_id).unwrap().len(), 1);

    // A declared checksum that does not match the bytes is a rejection.
    let mismatch = transport.upload(&archive, &"0".repeat(64), "idem-mismatch-1").unwrap();
    assert_eq!(mismatch.reason, "declared_checksum_mismatch");
    // A tampered archive fails server-side validation.
    let mut tampered = archive.clone();
    let riff = tampered.windows(4).position(|window| window == b"RIFF").unwrap();
    tampered[riff + 400] ^= 0x55;
    let tampered_digest = hex::encode(Sha256::digest(&tampered));
    let refused = transport.upload(&tampered, &tampered_digest, "idem-tampered-1").unwrap();
    assert!(!refused.accepted);
    assert!(refused.reason.starts_with("invalid_package"), "{}", refused.reason);

    // Withdrawal deletes what the server received and tombstones it.
    let withdrawal = transport.withdraw_consent("consent-e2e").unwrap();
    assert_eq!(withdrawal.deleted_samples, 1);
    assert!(server.lock().unwrap().tenant_sample_ids(&tenant_a.tenant_id).unwrap().is_empty());
    let replay = transport.upload(&archive, &digest, "idem-replay-00001").unwrap();
    assert!(!replay.accepted);

    // Pending local deletion requests are delivered and completed.
    queue
        .insert_deletion_request(StoredDeletion { request_id: "d-everything", scope: "everything" }, now())
        .unwrap();
    assert_eq!(send_deletion_requests(&queue, &mut transport, now()).unwrap(), 1);

    // Wrong credentials are refused and repeated failures are throttled.
    let mut intruder = HttpTransport::new(&base, "ldt_wrong", UploadTarget::NonProduction).unwrap();
    let mut statuses = Vec::new();
    for attempt in 0..12 {
        statuses.push(intruder.upload(b"x", "y", &format!("idem-intruder-{attempt:04}")));
    }
    assert!(statuses.iter().all(Result::is_err));
    assert!(matches!(
        statuses.last(),
        Some(Err(dictation_engine::upload::TransportError::Retriable(code))) if code == "http_429"
    ));
    let still_valid = transport.withdraw_consent("consent-e2e");
    assert!(still_valid.is_ok(), "valid clients must not be locked out by others' failures");

    let _ = std::fs::remove_dir_all(root);
}
