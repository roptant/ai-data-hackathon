//! Hands a delivered session's training copy to the encrypted queue.
//!
//! Only after delivery, only with consent (checked inside `enqueue_session`),
//! never for sessions the user marked "do not contribute". With contribution
//! off, the audio is simply dropped here.

use dictation_core::asr::{RecognizedSegment, freeze};
use dictation_engine::training::{EnqueueOutcome, RawSession, Stores as EngineStores, enqueue_session};

use crate::{settings::ContributionTarget, state::Shared};

pub fn maybe_enqueue(
    shared: &Shared,
    session_id: &str,
    started_at: i64,
    language: &str,
    segments: &[RecognizedSegment],
    pcm: Vec<i16>,
) {
    let settings = shared.settings();
    if settings.contribution_target == ContributionTarget::Disabled || segments.is_empty() || pcm.is_empty() {
        return;
    }
    let opted_out = shared
        .opted_out_sessions
        .lock()
        .is_ok_and(|sessions| sessions.iter().any(|id| id == session_id));
    let Some(stores) = &shared.stores else { return };
    let Ok(transcript) = freeze(session_id, 1, segments, 16_000, pcm.len() as u64, language) else {
        return;
    };
    let session = RawSession {
        session_id: session_id.to_owned(),
        started_at,
        opted_out,
        asr_model: settings.asr_model.clone(),
        asr_model_revision: dictation_models::spec(&settings.asr_model).map_or_else(String::new, |spec| spec.revision.to_owned()),
        transcript,
        pcm,
    };
    let (Ok(mut sessions), Ok(mut queue)) = (stores.sessions.lock(), stores.queue.lock()) else {
        return;
    };
    let mut engine_stores = EngineStores { sessions: &mut sessions, queue: &mut queue };
    let outcome = enqueue_session(&mut engine_stores, &session, settings.contribution_target.upload_target(), crate::unix_now());
    drop(engine_stores);
    let label = match outcome {
        Ok(EnqueueOutcome::Enqueued { .. }) => "queued_for_privacy_filtering".to_owned(),
        Ok(EnqueueOutcome::Refused(refusal)) => format!("not_contributed:{}", refusal.code()),
        Err(_) => "storage_error".to_owned(),
    };
    shared.publish(|status| status.training_status = label);
    let _ = shared.background_wake.send(());
}
