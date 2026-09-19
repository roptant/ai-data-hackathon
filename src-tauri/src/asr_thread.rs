//! Owns the recognition worker. Requests carry the session epoch so a late
//! result from a cancelled or replaced session is recognized as stale.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
    time::Duration,
};

use dictation_core::asr::RecognizedSegment;
use dictation_engine::workers::{AsrEngine, ModelVerifier, WorkerPaths};
use dictation_models::delivery::PersonalModels;

use crate::{controller::Command, state::Shared};

pub enum AsrRequest {
    Warm,
    Partial { epoch: u64, pcm: Vec<i16>, offset: u64, audio_end: u64 },
    Final { epoch: u64, pcm: Vec<i16> },
    /// Frees model memory (low-memory scheduling before privacy work).
    Release,
    /// Model choice or personalized model changed.
    Reload,
}

pub type AsrResult = Result<(String, Vec<RecognizedSegment>), String>;

#[derive(Clone)]
pub struct AsrHandle {
    sender: Sender<AsrRequest>,
}

impl AsrHandle {
    pub fn send(&self, request: AsrRequest) {
        let _ = self.sender.send(request);
    }
}

pub fn spawn(shared: Arc<Shared>, workers: WorkerPaths, verifier: Arc<ModelVerifier>) -> AsrHandle {
    let (sender, receiver) = channel();
    std::thread::Builder::new()
        .name("asr".to_owned())
        .spawn(move || run(&shared, &workers, &verifier, &receiver))
        .expect("the ASR thread can be spawned");
    AsrHandle { sender }
}

fn build_engine(shared: &Shared, workers: &WorkerPaths, verifier: &ModelVerifier) -> Result<AsrEngine, String> {
    let settings = shared.settings();
    let spec = dictation_models::spec(&settings.asr_model)
        .unwrap_or_else(|| dictation_models::default_for(dictation_models::Role::Asr));
    let personal: Option<PathBuf> = PersonalModels::new(&shared.layout.models()).active();
    AsrEngine::new(workers, &shared.layout.models(), spec, verifier, personal.as_deref()).map_err(|error| error.to_string())
}

fn vocabulary_prompt(shared: &Shared) -> Option<String> {
    let vocabulary = shared.settings().vocabulary;
    (!vocabulary.is_empty()).then(|| format!("Vocabulary: {}.", vocabulary.join(", ")))
}

fn language(shared: &Shared) -> Option<String> {
    let language = shared.settings().language;
    (!language.is_empty()).then_some(language)
}

fn run(shared: &Shared, workers: &WorkerPaths, verifier: &ModelVerifier, receiver: &Receiver<AsrRequest>) {
    let mut engine: Option<AsrEngine> = None;
    let ensure = |engine: &mut Option<AsrEngine>| -> Result<(), String> {
        if engine.is_none() {
            shared.publish(|status| status.asr_loading = true);
            let built = build_engine(shared, workers, verifier);
            let ready = built.and_then(|mut built| {
                built.warm().map_err(|error| error.to_string())?;
                Ok(built)
            });
            match ready {
                Ok(built) => {
                    *engine = Some(built);
                    shared.publish(|status| {
                        status.asr_loading = false;
                        status.asr_ready = true;
                    });
                }
                Err(error) => {
                    shared.publish(|status| {
                        status.asr_loading = false;
                        status.asr_ready = false;
                    });
                    return Err(error);
                }
            }
        }
        Ok(())
    };
    while let Ok(request) = receiver.recv() {
        match request {
            AsrRequest::Warm => {
                let _ = ensure(&mut engine);
            }
            AsrRequest::Release => {
                if let Some(mut built) = engine.take() {
                    built.release();
                }
                shared.publish(|status| status.asr_ready = false);
            }
            AsrRequest::Reload => {
                if let Some(mut built) = engine.take() {
                    built.release();
                }
                let _ = ensure(&mut engine);
            }
            AsrRequest::Partial { epoch, pcm, offset, audio_end } => {
                let result = ensure(&mut engine).and_then(|()| {
                    engine
                        .as_mut()
                        .ok_or_else(|| "asr_unavailable".to_owned())?
                        .transcribe(&pcm, false, language(shared).as_deref(), vocabulary_prompt(shared).as_deref(), Duration::from_secs(20))
                        .map_err(|error| error.to_string())
                });
                let _ = shared.commands.send(Command::AsrPartial { epoch, result, offset, audio_end });
            }
            AsrRequest::Final { epoch, pcm } => {
                // Final passes can be long; allow up to 3x real time plus load.
                let seconds = (pcm.len() as u64 / 16_000).saturating_mul(3).max(30);
                let result = ensure(&mut engine).and_then(|()| {
                    engine
                        .as_mut()
                        .ok_or_else(|| "asr_unavailable".to_owned())?
                        .transcribe(&pcm, true, language(shared).as_deref(), vocabulary_prompt(shared).as_deref(), Duration::from_secs(seconds))
                        .map_err(|error| error.to_string())
                });
                let _ = shared.commands.send(Command::AsrFinal { epoch, result });
            }
        }
    }
}
