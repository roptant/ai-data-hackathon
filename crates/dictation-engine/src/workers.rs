//! Supervision of the two private model workers.
//!
//! Model files are checksum-verified once per process before a worker may load
//! them. Workers run with a cleared environment, discarded diagnostics, and
//! self-applied confinement whose result is reported, not assumed.

use std::{
    collections::BTreeMap,
    ffi::OsString,
    fmt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime},
};

use dictation_core::{
    asr::{RecognizedSegment, RecognizedWord},
    privacy::{Classifier, ClassifierFailure},
};
use dictation_models::{ModelSpec, installed_path, verify};
use dictation_worker::{
    ProtocolError,
    messages::{ClassifyRequest, Ready, Request, Response, TranscribeRequest},
    persistent::{PersistentWorker, WorkerSpec},
    serve::{MAXIMUM_AUDIO_BYTES, MAXIMUM_CONTROL_BYTES},
};

pub const ASR_WORKER: &str = "dictation-asr-worker";
pub const PRIVACY_WORKER: &str = "dictation-privacy-worker";

#[derive(Debug)]
pub enum EngineError {
    WorkerMissing(PathBuf),
    ModelMissing(&'static str),
    ModelVerification(String),
    Worker(ProtocolError),
    WorkerRefused(String),
    UnexpectedResponse,
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerMissing(path) => write!(formatter, "worker binary missing: {}", path.display()),
            Self::ModelMissing(id) => write!(formatter, "model {id} is not installed"),
            Self::ModelVerification(error) => write!(formatter, "model verification failed: {error}"),
            Self::Worker(error) => write!(formatter, "{error}"),
            Self::WorkerRefused(code) => write!(formatter, "worker refused request: {code}"),
            Self::UnexpectedResponse => write!(formatter, "worker sent an unexpected response"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<ProtocolError> for EngineError {
    fn from(error: ProtocolError) -> Self {
        Self::Worker(error)
    }
}

/// Worker executables bundled next to the application binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerPaths {
    pub asr: PathBuf,
    pub privacy: PathBuf,
}

impl WorkerPaths {
    /// Looks beside `executable_dir` for the platform-named sidecars, including
    /// the target-triple suffix Tauri uses for bundled external binaries.
    #[must_use]
    pub fn locate(executable_dir: &Path) -> Self {
        let find = |stem: &str| {
            let suffix = std::env::consts::EXE_SUFFIX;
            let plain = executable_dir.join(format!("{stem}{suffix}"));
            if plain.exists() {
                return plain;
            }
            let triple = option_env!("TARGET_TRIPLE").unwrap_or("");
            let tagged = executable_dir.join(format!("{stem}-{triple}{suffix}"));
            if tagged.exists() { tagged } else { plain }
        };
        Self {
            asr: find(ASR_WORKER),
            privacy: find(PRIVACY_WORKER),
        }
    }
}

type VerificationCache = Mutex<BTreeMap<PathBuf, (u64, SystemTime)>>;

/// Verifies model files once per process, keyed by size and mtime so a file
/// replaced on disk is verified again.
#[derive(Debug, Default)]
pub struct ModelVerifier {
    verified: VerificationCache,
}

impl ModelVerifier {
    /// # Errors
    ///
    /// Fails when the model is absent or does not match its pin.
    pub fn verified_path(&self, models_dir: &Path, spec: &ModelSpec) -> Result<PathBuf, EngineError> {
        let path = installed_path(models_dir, spec);
        let metadata = std::fs::metadata(&path).map_err(|_| EngineError::ModelMissing(spec.id))?;
        let fingerprint = (metadata.len(), metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH));
        let mut cache = self
            .verified
            .lock()
            .map_err(|_| EngineError::ModelVerification("cache_poisoned".to_owned()))?;
        if cache.get(&path) == Some(&fingerprint) {
            return Ok(path);
        }
        verify(&path, spec).map_err(|error| EngineError::ModelVerification(error.to_string()))?;
        cache.insert(path.clone(), fingerprint);
        Ok(path)
    }
}

fn worker_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .clamp(1, 4)
}

fn spec_for(executable: &Path, arguments: Vec<OsString>, startup: Duration) -> Result<WorkerSpec, EngineError> {
    if !executable.exists() {
        return Err(EngineError::WorkerMissing(executable.to_path_buf()));
    }
    let executable = executable
        .canonicalize()
        .map_err(|_| EngineError::WorkerMissing(executable.to_path_buf()))?;
    let working_directory = executable
        .parent()
        .map_or_else(std::env::temp_dir, Path::to_path_buf);
    Ok(WorkerSpec {
        executable,
        arguments,
        working_directory,
        maximum_control_bytes: MAXIMUM_CONTROL_BYTES,
        maximum_audio_bytes: MAXIMUM_AUDIO_BYTES,
        startup_timeout: startup,
        environment: Vec::new(),
    })
}

/// The whisper.cpp recognition worker.
pub struct AsrEngine {
    worker: PersistentWorker,
    pub model_id: &'static str,
    pub model_revision: &'static str,
}

impl AsrEngine {
    /// Loads user-imported GGML weights without assuming a base-model DTW
    /// architecture. Custom weights are verified by the caller.
    pub fn custom(paths: &WorkerPaths, model: &Path) -> Result<Self, EngineError> {
        let arguments = vec!["--model".into(), model.as_os_str().to_owned(), "--threads".into(), worker_threads().to_string().into()];
        Ok(Self {
            worker: PersistentWorker::new(spec_for(&paths.asr, arguments, Duration::from_secs(180))?),
            model_id: dictation_models::custom::ID,
            model_revision: "user-imported",
        })
    }

    /// # Errors
    ///
    /// Fails when the worker binary or verified model is unavailable.
    pub fn new(
        paths: &WorkerPaths,
        models_dir: &Path,
        spec: &'static ModelSpec,
        verifier: &ModelVerifier,
        model_override: Option<&Path>,
    ) -> Result<Self, EngineError> {
        let model = match model_override {
            Some(path) => path.to_path_buf(),
            None => verifier.verified_path(models_dir, spec)?,
        };
        let mut arguments: Vec<OsString> = vec![
            "--model".into(),
            model.into_os_string(),
            "--threads".into(),
            worker_threads().to_string().into(),
        ];
        if let Some(preset) = spec.dtw_preset {
            arguments.extend(["--dtw".into(), preset.into()]);
        }
        Ok(Self {
            worker: PersistentWorker::new(spec_for(&paths.asr, arguments, Duration::from_secs(60))?),
            model_id: spec.id,
            model_revision: spec.revision,
        })
    }

    /// Loads the model now so the first dictation does not pay for it.
    ///
    /// # Errors
    ///
    /// Fails when the worker cannot start.
    pub fn warm(&mut self) -> Result<Ready, EngineError> {
        Ok(self.worker.ensure_started()?.clone())
    }

    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.worker.is_running()
    }

    /// Frees model memory; the next request reloads it.
    pub fn release(&mut self) {
        self.worker.kill();
    }

    /// Transcribes canonical PCM. Sample positions are relative to `pcm`.
    ///
    /// # Errors
    ///
    /// Fails on worker failure or timeout; the worker is restarted next time.
    pub fn transcribe(
        &mut self,
        pcm: &[i16],
        final_pass: bool,
        language: Option<&str>,
        vocabulary: Option<&str>,
        timeout: Duration,
    ) -> Result<(String, Vec<RecognizedSegment>), EngineError> {
        self.transcribe_session(pcm, final_pass, language, vocabulary, timeout, None)
    }

    pub fn transcribe_session(
        &mut self,
        pcm: &[i16],
        final_pass: bool,
        language: Option<&str>,
        vocabulary: Option<&str>,
        timeout: Duration,
        stream_epoch: Option<u64>,
    ) -> Result<(String, Vec<RecognizedSegment>), EngineError> {
        let bytes: Vec<u8> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        let request = Request::Transcribe(TranscribeRequest {
            request_id: self.worker.next_request_id(),
            stream_epoch,
            language: language.map(str::to_owned),
            initial_prompt: vocabulary.filter(|text| !text.is_empty()).map(str::to_owned),
            final_pass,
        });
        match self.worker.request(&request, Some(&bytes), timeout)? {
            Response::Transcript(transcript) => Ok((
                transcript.language,
                transcript
                    .segments
                    .into_iter()
                    .map(|segment| RecognizedSegment {
                        text: segment.text,
                        start_sample: segment.start_sample,
                        end_sample: segment.end_sample,
                        no_speech_probability: segment.no_speech_probability,
                        words: segment
                            .words
                            .into_iter()
                            .map(|word| RecognizedWord {
                                text: word.text,
                                start_sample: word.start_sample,
                                end_sample: word.end_sample,
                                probability: word.probability,
                                timing_unreliable: word.timing_unreliable,
                            })
                            .collect(),
                    })
                    .collect(),
            )),
            Response::Error(error) => Err(EngineError::WorkerRefused(error.code)),
            _ => Err(EngineError::UnexpectedResponse),
        }
    }
}

/// Shared flag set while dictation is active; privacy work yields to it.
#[derive(Debug, Clone, Default)]
pub struct Preemption(Arc<AtomicBool>);

impl Preemption {
    pub fn set(&self, active: bool) {
        self.0.store(active, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// The llama.cpp privacy worker, used only for training copies.
pub struct PrivacyEngine {
    worker: PersistentWorker,
    pub model_id: &'static str,
    pub model_revision: &'static str,
    timeout: Duration,
    preemption: Preemption,
    preempted: bool,
}

impl PrivacyEngine {
    /// # Errors
    ///
    /// Fails when the worker binary or verified model is unavailable.
    pub fn new(
        paths: &WorkerPaths,
        models_dir: &Path,
        spec: &'static ModelSpec,
        verifier: &ModelVerifier,
        preemption: Preemption,
    ) -> Result<Self, EngineError> {
        let model = verifier.verified_path(models_dir, spec)?;
        let arguments: Vec<OsString> = vec![
            "--model".into(),
            model.into_os_string(),
            "--threads".into(),
            worker_threads().to_string().into(),
            "--context".into(),
            "2048".into(),
        ];
        Ok(Self {
            worker: PersistentWorker::new(spec_for(&paths.privacy, arguments, Duration::from_secs(300))?),
            model_id: spec.id,
            model_revision: spec.revision,
            timeout: Duration::from_secs(180),
            preemption,
            preempted: false,
        })
    }

    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.worker.is_running()
    }

    pub fn release(&mut self) {
        self.worker.kill();
    }

    /// True if the last analysis stopped because dictation started.
    #[must_use]
    pub const fn was_preempted(&self) -> bool {
        self.preempted
    }

    pub const fn clear_preempted(&mut self) {
        self.preempted = false;
    }

    /// # Errors
    ///
    /// Fails when the worker cannot start.
    pub fn warm(&mut self) -> Result<Ready, EngineError> {
        Ok(self.worker.ensure_started()?.clone())
    }
}

impl Classifier for PrivacyEngine {
    fn model_id(&self) -> String {
        self.model_id.to_owned()
    }

    fn classify(
        &mut self,
        system_prompt: &str,
        user_prompt: &str,
        grammar: &str,
        max_tokens: u32,
    ) -> Result<String, ClassifierFailure> {
        if self.preemption.is_set() {
            self.preempted = true;
            return Err(ClassifierFailure::Unavailable);
        }
        let request = Request::Classify(ClassifyRequest {
            request_id: self.worker.next_request_id(),
            system_prompt: system_prompt.to_owned(),
            user_prompt: user_prompt.to_owned(),
            grammar: grammar.to_owned(),
            max_tokens,
        });
        match self.worker.request(&request, None, self.timeout) {
            Ok(Response::Classification(output)) if output.truncated => {
                Err(ClassifierFailure::Truncated)
            }
            Ok(Response::Classification(output)) => Ok(output.output),
            Ok(Response::Error(error)) if error.code == "context_exceeded" => {
                Err(ClassifierFailure::ContextExceeded)
            }
            Err(ProtocolError::Timeout) => Err(ClassifierFailure::Timeout),
            Err(ProtocolError::WorkerSpawn(_)) => Err(ClassifierFailure::Unavailable),
            Ok(_) | Err(_) => Err(ClassifierFailure::WorkerError),
        }
    }
}

/// Total physical memory in MiB, when the platform exposes it cheaply.
#[must_use]
pub fn total_memory_mib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib / 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// On machines below this size the ASR worker is released before the privacy
/// model loads (plan §4). Unknown memory is treated as low.
pub const LOW_MEMORY_THRESHOLD_MIB: u64 = 12 * 1024;

#[must_use]
pub fn is_low_memory() -> bool {
    total_memory_mib().is_none_or(|total| total < LOW_MEMORY_THRESHOLD_MIB)
}

#[cfg(test)]
mod custom_model_tests {
    use super::*;

    /// Optional real-runtime check, using explicitly supplied local test weights.
    #[test]
    fn imported_model_loads_in_the_native_worker() {
        let (Ok(source), Ok(worker)) = (std::env::var("TEST_WHISPER_MODEL"), std::env::var("TEST_ASR_WORKER")) else { return };
        let directory = std::env::temp_dir().join(format!("custom-asr-smoke-{}", std::process::id()));
        let model = dictation_models::custom::install(&source, "", &directory, &AtomicBool::new(false), |_, _| {}).unwrap();
        let path = model.verified_path(&directory).unwrap();
        let paths = WorkerPaths { asr: worker.into(), privacy: PathBuf::new() };
        let mut engine = AsrEngine::custom(&paths, &path).unwrap();
        let ready = engine.warm().unwrap();
        assert!(ready.model_description.contains("dtw=false"));
        let result = engine.transcribe(&vec![0; 16_000], true, Some("en"), None, Duration::from_secs(30));
        assert!(result.is_ok(), "{result:?}");
        engine.release();
        std::fs::remove_dir_all(directory).unwrap();
    }
}
