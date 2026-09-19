//! Customer-isolated personalization jobs (plan §10).
//!
//! One job trains one tenant's model from that tenant's samples only. The
//! dataset is snapshotted with consent rechecked, split by time so evaluation
//! never scores the sitting it trained on, and handed to an external trainer
//! in a private job directory that is deleted afterwards. The server then
//! measures the candidate itself with the desktop ASR runtime on held-out and
//! general regression audio; that run also proves the export loads. Promotion
//! requires thresholds fixed before training. Otherwise the base model stays
//! and the outcome is reported honestly.

use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use dictation_core::{
    package::{PackageLimits, decode_archive},
    wer::corpus_wer,
};
use dictation_models::delivery::{DeliveryManifest, Evaluation, SUPPORTED_FORMAT, SignedManifest};
use dictation_worker::{
    messages::{Request, Response, TranscribeRequest},
    persistent::{PersistentWorker, WorkerSpec},
    serve::{MAXIMUM_AUDIO_BYTES, MAXIMUM_CONTROL_BYTES},
};
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::store::{ServerError, ServerStore, random_hex, write_private};

/// Thresholds fixed before a run. Provisional engineering targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotionPolicy {
    pub min_training_examples: usize,
    pub min_heldout_examples: usize,
    pub heldout_fraction: f64,
    /// Absolute held-out WER reduction required.
    pub improvement_threshold: f64,
    /// Maximum absolute WER increase tolerated on the regression set.
    pub regression_ceiling: f64,
    pub device_budget_mib: u32,
}

impl Default for PromotionPolicy {
    fn default() -> Self {
        Self {
            min_training_examples: 200,
            min_heldout_examples: 30,
            heldout_fraction: 0.2,
            improvement_threshold: 0.01,
            regression_ceiling: 0.005,
            device_budget_mib: 1_500,
        }
    }
}

pub struct TrainerConfig {
    /// External trainer executable and fixed arguments. It receives
    /// `--job-dir DIR --base-model PATH --output DIR` and must write
    /// `OUTPUT/model.bin` in the whisper.cpp ggml format.
    pub program: PathBuf,
    pub arguments: Vec<String>,
    pub base_model: PathBuf,
    pub base_model_id: String,
    /// Directory of `name.wav` + `name.txt` pairs of general speech.
    pub regression_set: Option<PathBuf>,
    pub asr_worker: PathBuf,
    pub signing_key: SigningKey,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingOutcome {
    Delivered { model_version: String, evaluation: Evaluation },
    KeptBaseModel { reason: String, evaluation: Option<Evaluation> },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
struct Clip {
    wav: Vec<u8>,
    text: String,
}

fn clips_of(archive: &[u8]) -> Result<Vec<Clip>, &'static str> {
    let (manifest, payloads) = decode_archive(archive, PackageLimits::default()).map_err(|error| error.code)?;
    let clips = manifest["clips"].as_array().ok_or("invalid_clips")?;
    clips
        .iter()
        .map(|clip| {
            let audio = payloads.get(clip["audio"].as_str().unwrap_or_default()).ok_or("missing_audio")?;
            let text = payloads.get(clip["text"].as_str().unwrap_or_default()).ok_or("missing_text")?;
            Ok(Clip {
                wav: audio.clone(),
                text: String::from_utf8(text.clone()).map_err(|_| "text_not_utf8")?,
            })
        })
        .collect()
}

fn pcm_of(wav: &[u8]) -> Option<Vec<u8>> {
    // Canonical 44-byte header validated at admission.
    wav.get(44..).map(<[u8]>::to_vec)
}

fn write_split(directory: &Path, clips: &[(String, Clip)]) -> std::io::Result<()> {
    fs::create_dir_all(directory)?;
    for (name, clip) in clips {
        write_private(&directory.join(format!("{name}.wav")), &clip.wav)?;
        write_private(&directory.join(format!("{name}.txt")), clip.text.as_bytes())?;
    }
    Ok(())
}

fn read_pairs(directory: &Path) -> std::io::Result<Vec<Clip>> {
    let mut clips = Vec::new();
    let mut names: Vec<PathBuf> = fs::read_dir(directory)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "wav"))
        .collect();
    names.sort();
    for wav in names {
        let text = fs::read_to_string(wav.with_extension("txt"))?;
        clips.push(Clip { wav: fs::read(&wav)?, text });
    }
    Ok(clips)
}

/// Transcribes clips with the desktop runtime and returns corpus WER.
fn measure(worker: &Path, model: &Path, clips: &[Clip], dtw: Option<&str>) -> Result<f64, String> {
    let mut arguments: Vec<OsString> = vec!["--model".into(), model.as_os_str().to_owned(), "--threads".into(), "4".into()];
    if let Some(preset) = dtw {
        arguments.extend(["--dtw".into(), preset.into()]);
    }
    let executable = worker.canonicalize().map_err(|_| "asr_worker_missing".to_owned())?;
    let mut runner = PersistentWorker::new(WorkerSpec {
        working_directory: executable.parent().map_or_else(std::env::temp_dir, Path::to_path_buf),
        executable,
        arguments,
        maximum_control_bytes: MAXIMUM_CONTROL_BYTES,
        maximum_audio_bytes: MAXIMUM_AUDIO_BYTES,
        startup_timeout: Duration::from_secs(120),
        environment: Vec::new(),
    });
    runner.ensure_started().map_err(|_| "model_failed_to_load".to_owned())?;
    let mut pairs = Vec::with_capacity(clips.len());
    for clip in clips {
        let pcm = pcm_of(&clip.wav).ok_or("invalid_wav")?;
        let request = Request::Transcribe(TranscribeRequest {
            request_id: runner.next_request_id(),
            language: Some("en".to_owned()),
            initial_prompt: None,
            final_pass: true,
        });
        let hypothesis = match runner.request(&request, Some(&pcm), Duration::from_secs(300)) {
            Ok(Response::Transcript(transcript)) => transcript
                .segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            _ => return Err("evaluation_transcription_failed".to_owned()),
        };
        pairs.push((clip.text.clone(), hypothesis));
    }
    corpus_wer(pairs.iter().map(|(reference, hypothesis)| (reference.as_str(), hypothesis.as_str())))
        .ok_or_else(|| "empty_evaluation_set".to_owned())
}

fn peak_rss_mib(pid: u32) -> Option<u32> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    u32::try_from(kib / 1024).ok()
}

/// Loads a candidate in the desktop runtime and reports its peak memory.
fn measure_memory(worker: &Path, model: &Path, clip: &Clip) -> Option<u32> {
    let executable = worker.canonicalize().ok()?;
    let mut runner = PersistentWorker::new(WorkerSpec {
        working_directory: executable.parent().map_or_else(std::env::temp_dir, Path::to_path_buf),
        executable,
        arguments: vec!["--model".into(), model.as_os_str().to_owned()],
        maximum_control_bytes: MAXIMUM_CONTROL_BYTES,
        maximum_audio_bytes: MAXIMUM_AUDIO_BYTES,
        startup_timeout: Duration::from_secs(120),
        environment: Vec::new(),
    });
    runner.ensure_started().ok()?;
    let request = Request::Transcribe(TranscribeRequest {
        request_id: runner.next_request_id(),
        language: Some("en".to_owned()),
        initial_prompt: None,
        final_pass: true,
    });
    runner.request(&request, Some(&pcm_of(&clip.wav)?), Duration::from_secs(300)).ok()?;
    peak_rss_mib(runner.process_id()?)
}

fn job_cancelled(store: &ServerStore, job_id: &str) -> Result<bool, ServerError> {
    Ok(store
        .training_job_state(job_id)?
        .is_some_and(|(state, _)| state == "cancelled"))
}

/// Runs one personalization job for `tenant_id`.
///
/// # Errors
///
/// Fails on storage errors; trainer and evaluation failures keep the base model.
#[allow(clippy::too_many_lines)]
pub fn train_tenant(
    store: &mut ServerStore,
    tenant_id: &str,
    config: &TrainerConfig,
    policy: PromotionPolicy,
    compute_cap: Duration,
    now: impl Fn() -> i64,
) -> Result<TrainingOutcome, ServerError> {
    let job_id = format!("train-{}", random_hex(10));
    store.insert_training_job(&job_id, tenant_id, now())?;
    let keep = |store: &ServerStore, reason: &str, evaluation: Option<Evaluation>| -> Result<TrainingOutcome, ServerError> {
        store.finish_training_job(&job_id, "kept_base_model", reason, now())?;
        Ok(TrainingOutcome::KeptBaseModel { reason: reason.to_owned(), evaluation })
    };
    // Consent is rechecked here, immediately before training.
    let samples = store.trainable_samples(tenant_id, now())?;
    if samples.len() < policy.min_training_examples + policy.min_heldout_examples {
        return keep(store, "insufficient_data", None);
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let heldout_count = ((samples.len() as f64 * policy.heldout_fraction).ceil() as usize).max(policy.min_heldout_examples);
    let split = samples.len() - heldout_count;
    let dataset_id = format!("dataset-{}", random_hex(10));
    let sample_ids: Vec<String> = samples.iter().map(|sample| sample.sample_id.clone()).collect();
    store.add_node(&dataset_id, "dataset", tenant_id, &sample_ids, &json!({ "train": split, "heldout": heldout_count }), now())?;
    store.add_node(&job_id, "job", tenant_id, &[dataset_id], &json!({}), now())?;

    let job_dir = store.root().join("jobs").join(&job_id);
    let result = (|| -> Result<TrainingOutcome, ServerError> {
        let mut train = Vec::new();
        let mut heldout = Vec::new();
        for (index, sample) in samples.iter().enumerate() {
            let archive = store.read_sample(tenant_id, &sample.sample_id)?;
            let Ok(clips) = clips_of(&archive) else {
                return keep(store, "stored_sample_unreadable", None);
            };
            for (clip_index, clip) in clips.into_iter().enumerate() {
                let name = format!("{index:06}-{clip_index:03}");
                if index < split { train.push((name, clip)) } else { heldout.push((name, clip)) }
            }
        }
        write_split(&job_dir.join("train"), &train)?;
        write_split(&job_dir.join("heldout"), &heldout)?;
        fs::write(
            job_dir.join("job.json"),
            serde_json::to_vec(&json!({
                "base_model_id": config.base_model_id,
                "train_examples": train.len(),
                "heldout_examples": heldout.len(),
            }))
            .unwrap_or_default(),
        )?;
        let output = job_dir.join("output");
        fs::create_dir_all(&output)?;
        let mut child = Command::new(&config.program)
            .args(&config.arguments)
            .arg("--job-dir").arg(&job_dir)
            .arg("--base-model").arg(&config.base_model)
            .arg("--output").arg(&output)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Some(status);
            }
            if started.elapsed() > compute_cap || job_cancelled(store, &job_id)? {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(500));
        };
        if job_cancelled(store, &job_id)? {
            return Ok(TrainingOutcome::Cancelled);
        }
        let Some(status) = status else {
            return keep(store, "compute_cap_exceeded", None);
        };
        if !status.success() {
            return keep(store, "trainer_failed", None);
        }
        let candidate = output.join("model.bin");
        if !candidate.exists() {
            return keep(store, "trainer_produced_no_model", None);
        }
        let heldout_clips: Vec<Clip> = heldout.into_iter().map(|(_, clip)| clip).collect();
        let Some(regression_dir) = &config.regression_set else {
            return keep(store, "regression_set_missing", None);
        };
        let regression = read_pairs(regression_dir)?;
        let measured = (|| -> Result<Evaluation, String> {
            Ok(Evaluation {
                heldout_wer_base: measure(&config.asr_worker, &config.base_model, &heldout_clips, None)?,
                heldout_wer_candidate: measure(&config.asr_worker, &candidate, &heldout_clips, None)?,
                regression_wer_base: measure(&config.asr_worker, &config.base_model, &regression, None)?,
                regression_wer_candidate: measure(&config.asr_worker, &candidate, &regression, None)?,
                improvement_threshold: policy.improvement_threshold,
                regression_ceiling: policy.regression_ceiling,
            })
        })();
        let evaluation = match measured {
            Ok(evaluation) => evaluation,
            Err(reason) => return keep(store, &reason, None),
        };
        if !evaluation.accepted() {
            return keep(store, "no_measured_improvement", Some(evaluation));
        }
        let peak = heldout_clips
            .first()
            .and_then(|clip| measure_memory(&config.asr_worker, &candidate, clip))
            .unwrap_or(u32::MAX);
        if peak > policy.device_budget_mib {
            return keep(store, "exceeds_device_budget", Some(evaluation));
        }
        // Consent may have been withdrawn while training ran.
        if job_cancelled(store, &job_id)?
            || store.trainable_samples(tenant_id, now())?.len() < samples.len()
        {
            store.finish_training_job(&job_id, "cancelled", "consent_changed_during_training", now())?;
            return Ok(TrainingOutcome::Cancelled);
        }
        let artifact = fs::read(&candidate)?;
        let model_version = format!("cm-{}", random_hex(8));
        let manifest = DeliveryManifest {
            model_version: model_version.clone(),
            base_model_id: config.base_model_id.clone(),
            format: SUPPORTED_FORMAT.to_owned(),
            artifact_sha256: hex::encode(Sha256::digest(&artifact)),
            artifact_size: artifact.len() as u64,
            peak_memory_mib: peak,
            created_at: now(),
            evaluation: evaluation.clone(),
        };
        let manifest_json = serde_json::to_string(&manifest).map_err(|_| ServerError::Conflict("manifest_encode"))?;
        let signed = SignedManifest {
            signature_hex: hex::encode(config.signing_key.sign(manifest_json.as_bytes()).to_bytes()),
            manifest_json,
        };
        let build_id = format!("build-{}", random_hex(8));
        store.add_node(&build_id, "build", tenant_id, std::slice::from_ref(&job_id), &json!({ "sha256": manifest.artifact_sha256 }), now())?;
        store.add_node(&model_version, "delivery", tenant_id, &[build_id], &json!({}), now())?;
        store.store_delivery(tenant_id, &model_version, &signed, &artifact, now())?;
        store.finish_training_job(&job_id, "delivered", "", now())?;
        Ok(TrainingOutcome::Delivered { model_version, evaluation })
    })();
    // Decrypted training material never outlives the job.
    let _ = fs::remove_dir_all(&job_dir);
    result
}

/// Loads an Ed25519 signing key, creating it (owner-only) when absent.
///
/// # Errors
///
/// Fails when the key file cannot be read or written.
pub fn load_or_create_signing_key(path: &Path) -> std::io::Result<SigningKey> {
    if path.exists() {
        let bytes = fs::read(path)?;
        let key: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| std::io::Error::other("signing key must be 32 bytes"))?;
        return Ok(SigningKey::from_bytes(&key));
    }
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    write_private(path, &key.to_bytes())?;
    Ok(key)
}

/// Content-free summary for operators.
#[must_use]
pub fn describe(outcome: &TrainingOutcome) -> Value {
    match outcome {
        TrainingOutcome::Delivered { model_version, evaluation } => json!({ "outcome": "delivered", "model_version": model_version, "evaluation": evaluation }),
        TrainingOutcome::KeptBaseModel { reason, evaluation } => json!({ "outcome": "kept_base_model", "reason": reason, "evaluation": evaluation }),
        TrainingOutcome::Cancelled => json!({ "outcome": "cancelled" }),
    }
}
