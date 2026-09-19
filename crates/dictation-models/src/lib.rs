//! Pinned model registry and verified installation (plan §3, §4).
//!
//! Exactly two learned roles exist. Every artifact is pinned by URL revision,
//! SHA-256, and size, and carries its license and conversion provenance.
//! Weight files are inert data read by the worker runtimes; nothing downloaded
//! is ever executed. Downloads are explicit, visible network operations;
//! dictation works offline once models are installed.

pub mod delivery;

use std::{
    fmt,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Asr,
    Privacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub role: Role,
    pub display_name: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub revision: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    pub quantization: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    /// Who produced the quantized artifact, when not the original publisher.
    pub conversion_provenance: &'static str,
    /// whisper.cpp DTW alignment preset (ASR only).
    pub dtw_preset: Option<&'static str>,
    /// Measured on the 8 GB reference machine; see docs/BENCHMARKS.md.
    pub measured_peak_rss_mib: u32,
    pub notes: &'static str,
}

const WHISPER_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
const QWEN_GGUF_REVISION: &str = "ae44f08e1392f39c0e474af10c3ff8355c8b6688";

/// The pinned candidates. The defaults are chosen from measurements in
/// docs/BENCHMARKS.md; the others remain available for evaluation.
pub const MODELS: [ModelSpec; 3] = [
    ModelSpec {
        id: "whisper-base-q5_1",
        role: Role::Asr,
        display_name: "Whisper base (multilingual, q5_1)",
        filename: "ggml-base-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base-q5_1.bin",
        revision: WHISPER_REVISION,
        sha256: "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
        size_bytes: 59_707_625,
        quantization: "q5_1",
        license: "MIT",
        license_url: "https://github.com/openai/whisper/blob/main/LICENSE",
        conversion_provenance: "ggml conversion published by the whisper.cpp maintainers",
        dtw_preset: Some("base"),
        measured_peak_rss_mib: 290,
        notes: "Default recognition model on the 8 GB baseline.",
    },
    ModelSpec {
        id: "whisper-small-q5_1",
        role: Role::Asr,
        display_name: "Whisper small (multilingual, q5_1)",
        filename: "ggml-small-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q5_1.bin",
        revision: WHISPER_REVISION,
        sha256: "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
        size_bytes: 190_085_487,
        quantization: "q5_1",
        license: "MIT",
        license_url: "https://github.com/openai/whisper/blob/main/LICENSE",
        conversion_provenance: "ggml conversion published by the whisper.cpp maintainers",
        dtw_preset: Some("small"),
        measured_peak_rss_mib: 0,
        notes: "Higher accuracy candidate; see benchmark for latency on the baseline.",
    },
    ModelSpec {
        id: "qwen3-4b-instruct-2507-q4_k_m",
        role: Role::Privacy,
        display_name: "Qwen3-4B-Instruct-2507 (Q4_K_M)",
        filename: "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/ae44f08e1392f39c0e474af10c3ff8355c8b6688/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        revision: QWEN_GGUF_REVISION,
        sha256: "2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e",
        size_bytes: 2_497_280_736,
        quantization: "Q4_K_M (imatrix)",
        license: "Apache-2.0",
        license_url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE",
        conversion_provenance: "Third-party GGUF quantization by bartowski of Qwen/Qwen3-4B-Instruct-2507 at cdbee75f17c01a7cc42f958dc650907174af0554; conversion not independently reproduced (docs/MODELS.md).",
        dtw_preset: None,
        measured_peak_rss_mib: 2_759,
        notes: "Privacy classifier candidate. Adequate recall is unmeasured; automatic upload stays gated.",
    },
];

#[must_use]
pub fn spec(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|model| model.id == id)
}

#[must_use]
pub fn default_for(role: Role) -> &'static ModelSpec {
    match role {
        Role::Asr => &MODELS[0],
        Role::Privacy => &MODELS[2],
    }
}

#[derive(Debug)]
pub enum InstallError {
    Io(io::Error),
    Network(String),
    SizeMismatch { expected: u64, actual: u64 },
    ChecksumMismatch,
    Cancelled,
    NotHttps,
}

impl fmt::Display for InstallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "model file error: {error}"),
            Self::Network(error) => write!(formatter, "model download failed: {error}"),
            Self::SizeMismatch { expected, actual } => {
                write!(formatter, "model size {actual} does not match pinned {expected}")
            }
            Self::ChecksumMismatch => write!(formatter, "model checksum does not match the pin"),
            Self::Cancelled => write!(formatter, "model download was cancelled"),
            Self::NotHttps => write!(formatter, "model source must use HTTPS"),
        }
    }
}

impl std::error::Error for InstallError {}

impl From<io::Error> for InstallError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstallState {
    Missing,
    Installed,
    /// Present but failing verification; it will not be loaded.
    Corrupt,
}

#[must_use]
pub fn installed_path(models_dir: &Path, spec: &ModelSpec) -> PathBuf {
    models_dir.join(spec.filename)
}

fn sha256_file(path: &Path) -> io::Result<(u64, String)> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((total, hex::encode(digest.finalize())))
}

/// Verifies an installed file against its pin before a worker may load it.
///
/// # Errors
///
/// Returns a mismatch error, or an I/O error for an unreadable file.
pub fn verify(path: &Path, spec: &ModelSpec) -> Result<(), InstallError> {
    let size = fs::metadata(path)?.len();
    if size != spec.size_bytes {
        return Err(InstallError::SizeMismatch {
            expected: spec.size_bytes,
            actual: size,
        });
    }
    let (_, digest) = sha256_file(path)?;
    if digest == spec.sha256 {
        Ok(())
    } else {
        Err(InstallError::ChecksumMismatch)
    }
}

/// Cheap state check (size only); full verification happens before loading.
#[must_use]
pub fn state(models_dir: &Path, spec: &ModelSpec) -> InstallState {
    match fs::metadata(installed_path(models_dir, spec)) {
        Ok(metadata) if metadata.len() == spec.size_bytes => InstallState::Installed,
        Ok(_) => InstallState::Corrupt,
        Err(_) => InstallState::Missing,
    }
}

/// Copies a model the user obtained out of band, verifying it first.
///
/// # Errors
///
/// Fails when the source does not match the pin or cannot be copied.
pub fn import_local(source: &Path, models_dir: &Path, spec: &ModelSpec) -> Result<PathBuf, InstallError> {
    verify(source, spec)?;
    let target = installed_path(models_dir, spec);
    let partial = target.with_extension("part");
    fs::copy(source, &partial)?;
    finish(&partial, &target, spec)
}

fn finish(partial: &Path, target: &Path, spec: &ModelSpec) -> Result<PathBuf, InstallError> {
    match verify(partial, spec) {
        Ok(()) => {
            fs::rename(partial, target)?;
            Ok(target.to_path_buf())
        }
        Err(error) => {
            let _ = fs::remove_file(partial);
            Err(error)
        }
    }
}

/// Downloads a pinned model over HTTPS, verifying size and checksum before an
/// atomic rename. An interrupted download leaves only a `.part` file, which
/// the next attempt replaces.
///
/// # Errors
///
/// Fails on network errors, cancellation, or verification failure.
pub fn download(
    spec: &ModelSpec,
    models_dir: &Path,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf, InstallError> {
    if !spec.url.starts_with("https://") {
        return Err(InstallError::NotHttps);
    }
    fs::create_dir_all(models_dir)?;
    let target = installed_path(models_dir, spec);
    let partial = target.with_extension("part");
    let client = reqwest::blocking::Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(30))
        .timeout(None)
        .user_agent(concat!("local-dictation/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| InstallError::Network(error.to_string()))?;
    let mut response = client
        .get(spec.url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| InstallError::Network(error.to_string()))?;
    let mut file = File::create(&partial)?;
    let mut buffer = vec![0_u8; 1 << 20];
    let mut received = 0_u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = fs::remove_file(&partial);
            return Err(InstallError::Cancelled);
        }
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        received += read as u64;
        if received > spec.size_bytes {
            drop(file);
            let _ = fs::remove_file(&partial);
            return Err(InstallError::SizeMismatch {
                expected: spec.size_bytes,
                actual: received,
            });
        }
        file.write_all(&buffer[..read])?;
        progress(received, spec.size_bytes);
    }
    file.sync_all()?;
    drop(file);
    finish(&partial, &target, spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_two_roles_are_pinned_with_complete_provenance() {
        for model in MODELS {
            assert!(model.url.starts_with("https://"));
            assert!(model.url.contains(model.revision));
            assert_eq!(model.sha256.len(), 64);
            assert!(model.size_bytes > 0);
            assert!(!model.license.is_empty());
            assert!(!model.conversion_provenance.is_empty());
            assert!(model.filename.ends_with(".bin") || model.filename.ends_with(".gguf"));
        }
        assert_eq!(default_for(Role::Asr).role, Role::Asr);
        assert_eq!(default_for(Role::Privacy).role, Role::Privacy);
    }

    #[test]
    fn verification_rejects_a_tampered_file() {
        let directory = std::env::temp_dir().join(format!("ld-models-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let payload = b"inert weights".to_vec();
        let digest = hex::encode(Sha256::digest(&payload));
        let digest: &'static str = Box::leak(digest.into_boxed_str());
        let spec = ModelSpec {
            sha256: digest,
            size_bytes: payload.len() as u64,
            filename: "test.bin",
            ..MODELS[0]
        };
        let source = directory.join("source.bin");
        fs::write(&source, &payload).unwrap();
        let installed = import_local(&source, &directory, &spec).unwrap();
        assert!(verify(&installed, &spec).is_ok());
        assert_eq!(state(&directory, &spec), InstallState::Installed);
        fs::write(&installed, b"inert weightz").unwrap();
        assert!(matches!(verify(&installed, &spec), Err(InstallError::ChecksumMismatch)));
        fs::remove_dir_all(directory).unwrap();
    }
}
