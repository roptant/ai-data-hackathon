//! Signed delivery of personalized recognition models (plan §10).
//!
//! The server signs the exact manifest bytes with Ed25519. The client verifies
//! the signature against a pinned public key, the artifact's size and SHA-256,
//! compatibility with the installed base model, and the device memory budget
//! before an atomic install. The previous model stays available for rollback,
//! and deleting personalization returns to the unchanged base model.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliveryManifest {
    pub model_version: String,
    /// Base model the adaptation was trained from; must be installed.
    pub base_model_id: String,
    pub format: String,
    pub artifact_sha256: String,
    pub artifact_size: u64,
    pub peak_memory_mib: u32,
    pub created_at: i64,
    pub evaluation: Evaluation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub heldout_wer_base: f64,
    pub heldout_wer_candidate: f64,
    pub regression_wer_base: f64,
    pub regression_wer_candidate: f64,
    pub improvement_threshold: f64,
    pub regression_ceiling: f64,
}

impl Evaluation {
    /// The acceptance rule defined before training (plan §10).
    #[must_use]
    pub fn accepted(&self) -> bool {
        self.heldout_wer_base - self.heldout_wer_candidate >= self.improvement_threshold
            && self.regression_wer_candidate - self.regression_wer_base <= self.regression_ceiling
    }
}

/// Wire form: the manifest travels as the exact signed JSON string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedManifest {
    pub manifest_json: String,
    pub signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryError {
    BadSignature,
    MalformedManifest,
    IncompatibleBase,
    UnsupportedFormat,
    ExceedsDeviceBudget,
    EvaluationNotAccepted,
    ArtifactMismatch,
    Io(String),
}

impl std::fmt::Display for DeliveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::BadSignature => "model manifest signature is invalid",
            Self::MalformedManifest => "model manifest is malformed",
            Self::IncompatibleBase => "model was trained from a different base model",
            Self::UnsupportedFormat => "model format is not loadable by this runtime",
            Self::ExceedsDeviceBudget => "model exceeds this device's memory budget",
            Self::EvaluationNotAccepted => "model did not meet its evaluation thresholds",
            Self::ArtifactMismatch => "model artifact does not match its manifest",
            Self::Io(message) => return write!(formatter, "model install failed: {message}"),
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for DeliveryError {}

impl From<io::Error> for DeliveryError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

pub const SUPPORTED_FORMAT: &str = "whisper.cpp-ggml";

/// Verifies the signature and every acceptance condition.
///
/// # Errors
///
/// Returns the first failed condition.
pub fn verify_manifest(
    signed: &SignedManifest,
    key: &VerifyingKey,
    installed_base_id: &str,
    device_budget_mib: u32,
) -> Result<DeliveryManifest, DeliveryError> {
    let signature_bytes: [u8; 64] = hex::decode(&signed.signature_hex)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or(DeliveryError::BadSignature)?;
    key.verify(
        signed.manifest_json.as_bytes(),
        &Signature::from_bytes(&signature_bytes),
    )
    .map_err(|_| DeliveryError::BadSignature)?;
    let manifest: DeliveryManifest =
        serde_json::from_str(&signed.manifest_json).map_err(|_| DeliveryError::MalformedManifest)?;
    if manifest.base_model_id != installed_base_id {
        return Err(DeliveryError::IncompatibleBase);
    }
    if manifest.format != SUPPORTED_FORMAT {
        return Err(DeliveryError::UnsupportedFormat);
    }
    if manifest.peak_memory_mib > device_budget_mib {
        return Err(DeliveryError::ExceedsDeviceBudget);
    }
    if !manifest.evaluation.accepted() {
        return Err(DeliveryError::EvaluationNotAccepted);
    }
    if manifest.artifact_sha256.len() != 64
        || !manifest
            .model_version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        || manifest.model_version.is_empty()
        || manifest.model_version.len() > 64
    {
        return Err(DeliveryError::MalformedManifest);
    }
    Ok(manifest)
}

/// Personalized models live under `models/personal/` with an atomically
/// replaced `active.json` pointer.
#[derive(Debug, Clone)]
pub struct PersonalModels {
    directory: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Pointer {
    active: Option<String>,
    previous: Option<String>,
}

impl PersonalModels {
    #[must_use]
    pub fn new(models_dir: &Path) -> Self {
        Self {
            directory: models_dir.join("personal"),
        }
    }

    fn pointer_path(&self) -> PathBuf {
        self.directory.join("active.json")
    }

    fn artifact_path(&self, version: &str) -> PathBuf {
        self.directory.join(format!("{version}.bin"))
    }

    fn read_pointer(&self) -> Pointer {
        fs::read(self.pointer_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(Pointer {
                active: None,
                previous: None,
            })
    }

    fn write_pointer(&self, pointer: &Pointer) -> io::Result<()> {
        fs::create_dir_all(&self.directory)?;
        let temporary = self.directory.join("active.json.tmp");
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&serde_json::to_vec(pointer).map_err(io::Error::other)?)?;
        file.sync_all()?;
        fs::rename(temporary, self.pointer_path())
    }

    /// Path of the active personalized model, if one is installed and intact.
    #[must_use]
    pub fn active(&self) -> Option<PathBuf> {
        let path = self.artifact_path(&self.read_pointer().active?);
        path.exists().then_some(path)
    }

    /// Installs a verified artifact and activates it; the prior model becomes
    /// the rollback target.
    ///
    /// # Errors
    ///
    /// Fails when the artifact does not match the manifest or cannot be
    /// written.
    pub fn install(
        &self,
        manifest: &DeliveryManifest,
        artifact: &[u8],
    ) -> Result<PathBuf, DeliveryError> {
        if artifact.len() as u64 != manifest.artifact_size
            || hex::encode(Sha256::digest(artifact)) != manifest.artifact_sha256
        {
            return Err(DeliveryError::ArtifactMismatch);
        }
        fs::create_dir_all(&self.directory)?;
        let target = self.artifact_path(&manifest.model_version);
        let partial = target.with_extension("part");
        fs::write(&partial, artifact)?;
        fs::rename(&partial, &target)?;
        let current = self.read_pointer();
        self.write_pointer(&Pointer {
            active: Some(manifest.model_version.clone()),
            previous: current.active,
        })?;
        Ok(target)
    }

    /// Reactivates the previous model, or the base model if there is none.
    ///
    /// # Errors
    ///
    /// Fails when the pointer cannot be written.
    pub fn rollback(&self) -> Result<Option<PathBuf>, DeliveryError> {
        let current = self.read_pointer();
        self.write_pointer(&Pointer {
            active: current.previous.clone(),
            previous: None,
        })?;
        if let Some(abandoned) = current.active {
            let _ = fs::remove_file(self.artifact_path(&abandoned));
        }
        Ok(self.active())
    }

    /// Deletes every personalized model; the unchanged base model is used.
    ///
    /// # Errors
    ///
    /// Fails when files cannot be removed.
    pub fn delete_all(&self) -> Result<(), DeliveryError> {
        if self.directory.exists() {
            fs::remove_dir_all(&self.directory)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signer, SigningKey};

    use super::*;

    fn manifest(artifact: &[u8]) -> DeliveryManifest {
        DeliveryManifest {
            model_version: "cm-1".to_owned(),
            base_model_id: "whisper-base-q5_1".to_owned(),
            format: SUPPORTED_FORMAT.to_owned(),
            artifact_sha256: hex::encode(Sha256::digest(artifact)),
            artifact_size: artifact.len() as u64,
            peak_memory_mib: 300,
            created_at: 1,
            evaluation: Evaluation {
                heldout_wer_base: 0.20,
                heldout_wer_candidate: 0.15,
                regression_wer_base: 0.10,
                regression_wer_candidate: 0.105,
                improvement_threshold: 0.02,
                regression_ceiling: 0.01,
            },
        }
    }

    fn sign(key: &SigningKey, manifest: &DeliveryManifest) -> SignedManifest {
        let manifest_json = serde_json::to_string(manifest).unwrap();
        SignedManifest {
            signature_hex: hex::encode(key.sign(manifest_json.as_bytes()).to_bytes()),
            manifest_json,
        }
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[9; 32])
    }

    #[test]
    fn valid_delivery_installs_and_rolls_back_to_base() {
        let signing = key();
        let artifact = b"adapted weights".to_vec();
        let signed = sign(&signing, &manifest(&artifact));
        let verified =
            verify_manifest(&signed, &signing.verifying_key(), "whisper-base-q5_1", 1_000)
                .unwrap();
        let directory = std::env::temp_dir().join(format!("ld-delivery-{}", std::process::id()));
        let models = PersonalModels::new(&directory);
        models.install(&verified, &artifact).unwrap();
        assert!(models.active().is_some());
        assert_eq!(models.rollback().unwrap(), None);
        assert!(models.active().is_none());
        models.delete_all().unwrap();
        fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn tampering_and_policy_violations_are_refused() {
        let signing = key();
        let artifact = b"adapted weights".to_vec();
        let mut signed = sign(&signing, &manifest(&artifact));
        signed.manifest_json = signed.manifest_json.replace("cm-1", "cm-2");
        assert_eq!(
            verify_manifest(&signed, &signing.verifying_key(), "whisper-base-q5_1", 1_000),
            Err(DeliveryError::BadSignature)
        );
        let signed = sign(&signing, &manifest(&artifact));
        assert_eq!(
            verify_manifest(&signed, &signing.verifying_key(), "whisper-small-q5_1", 1_000),
            Err(DeliveryError::IncompatibleBase)
        );
        assert_eq!(
            verify_manifest(&signed, &signing.verifying_key(), "whisper-base-q5_1", 100),
            Err(DeliveryError::ExceedsDeviceBudget)
        );
        let mut weak = manifest(&artifact);
        weak.evaluation.heldout_wer_candidate = 0.195;
        assert_eq!(
            verify_manifest(&sign(&signing, &weak), &signing.verifying_key(), "whisper-base-q5_1", 1_000),
            Err(DeliveryError::EvaluationNotAccepted)
        );
        let other = SigningKey::from_bytes(&[3; 32]);
        assert_eq!(
            verify_manifest(&signed, &other.verifying_key(), "whisper-base-q5_1", 1_000),
            Err(DeliveryError::BadSignature)
        );
        let verified =
            verify_manifest(&signed, &signing.verifying_key(), "whisper-base-q5_1", 1_000)
                .unwrap();
        let directory = std::env::temp_dir().join(format!("ld-delivery-x-{}", std::process::id()));
        assert_eq!(
            PersonalModels::new(&directory).install(&verified, b"other bytes"),
            Err(DeliveryError::ArtifactMismatch)
        );
    }
}
