//! Explicitly imported whisper.cpp GGML weights, kept separate from pinned models.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

pub const ID: &str = "custom-whisper";
const MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModel {
    pub sha256: String,
    pub size_bytes: u64,
}

impl CustomModel {
    pub fn path(&self, directory: &Path) -> Result<PathBuf, String> {
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("Invalid custom model checksum.".into());
        }
        Ok(directory.join(format!("custom-{}.bin", self.sha256)))
    }

    pub fn verified_path(&self, directory: &Path) -> Result<PathBuf, String> {
        let path = self.path(directory)?;
        let (size, digest) = super::sha256_file(&path).map_err(|e| e.to_string())?;
        if size != self.size_bytes || digest != self.sha256 {
            return Err("Custom model changed on disk; import it again.".into());
        }
        Ok(path)
    }
}

pub fn installed(directory: &Path) -> Option<CustomModel> {
    serde_json::from_slice(&fs::read(directory.join("custom-whisper.json")).ok()?).ok()
}

/// A local file is fingerprinted on import. HTTPS downloads require the
/// publisher's SHA-256, so an HTML login/error page cannot become a model.
pub fn install(
    source: &str,
    expected_sha256: &str,
    directory: &Path,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<CustomModel, String> {
    let expected = expected_sha256.trim().to_ascii_lowercase();
    if !expected.is_empty()
        && (expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("SHA-256 must contain 64 hexadecimal characters.".into());
    }
    let mut reader: Box<dyn Read> = if source.starts_with("https://") {
        if expected.is_empty() {
            return Err("Provide the publisher's SHA-256 for a custom download.".into());
        }
        let client = reqwest::blocking::Client::builder()
            .https_only(true)
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(3600))
            .build()
            .map_err(|e| e.to_string())?;
        Box::new(
            client
                .get(source)
                .send()
                .and_then(reqwest::blocking::Response::error_for_status)
                .map_err(|e| e.to_string())?,
        )
    } else {
        if source.contains("://") {
            return Err("Downloads must use HTTPS.".into());
        }
        Box::new(File::open(source).map_err(|e| e.to_string())?)
    };
    fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let partial = directory.join("custom-whisper.part");
    let result = (|| {
        let mut header = [0_u8; 4];
        reader.read_exact(&mut header).map_err(|e| e.to_string())?;
        if header != 0x6767_6d6c_u32.to_le_bytes() {
            return Err("Expected whisper.cpp GGML .bin weights. GGUF, safetensors and PyTorch checkpoints must be converted first.".into());
        }
        let mut file = File::create(&partial).map_err(|e| e.to_string())?;
        file.write_all(&header).map_err(|e| e.to_string())?;
        let mut digest = Sha256::new();
        digest.update(header);
        let mut size = 4_u64;
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("Model installation cancelled.".into());
            }
            let read = reader.read(&mut buffer).map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            size += read as u64;
            if size > MAX_BYTES {
                return Err("Custom model exceeds the 8 GiB limit.".into());
            }
            digest.update(&buffer[..read]);
            file.write_all(&buffer[..read]).map_err(|e| e.to_string())?;
            progress(size, 0);
        }
        let sha256 = hex::encode(digest.finalize());
        if !expected.is_empty() && expected != sha256 {
            return Err("Custom model checksum mismatch.".into());
        }
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        let model = CustomModel {
            sha256,
            size_bytes: size,
        };
        fs::rename(&partial, model.path(directory)?).map_err(|e| e.to_string())?;
        let manifest = directory.join("custom-whisper.json.part");
        fs::write(
            &manifest,
            serde_json::to_vec(&model).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::rename(manifest, directory.join("custom-whisper.json")).map_err(|e| e.to_string())?;
        Ok(model)
    })();
    if result.is_err() {
        let _ = fs::remove_file(partial);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn import_verify_replace_and_reject_invalid_weights() {
        let directory = std::env::temp_dir().join(format!("custom-model-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("source.bin");
        let payload = [0x6767_6d6c_u32.to_le_bytes().as_slice(), &[1; 64]].concat();
        fs::write(&source, &payload).unwrap();
        let cancel = AtomicBool::new(false);
        let install_local = |checksum: &str| {
            install(
                source.to_str().unwrap(),
                checksum,
                &directory,
                &cancel,
                |_, _| {},
            )
        };
        let model = install_local("").unwrap();
        assert!(model.verified_path(&directory).is_ok());
        install_local("").unwrap();
        assert!(install_local(&"0".repeat(64)).is_err());
        assert_eq!(installed(&directory).unwrap().sha256, model.sha256);
        fs::write(model.path(&directory).unwrap(), b"tampered").unwrap();
        assert!(model.verified_path(&directory).is_err());
        fs::write(&source, b"<html>not a model</html>").unwrap();
        assert!(install_local("").is_err());
        assert!(!directory.join("custom-whisper.part").exists());
        fs::remove_dir_all(directory).unwrap();
    }
}
