//! Server-grade validation for extracted upload-package contents.
//!
//! The validator requires a complete allowlisted manifest, generated safe entry
//! names, bounded payloads, checksum coverage, canonical WAV parameters, and
//! agreement between declared and observed clip metrics.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const MANIFEST_KEYS: [&str; 16] = [
    "eligibility_version",
    "sample_id",
    "sample_rate",
    "language",
    "duration_ms",
    "clip_count",
    "quality",
    "asr_model",
    "asr_model_revision",
    "privacy_model",
    "privacy_model_revision",
    "policy_version",
    "consent_version",
    "consent_reference",
    "clips",
    "checksums",
];
const CLIP_KEYS: [&str; 5] = ["audio", "text", "duration_ms", "word_count", "sha256"];
const QUALITY_KEYS: [&str; 7] = [
    "duration_ms",
    "word_count",
    "mean_confidence",
    "speech_ratio",
    "removed_interval_count",
    "removed_duration_ms",
    "clip_count",
];
const FORBIDDEN_KEYS: [&str; 36] = [
    "tenant",
    "tenant_id",
    "customer_id",
    "account_id",
    "session_id",
    "session",
    "user",
    "username",
    "user_name",
    "hostname",
    "device_name",
    "machine",
    "path",
    "filename",
    "file_path",
    "app",
    "application",
    "window_title",
    "foreground_app",
    "prompt",
    "system_prompt",
    "spans",
    "removed",
    "removal_map",
    "boundary_manifest",
    "source_start",
    "source_end",
    "source_interval",
    "original_text",
    "full_transcript",
    "latitude",
    "longitude",
    "location",
    "ip",
    "ip_address",
    "created_at",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageLimits {
    pub max_payload_bytes: u64,
    pub max_clip_bytes: u64,
    pub max_clips: usize,
    pub max_text_bytes: usize,
}

impl Default for PackageLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: 32 * 1024 * 1024,
            max_clip_bytes: 8 * 1024 * 1024,
            max_clips: 128,
            max_text_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPackage {
    pub sample_id: String,
    pub eligibility_version: u64,
    pub sample_rate: u32,
    pub language: String,
    pub duration_ms: u64,
    pub clip_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageError {
    pub code: &'static str,
    pub detail: String,
}

impl PackageError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for PackageError {}

/// Validates a parsed manifest and files extracted from its archive.
///
/// # Errors
///
/// Returns [`PackageError`] for any missing, extra, malformed, oversized,
/// inconsistent, unsafe, or checksum-invalid field or payload.
pub fn validate_package(
    manifest: &Value,
    payloads: &BTreeMap<String, Vec<u8>>,
    known_eligibility_versions: &BTreeSet<u64>,
    limits: PackageLimits,
) -> Result<ValidatedPackage, PackageError> {
    let object = manifest
        .as_object()
        .ok_or_else(|| PackageError::new("package_manifest_not_object", "manifest"))?;
    require_exact_keys(object, &MANIFEST_KEYS, "manifest")?;
    reject_forbidden_keys(manifest, "manifest")?;
    let header = validate_header(object, known_eligibility_versions, limits)?;
    validate_payload_bounds(payloads, limits)?;
    let checksums = object
        .get("checksums")
        .and_then(Value::as_object)
        .ok_or_else(|| PackageError::new("invalid_checksums", "must be an object"))?;
    validate_checksum_coverage(checksums, payloads)?;
    let (observed_duration, observed_words) = validate_clips(
        object.get("clips"),
        payloads,
        header.sample_rate,
        header.clip_count,
        limits,
    )?;
    if header.duration_ms != observed_duration {
        return Err(PackageError::new(
            "package_duration_mismatch",
            format!(
                "declared {}, observed {observed_duration}",
                header.duration_ms
            ),
        ));
    }
    validate_quality(
        object.get("quality"),
        observed_duration,
        observed_words,
        header.clip_count,
    )?;

    Ok(ValidatedPackage {
        sample_id: header.sample_id,
        eligibility_version: header.eligibility_version,
        sample_rate: header.sample_rate,
        language: header.language,
        duration_ms: header.duration_ms,
        clip_count: header.clip_count,
    })
}

struct ManifestHeader {
    sample_id: String,
    eligibility_version: u64,
    sample_rate: u32,
    language: String,
    duration_ms: u64,
    clip_count: usize,
}

fn validate_header(
    object: &Map<String, Value>,
    known_versions: &BTreeSet<u64>,
    limits: PackageLimits,
) -> Result<ManifestHeader, PackageError> {
    let eligibility_version = required_u64(object, "eligibility_version", "manifest")?;
    if !known_versions.contains(&eligibility_version) {
        return Err(PackageError::new(
            "unknown_eligibility_version",
            eligibility_version.to_string(),
        ));
    }
    let sample_id = required_string(object, "sample_id", "manifest")?;
    validate_sample_id(sample_id)?;
    let rate_value = required_u64(object, "sample_rate", "manifest")?;
    let sample_rate = u32::try_from(rate_value)
        .ok()
        .filter(|rate| *rate == 16_000)
        .ok_or_else(|| PackageError::new("invalid_sample_rate", rate_value.to_string()))?;
    let language = required_string(object, "language", "manifest")?;
    if language != "en" {
        return Err(PackageError::new("language_not_validated", language));
    }
    let duration_ms = required_u64(object, "duration_ms", "manifest")?;
    let clip_count = required_usize(object, "clip_count", "manifest")?;
    if clip_count == 0 || clip_count > limits.max_clips {
        return Err(PackageError::new(
            "invalid_clip_count",
            clip_count.to_string(),
        ));
    }
    for field in [
        "asr_model",
        "asr_model_revision",
        "privacy_model",
        "privacy_model_revision",
        "policy_version",
        "consent_version",
        "consent_reference",
    ] {
        let value = required_string(object, field, "manifest")?;
        if value.is_empty() || value.len() > 256 {
            return Err(PackageError::new("invalid_manifest_string", field));
        }
    }
    Ok(ManifestHeader {
        sample_id: sample_id.to_owned(),
        eligibility_version,
        sample_rate,
        language: language.to_owned(),
        duration_ms,
        clip_count,
    })
}

fn validate_clips(
    value: Option<&Value>,
    payloads: &BTreeMap<String, Vec<u8>>,
    sample_rate: u32,
    declared_count: usize,
    limits: PackageLimits,
) -> Result<(u64, u64), PackageError> {
    let clips = value
        .and_then(Value::as_array)
        .ok_or_else(|| PackageError::new("invalid_clips", "must be an array"))?;
    if clips.len() != declared_count {
        return Err(PackageError::new(
            "clip_count_mismatch",
            format!("declared {declared_count}, found {}", clips.len()),
        ));
    }
    let mut duration = 0_u64;
    let mut words = 0_u64;
    let mut referenced = BTreeSet::new();
    for (index, clip) in clips.iter().enumerate() {
        let (clip_duration, clip_words, audio_name, text_name) =
            validate_clip(clip, index, payloads, sample_rate, limits)?;
        referenced.insert(audio_name);
        referenced.insert(text_name);
        duration = duration.saturating_add(clip_duration);
        words = words.saturating_add(clip_words);
    }
    if referenced != payloads.keys().cloned().collect() {
        return Err(PackageError::new(
            "package_unreferenced_payload",
            "payload not referenced by clips",
        ));
    }
    Ok((duration, words))
}

fn validate_clip(
    value: &Value,
    index: usize,
    payloads: &BTreeMap<String, Vec<u8>>,
    sample_rate: u32,
    limits: PackageLimits,
) -> Result<(u64, u64, String, String), PackageError> {
    let location = format!("clip[{index}]");
    let clip = value
        .as_object()
        .ok_or_else(|| PackageError::new("invalid_clip", index.to_string()))?;
    require_exact_keys(clip, &CLIP_KEYS, &location)?;
    let audio_name = required_string(clip, "audio", &location)?;
    let text_name = required_string(clip, "text", &location)?;
    if audio_name != format!("clip-{index:03}.wav") || text_name != format!("clip-{index:03}.txt") {
        return Err(PackageError::new(
            "unsafe_clip_filename",
            format!("{audio_name}, {text_name}"),
        ));
    }
    let audio = payloads
        .get(audio_name)
        .ok_or_else(|| PackageError::new("package_missing_file", audio_name))?;
    let text = payloads
        .get(text_name)
        .ok_or_else(|| PackageError::new("package_missing_file", text_name))?;
    if required_string(clip, "sha256", &location)? != sha256_hex(audio) {
        return Err(PackageError::new("package_checksum_mismatch", audio_name));
    }
    let samples = validate_wav(audio, sample_rate)?;
    let actual_duration = samples
        .saturating_mul(1_000)
        .saturating_add(u64::from(sample_rate) / 2)
        / u64::from(sample_rate);
    if required_u64(clip, "duration_ms", &location)? != actual_duration {
        return Err(PackageError::new(
            "clip_duration_mismatch",
            index.to_string(),
        ));
    }
    let text = std::str::from_utf8(text)
        .map_err(|_| PackageError::new("clip_text_not_utf8", text_name))?;
    if text.trim().is_empty() || text.len() > limits.max_text_bytes {
        return Err(PackageError::new("invalid_clip_text", text_name));
    }
    let actual_words = u64::try_from(text.split_whitespace().count()).unwrap_or(u64::MAX);
    if required_u64(clip, "word_count", &location)? != actual_words {
        return Err(PackageError::new(
            "clip_word_count_mismatch",
            index.to_string(),
        ));
    }
    Ok((
        actual_duration,
        actual_words,
        audio_name.to_owned(),
        text_name.to_owned(),
    ))
}

fn validate_payload_bounds(
    payloads: &BTreeMap<String, Vec<u8>>,
    limits: PackageLimits,
) -> Result<(), PackageError> {
    let mut total = 0_u64;
    for (name, payload) in payloads {
        if !safe_entry_name(name) {
            return Err(PackageError::new("unsafe_archive_entry", name));
        }
        let size = u64::try_from(payload.len()).unwrap_or(u64::MAX);
        if size > limits.max_clip_bytes {
            return Err(PackageError::new("package_file_too_large", name));
        }
        total = total.saturating_add(size);
    }
    if total > limits.max_payload_bytes {
        return Err(PackageError::new("package_too_large", total.to_string()));
    }
    Ok(())
}

fn validate_checksum_coverage(
    checksums: &Map<String, Value>,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), PackageError> {
    let checksum_names: BTreeSet<_> = checksums.keys().cloned().collect();
    let payload_names: BTreeSet<_> = payloads.keys().cloned().collect();
    if checksum_names != payload_names {
        return Err(PackageError::new(
            "package_checksum_coverage_mismatch",
            "checksums must cover exactly all payloads",
        ));
    }
    for (name, payload) in payloads {
        let expected = checksums
            .get(name)
            .and_then(Value::as_str)
            .filter(|digest| valid_digest(digest))
            .ok_or_else(|| PackageError::new("invalid_checksum", name))?;
        if sha256_hex(payload) != expected {
            return Err(PackageError::new("package_checksum_mismatch", name));
        }
    }
    Ok(())
}

fn validate_quality(
    value: Option<&Value>,
    duration_ms: u64,
    word_count: u64,
    clip_count: usize,
) -> Result<(), PackageError> {
    let quality = value
        .and_then(Value::as_object)
        .ok_or_else(|| PackageError::new("invalid_quality", "must be an object"))?;
    require_exact_keys(quality, &QUALITY_KEYS, "quality")?;
    if required_u64(quality, "duration_ms", "quality")? != duration_ms
        || required_u64(quality, "word_count", "quality")? != word_count
        || required_usize(quality, "clip_count", "quality")? != clip_count
    {
        return Err(PackageError::new(
            "quality_metrics_mismatch",
            "duration, word count, or clip count",
        ));
    }
    for field in ["removed_interval_count", "removed_duration_ms"] {
        required_u64(quality, field, "quality")?;
    }
    for field in ["mean_confidence", "speech_ratio"] {
        quality
            .get(field)
            .and_then(Value::as_f64)
            .filter(|number| number.is_finite() && (0.0..=1.0).contains(number))
            .ok_or_else(|| PackageError::new("invalid_quality_metric", field))?;
    }
    Ok(())
}

fn validate_wav(bytes: &[u8], expected_rate: u32) -> Result<u64, PackageError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(PackageError::new("invalid_wav", "missing RIFF/WAVE header"));
    }
    let declared_size = usize::try_from(read_u32(bytes, 4)?).unwrap_or(usize::MAX);
    if declared_size.checked_add(8) != Some(bytes.len()) {
        return Err(PackageError::new("invalid_wav", "RIFF size mismatch"));
    }
    let mut offset = 12_usize;
    let mut format_ok = false;
    let mut data_samples = None;
    while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let chunk_id = &bytes[offset..offset + 4];
        let size = usize::try_from(read_u32(bytes, offset + 4)?).unwrap_or(usize::MAX);
        let start = offset + 8;
        let end = start
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| PackageError::new("invalid_wav", "chunk exceeds payload"))?;
        if chunk_id == b"fmt " {
            if size < 16 {
                return Err(PackageError::new("invalid_wav", "short format chunk"));
            }
            format_ok = read_u16(bytes, start)? == 1
                && read_u16(bytes, start + 2)? == 1
                && read_u32(bytes, start + 4)? == expected_rate
                && read_u16(bytes, start + 12)? == 2
                && read_u16(bytes, start + 14)? == 16;
        } else if chunk_id == b"data" {
            if size % 2 != 0 {
                return Err(PackageError::new("invalid_wav", "unaligned PCM data"));
            }
            data_samples = Some(u64::try_from(size / 2).unwrap_or(u64::MAX));
        }
        offset = end.saturating_add(size % 2);
    }
    if !format_ok {
        return Err(PackageError::new(
            "invalid_wav_format",
            "expected 16 kHz mono PCM16",
        ));
    }
    data_samples.ok_or_else(|| PackageError::new("invalid_wav", "missing data chunk"))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, PackageError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| PackageError::new("invalid_wav", "truncated u16"))?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PackageError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| PackageError::new("invalid_wav", "truncated u32"))?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    location: &str,
) -> Result<&'a str, PackageError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| PackageError::new("invalid_field_type", format!("{location}.{key}")))
}

fn required_u64(
    object: &Map<String, Value>,
    key: &str,
    location: &str,
) -> Result<u64, PackageError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| PackageError::new("invalid_field_type", format!("{location}.{key}")))
}

fn required_usize(
    object: &Map<String, Value>,
    key: &str,
    location: &str,
) -> Result<usize, PackageError> {
    usize::try_from(required_u64(object, key, location)?)
        .map_err(|_| PackageError::new("integer_out_of_range", format!("{location}.{key}")))
}

fn require_exact_keys(
    object: &Map<String, Value>,
    required: &[&str],
    location: &str,
) -> Result<(), PackageError> {
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = required.iter().copied().collect();
    if actual == expected {
        Ok(())
    } else {
        Err(PackageError::new(
            "package_fields_mismatch",
            format!("{location}: expected {expected:?}, found {actual:?}"),
        ))
    }
}

fn reject_forbidden_keys(value: &Value, path: &str) -> Result<(), PackageError> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if FORBIDDEN_KEYS.contains(&key.to_ascii_lowercase().as_str()) {
                    return Err(PackageError::new(
                        "package_forbidden_field",
                        format!("{path}.{key}"),
                    ));
                }
                reject_forbidden_keys(child, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_forbidden_keys(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_sample_id(value: &str) -> Result<(), PackageError> {
    let suffix = value
        .strip_prefix("sample-")
        .ok_or_else(|| PackageError::new("invalid_sample_id", value))?;
    if matches!(suffix.len(), 16 | 32) && suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(PackageError::new("invalid_sample_id", value))
    }
}

fn safe_entry_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && !name.contains("..")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(payload: &[u8]) -> String {
    let digest = Sha256::digest(payload);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn wav(samples: usize) -> Vec<u8> {
        let data_size = u32::try_from(samples * 2).unwrap();
        let mut output = Vec::with_capacity(44 + samples * 2);
        output.extend_from_slice(b"RIFF");
        output.extend_from_slice(&(36 + data_size).to_le_bytes());
        output.extend_from_slice(b"WAVEfmt ");
        output.extend_from_slice(&16_u32.to_le_bytes());
        output.extend_from_slice(&1_u16.to_le_bytes());
        output.extend_from_slice(&1_u16.to_le_bytes());
        output.extend_from_slice(&16_000_u32.to_le_bytes());
        output.extend_from_slice(&32_000_u32.to_le_bytes());
        output.extend_from_slice(&2_u16.to_le_bytes());
        output.extend_from_slice(&16_u16.to_le_bytes());
        output.extend_from_slice(b"data");
        output.extend_from_slice(&data_size.to_le_bytes());
        output.resize(44 + samples * 2, 0);
        output
    }

    fn valid_fixture() -> (Value, BTreeMap<String, Vec<u8>>) {
        let audio = wav(16_000);
        let text = b"meeting starts tomorrow".to_vec();
        let audio_hash = sha256_hex(&audio);
        let text_hash = sha256_hex(&text);
        let payloads = BTreeMap::from([
            ("clip-000.wav".to_owned(), audio),
            ("clip-000.txt".to_owned(), text),
        ]);
        let manifest = json!({
            "eligibility_version": 1,
            "sample_id": "sample-0123456789abcdef",
            "sample_rate": 16000,
            "language": "en",
            "duration_ms": 1000,
            "clip_count": 1,
            "quality": {
                "duration_ms": 1000, "word_count": 3, "mean_confidence": 0.9,
                "speech_ratio": 0.8, "removed_interval_count": 1,
                "removed_duration_ms": 500, "clip_count": 1
            },
            "asr_model": "whisper", "asr_model_revision": "r1",
            "privacy_model": "local-llm", "privacy_model_revision": "r1",
            "policy_version": "policy-1", "consent_version": "consent-1",
            "consent_reference": "grant-1",
            "clips": [{
                "audio": "clip-000.wav", "text": "clip-000.txt", "duration_ms": 1000,
                "word_count": 3, "sha256": audio_hash
            }],
            "checksums": { "clip-000.wav": audio_hash, "clip-000.txt": text_hash }
        });
        (manifest, payloads)
    }

    fn validate(
        manifest: &Value,
        payloads: &BTreeMap<String, Vec<u8>>,
    ) -> Result<ValidatedPackage, PackageError> {
        validate_package(
            manifest,
            payloads,
            &BTreeSet::from([1]),
            PackageLimits::default(),
        )
    }

    #[test]
    fn valid_package_is_accepted() {
        let (manifest, payloads) = valid_fixture();
        let package = validate(&manifest, &payloads).unwrap();
        assert_eq!(package.sample_id, "sample-0123456789abcdef");
        assert_eq!(package.duration_ms, 1_000);
    }

    #[test]
    fn every_top_level_field_is_required() {
        let (manifest, payloads) = valid_fixture();
        for key in MANIFEST_KEYS {
            let mut candidate = manifest.clone();
            candidate.as_object_mut().unwrap().remove(key);
            assert!(
                validate(&candidate, &payloads).is_err(),
                "accepted without {key}"
            );
        }
    }

    #[test]
    fn forbidden_and_unknown_fields_are_rejected() {
        let (mut manifest, payloads) = valid_fixture();
        manifest["session_id"] = json!("leak");
        assert!(validate(&manifest, &payloads).is_err());
    }

    #[test]
    fn unsafe_filename_is_rejected() {
        let (mut manifest, mut payloads) = valid_fixture();
        let audio = payloads.remove("clip-000.wav").unwrap();
        payloads.insert("../escape.wav".to_owned(), audio);
        manifest["clips"][0]["audio"] = json!("../escape.wav");
        let digest = manifest["checksums"]
            .as_object_mut()
            .unwrap()
            .remove("clip-000.wav")
            .unwrap();
        manifest["checksums"]["../escape.wav"] = digest;
        assert_eq!(
            validate(&manifest, &payloads).unwrap_err().code,
            "unsafe_archive_entry"
        );
    }

    #[test]
    fn checksum_mismatch_is_rejected() {
        let (manifest, mut payloads) = valid_fixture();
        payloads.get_mut("clip-000.txt").unwrap().push(b'!');
        assert_eq!(
            validate(&manifest, &payloads).unwrap_err().code,
            "package_checksum_mismatch"
        );
    }

    #[test]
    fn declared_metrics_must_match_payloads() {
        let (mut manifest, payloads) = valid_fixture();
        manifest["clips"][0]["duration_ms"] = json!(999);
        assert_eq!(
            validate(&manifest, &payloads).unwrap_err().code,
            "clip_duration_mismatch"
        );
    }

    #[test]
    fn wav_must_be_canonical_pcm() {
        let (mut manifest, mut payloads) = valid_fixture();
        let audio = payloads.get_mut("clip-000.wav").unwrap();
        audio[22..24].copy_from_slice(&2_u16.to_le_bytes());
        let digest = sha256_hex(audio);
        manifest["checksums"]["clip-000.wav"] = json!(digest);
        manifest["clips"][0]["sha256"] = manifest["checksums"]["clip-000.wav"].clone();
        assert_eq!(
            validate(&manifest, &payloads).unwrap_err().code,
            "invalid_wav_format"
        );
    }
}
