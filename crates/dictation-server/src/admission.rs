//! Server admission (plan §9). Client checks are not a security boundary: the
//! archive is re-validated here in full, consent is rechecked, and a declared
//! checksum mismatch is a rejection. Rejected material is never written to
//! disk; it is dropped when the request completes.

use std::collections::BTreeSet;

use dictation_core::{
    contribution::ELIGIBILITY_VERSION,
    package::{PackageLimits, decode_archive, validate_package},
};
use dictation_protocol::{MAX_ARCHIVE_BYTES, Receipt};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::store::{ServerError, ServerStore, StoredReceipt, Tenant};

fn to_receipt(stored: StoredReceipt, duplicate: bool) -> Receipt {
    Receipt {
        receipt_id: stored.receipt_id,
        accepted: stored.accepted,
        reason: stored.reason,
        duplicate,
    }
}

fn valid_key(key: &str) -> bool {
    (8..=128).contains(&key.len())
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

/// Admits or refuses one archive for an authenticated tenant.
///
/// # Errors
///
/// Fails only on storage errors; every validation failure is a refusal receipt.
pub fn admit(
    store: &mut ServerStore,
    tenant: &Tenant,
    idempotency_key: &str,
    declared_sha256: &str,
    archive: &[u8],
    now: i64,
) -> Result<Receipt, ServerError> {
    if !valid_key(idempotency_key) {
        return Ok(Receipt {
            receipt_id: String::new(),
            accepted: false,
            reason: "invalid_idempotency_key".to_owned(),
            duplicate: false,
        });
    }
    // Idempotency is tenant-scoped: another tenant's key never collides.
    if let Some(existing) = store.receipt_for(&tenant.tenant_id, idempotency_key)? {
        return Ok(to_receipt(existing, true));
    }
    let refuse = |store: &ServerStore, reason: &str| -> Result<Receipt, ServerError> {
        Ok(to_receipt(
            store.record_refusal(&tenant.tenant_id, idempotency_key, reason, now)?,
            false,
        ))
    };
    if archive.len() > MAX_ARCHIVE_BYTES {
        return refuse(store, "package_too_large");
    }
    if hex::encode(Sha256::digest(archive)) != declared_sha256 {
        return refuse(store, "declared_checksum_mismatch");
    }
    let limits = PackageLimits::default();
    let (manifest, payloads) = match decode_archive(archive, limits) {
        Ok(decoded) => decoded,
        Err(error) => return refuse(store, &format!("invalid_package:{}", error.code)),
    };
    let package = match validate_package(
        &manifest,
        &payloads,
        &BTreeSet::from([ELIGIBILITY_VERSION]),
        limits,
    ) {
        Ok(package) => package,
        Err(error) => return refuse(store, &format!("invalid_package:{}", error.code)),
    };
    let consent_id = manifest
        .get("consent_reference")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if !store.consent_active(&tenant.tenant_id, &consent_id, now)? {
        return refuse(store, "consent_not_active");
    }
    if store.is_tombstoned(&package.sample_id)? {
        // A replayed upload or restored backup must not reintroduce it.
        return refuse(store, "sample_previously_deleted");
    }
    if store.sample_exists(&package.sample_id)? {
        return refuse(store, "sample_id_conflict");
    }
    let (count, bytes) = store.usage(&tenant.tenant_id)?;
    if count >= tenant.max_samples || bytes + archive.len() as u64 > tenant.max_storage_bytes {
        return refuse(store, "tenant_storage_cap_reached");
    }
    let stored = store.admit_sample(
        &tenant.tenant_id,
        idempotency_key,
        &package.sample_id,
        &consent_id,
        archive,
        package.duration_ms,
        package.clip_count,
        now,
    )?;
    Ok(to_receipt(stored, false))
}
