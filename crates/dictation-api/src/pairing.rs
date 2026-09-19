//! Client pairing: requested by the client, approved in the desktop UI.
//!
//! A pending request shows the client's chosen name, requested scopes, and a
//! short verification code that the user compares with the client. Approval
//! may grant fewer scopes than requested. The token is released exactly once
//! to the holder of the request's poll secret.

use std::{
    collections::BTreeMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use dictation_storage::api_clients::Scope;
use rand::{Rng, RngCore};

pub const MAX_PENDING: usize = 4;
pub const PENDING_LIFETIME: Duration = Duration::from_secs(300);
pub const MAX_REQUESTS_PER_MINUTE: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingRequest {
    pub pairing_id: String,
    pub client_name: String,
    pub requested_scopes: Vec<Scope>,
    pub verification_code: String,
}

#[derive(Debug)]
enum Stage {
    Pending,
    Approved { token: String },
    Denied,
}

#[derive(Debug)]
struct Entry {
    request: PairingRequest,
    poll_secret: String,
    created: Instant,
    stage: Stage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollResult {
    Pending,
    Approved { token: String },
    Denied,
    Unknown,
}

#[derive(Debug, Default)]
pub struct Pairings {
    entries: Mutex<BTreeMap<String, Entry>>,
    recent: Mutex<Vec<Instant>>,
}

fn random_hex(bytes: usize) -> String {
    let mut buffer = vec![0_u8; bytes];
    rand::rngs::OsRng.fill_bytes(&mut buffer);
    hex::encode(buffer)
}

impl Pairings {
    fn prune(entries: &mut BTreeMap<String, Entry>) {
        entries.retain(|_, entry| entry.created.elapsed() < PENDING_LIFETIME);
    }

    /// Opens a request. Returns `(request, poll_secret)` or `None` when rate
    /// or capacity limits are hit.
    pub fn open(&self, client_name: &str, scopes: Vec<Scope>) -> Option<(PairingRequest, String)> {
        let mut recent = self.recent.lock().ok()?;
        recent.retain(|when| when.elapsed() < Duration::from_secs(60));
        if recent.len() >= MAX_REQUESTS_PER_MINUTE {
            return None;
        }
        recent.push(Instant::now());
        let mut entries = self.entries.lock().ok()?;
        Self::prune(&mut entries);
        if entries.values().filter(|entry| matches!(entry.stage, Stage::Pending)).count() >= MAX_PENDING {
            return None;
        }
        let name: String = client_name
            .chars()
            .filter(|character| !character.is_control())
            .take(48)
            .collect();
        let request = PairingRequest {
            pairing_id: format!("pair-{}", random_hex(8)),
            client_name: if name.trim().is_empty() { "Unnamed client".to_owned() } else { name },
            requested_scopes: scopes,
            verification_code: format!("{:04}", rand::rngs::OsRng.gen_range(0..10_000)),
        };
        let secret = random_hex(24);
        entries.insert(
            request.pairing_id.clone(),
            Entry {
                request: request.clone(),
                poll_secret: secret.clone(),
                created: Instant::now(),
                stage: Stage::Pending,
            },
        );
        Some((request, secret))
    }

    /// Pending requests for the desktop UI.
    #[must_use]
    pub fn pending(&self) -> Vec<PairingRequest> {
        let Ok(mut entries) = self.entries.lock() else { return Vec::new() };
        Self::prune(&mut entries);
        entries
            .values()
            .filter(|entry| matches!(entry.stage, Stage::Pending))
            .map(|entry| entry.request.clone())
            .collect()
    }

    /// Marks a request approved with an already-created token.
    pub fn approve(&self, pairing_id: &str, token: String) -> bool {
        let Ok(mut entries) = self.entries.lock() else { return false };
        match entries.get_mut(pairing_id) {
            Some(entry) if matches!(entry.stage, Stage::Pending) => {
                entry.stage = Stage::Approved { token };
                true
            }
            _ => false,
        }
    }

    pub fn deny(&self, pairing_id: &str) -> bool {
        let Ok(mut entries) = self.entries.lock() else { return false };
        match entries.get_mut(pairing_id) {
            Some(entry) if matches!(entry.stage, Stage::Pending) => {
                entry.stage = Stage::Denied;
                true
            }
            _ => false,
        }
    }

    #[must_use]
    pub fn request(&self, pairing_id: &str) -> Option<PairingRequest> {
        self.entries.lock().ok()?.get(pairing_id).map(|entry| entry.request.clone())
    }

    /// Releases the token once; later polls see `Unknown`.
    pub fn poll(&self, pairing_id: &str, secret: &str) -> PollResult {
        let Ok(mut entries) = self.entries.lock() else { return PollResult::Unknown };
        Self::prune(&mut entries);
        let Some(entry) = entries.get(pairing_id) else { return PollResult::Unknown };
        if !constant_time_eq(entry.poll_secret.as_bytes(), secret.as_bytes()) {
            return PollResult::Unknown;
        }
        match &entry.stage {
            Stage::Pending => PollResult::Pending,
            Stage::Denied => {
                entries.remove(pairing_id);
                PollResult::Denied
            }
            Stage::Approved { .. } => match entries.remove(pairing_id).map(|entry| entry.stage) {
                Some(Stage::Approved { token }) => PollResult::Approved { token },
                _ => PollResult::Unknown,
            },
        }
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && left.iter().zip(right).fold(0_u8, |diff, (a, b)| diff | (a ^ b)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_released_once_to_the_secret_holder() {
        let pairings = Pairings::default();
        let (request, secret) = pairings.open("Captions", vec![Scope::TranscriptLive]).unwrap();
        assert_eq!(pairings.poll(&request.pairing_id, &secret), PollResult::Pending);
        assert_eq!(pairings.poll(&request.pairing_id, "wrong"), PollResult::Unknown);
        assert!(pairings.approve(&request.pairing_id, "token-1".to_owned()));
        assert_eq!(pairings.poll(&request.pairing_id, "wrong"), PollResult::Unknown);
        assert_eq!(
            pairings.poll(&request.pairing_id, &secret),
            PollResult::Approved { token: "token-1".to_owned() }
        );
        assert_eq!(pairings.poll(&request.pairing_id, &secret), PollResult::Unknown);
    }

    #[test]
    fn pending_requests_are_bounded_and_rate_limited() {
        let pairings = Pairings::default();
        for _ in 0..MAX_PENDING {
            assert!(pairings.open("c", vec![]).is_some());
        }
        assert!(pairings.open("c", vec![]).is_none());
        let limited = Pairings::default();
        for index in 0..MAX_REQUESTS_PER_MINUTE {
            if let Some((request, _)) = limited.open("c", vec![]) {
                limited.deny(&request.pairing_id);
            } else {
                panic!("rejected early at {index}");
            }
        }
        assert!(limited.open("c", vec![]).is_none());
    }
}
