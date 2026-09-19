//! Versioned loopback HTTP/WebSocket API for native clients (plan §8).
//!
//! * Disabled until the user enables integration access; binds loopback only.
//! * Loopback is not authentication: every endpoint except pairing requires a
//!   bearer token with an explicit scope, sent in a header, never a URL.
//! * Host must name the loopback listener and any browser `Origin` is
//!   refused, so a malicious web page cannot drive the API.
//! * Pairing is requested by the client and approved by the user in the
//!   desktop UI; pending requests are bounded and rate-limited.
//! * Event buffers are bounded; a slow client receives a resynchronization
//!   error and is disconnected. There is no transcript replay archive.
//! * Revocation is rechecked while a stream is open.
//! * Errors and logs never contain dictated text.

pub mod events;
pub mod pairing;
pub mod server;

pub use events::{ApiEvent, EventBus, EventKind};
pub use pairing::{PairingRequest, Pairings};
pub use server::{ApiConfig, ApiHandle, ClientDirectory, Controller, ControlError, StatusSnapshot, start};

pub const API_VERSION: u32 = 1;
