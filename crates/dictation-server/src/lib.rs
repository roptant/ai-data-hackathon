//! Training service for Local Dictation (plan §9, §10).
//!
//! Admission re-validates every package, tenant identity comes only from the
//! credential, objects are encrypted per tenant, lineage drives deletion, and
//! personalized models are delivered only with a signed, measured manifest.

pub mod admission;
pub mod http;
pub mod store;
pub mod training;
