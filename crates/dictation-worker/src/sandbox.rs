//! Post-load confinement for model workers.
//!
//! Called after the model weights are open. On Linux, Landlock removes all new
//! filesystem access and seccomp makes `socket(2)` fail, so a compromised or
//! prompt-injected worker has no network or filesystem authority beyond its
//! inherited pipes. Other platforms report what was not confined; the host
//! displays that instead of assuming isolation.

use crate::messages::SandboxReport;

#[cfg(target_os = "linux")]
#[must_use]
pub fn confine() -> SandboxReport {
    let mut report = SandboxReport::default();
    match restrict_filesystem() {
        Ok(true) => report.filesystem_restricted = true,
        Ok(false) => report
            .notes
            .push("landlock_not_enforced_by_kernel".to_owned()),
        Err(error) => report.notes.push(format!("landlock_failed:{error}")),
    }
    match deny_sockets() {
        Ok(()) => report.network_denied = true,
        Err(error) => report.notes.push(format!("seccomp_failed:{error}")),
    }
    report
}

#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn confine() -> SandboxReport {
    SandboxReport {
        network_denied: false,
        filesystem_restricted: false,
        notes: vec!["os_sandbox_not_implemented_on_this_platform".to_owned()],
    }
}

#[cfg(target_os = "linux")]
fn restrict_filesystem() -> Result<bool, String> {
    use landlock::{ABI, Access, AccessFs, Ruleset, RulesetAttr, RulesetStatus};

    let status = Ruleset::default()
        .handle_access(AccessFs::from_all(ABI::V5))
        .map_err(|error| error.to_string())?
        .create()
        .map_err(|error| error.to_string())?
        .restrict_self()
        .map_err(|error| error.to_string())?;
    Ok(status.ruleset == RulesetStatus::FullyEnforced
        || status.ruleset == RulesetStatus::PartiallyEnforced)
}

#[cfg(target_os = "linux")]
fn deny_sockets() -> Result<(), String> {
    use std::collections::BTreeMap;

    use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, TargetArch};

    let arch: TargetArch = std::env::consts::ARCH
        .try_into()
        .map_err(|_| "unsupported_arch".to_owned())?;
    let mut rules = BTreeMap::new();
    for syscall in [libc_socket(), libc_socketpair()] {
        rules.insert(syscall, Vec::new());
    }
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(1), // EPERM for the listed syscalls
        arch,
    )
    .map_err(|error| error.to_string())?;
    let program: BpfProgram = filter.try_into().map_err(|error: seccompiler::BackendError| {
        error.to_string()
    })?;
    seccompiler::apply_filter(&program).map_err(|error| error.to_string())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const fn libc_socket() -> i64 {
    41
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const fn libc_socketpair() -> i64 {
    53
}
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const fn libc_socket() -> i64 {
    198
}
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const fn libc_socketpair() -> i64 {
    199
}
