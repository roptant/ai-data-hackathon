//! Native capability probes, microphone capture, and desktop integration.

pub mod clipboard;
pub mod desktop;
#[cfg(target_os = "linux")]
pub mod linux;
pub mod microphone;

/// Shortcut vocabulary shared with the pure core.
pub mod shortcuts {
    pub use dictation_core::shortcuts::{
        Chord, ShortcutAction, ShortcutBindings, ShortcutEvent, ShortcutInterpreter,
    };
}

/// Which lifecycle watchers are active on this platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LifecycleReport {
    pub sleep: bool,
    pub screen_lock: bool,
}

use cpal::traits::{DeviceTrait, HostTrait};
use dictation_core::platform::{Capability, PlatformCapabilities};

#[must_use]
pub fn probe_capabilities() -> PlatformCapabilities {
    let host = cpal::default_host();
    let microphone = host
        .default_input_device()
        .map_or(Capability::Unavailable, |device| {
            if device.default_input_config().is_ok() {
                Capability::Available
            } else {
                Capability::PermissionRequired
            }
        });
    PlatformCapabilities {
        microphone,
        ..PlatformCapabilities::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unimplemented_capabilities_are_never_claimed() {
        let capabilities = probe_capabilities();
        assert_eq!(capabilities.global_shortcut, Capability::Unavailable);
        assert_eq!(capabilities.focus_tracking, Capability::Unavailable);
        assert_eq!(capabilities.native_insertion, Capability::Unavailable);
        assert_eq!(
            capabilities.nonactivating_indicator,
            Capability::Unavailable
        );
    }
}
