//! Native capability probes and microphone capture.

pub mod microphone;

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
