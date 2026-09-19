//! Platform capability and focus-safe insertion policy.
//!
//! Native adapters report observed capabilities through this contract. Missing
//! support remains explicit; the policy never silently inserts into a newly
//! focused target and never requests an Enter/submit action.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Available,
    PermissionRequired,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformCapabilities {
    pub microphone: Capability,
    pub global_shortcut: Capability,
    pub focus_tracking: Capability,
    pub native_insertion: Capability,
    pub clipboard: Capability,
    pub nonactivating_indicator: Capability,
}

impl Default for PlatformCapabilities {
    fn default() -> Self {
        Self {
            microphone: Capability::Unavailable,
            global_shortcut: Capability::Unavailable,
            focus_tracking: Capability::Unavailable,
            native_insertion: Capability::Unavailable,
            clipboard: Capability::Unavailable,
            nonactivating_indicator: Capability::Unavailable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FocusTarget {
    pub application_id: String,
    pub window_id: String,
    pub editable_id: Option<String>,
    pub protected: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InsertionPolicy {
    pub excluded_applications: BTreeSet<String>,
    pub clipboard_fallback_disclosed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMethod {
    NativeInsertion,
    ClipboardFallback,
    ResultPanel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryReason {
    FocusChanged,
    ProtectedField,
    ApplicationExcluded,
    NativeInsertionAvailable,
    DisclosedClipboardFallback,
    NoSafeAutomaticMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveryDecision {
    pub method: DeliveryMethod,
    pub reason: DeliveryReason,
}

#[must_use]
pub fn choose_delivery(
    original: &FocusTarget,
    current: &FocusTarget,
    capabilities: PlatformCapabilities,
    policy: &InsertionPolicy,
) -> DeliveryDecision {
    if original != current {
        return manual(DeliveryReason::FocusChanged);
    }
    if current.protected {
        return manual(DeliveryReason::ProtectedField);
    }
    if policy
        .excluded_applications
        .contains(&current.application_id)
    {
        return manual(DeliveryReason::ApplicationExcluded);
    }
    if capabilities.native_insertion == Capability::Available {
        return DeliveryDecision {
            method: DeliveryMethod::NativeInsertion,
            reason: DeliveryReason::NativeInsertionAvailable,
        };
    }
    if capabilities.clipboard == Capability::Available && policy.clipboard_fallback_disclosed {
        return DeliveryDecision {
            method: DeliveryMethod::ClipboardFallback,
            reason: DeliveryReason::DisclosedClipboardFallback,
        };
    }
    manual(DeliveryReason::NoSafeAutomaticMethod)
}

const fn manual(reason: DeliveryReason) -> DeliveryDecision {
    DeliveryDecision {
        method: DeliveryMethod::ResultPanel,
        reason,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleEvent {
    ScreenLocked,
    SystemSleeping,
    InputDeviceRemoved,
    MicrophonePermissionRevoked,
    ProcessShuttingDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleAction {
    CancelWithoutDelivery,
}

#[must_use]
pub const fn lifecycle_action(_event: LifecycleEvent) -> LifecycleAction {
    LifecycleAction::CancelWithoutDelivery
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> FocusTarget {
        FocusTarget {
            application_id: "editor".to_owned(),
            window_id: "window-1".to_owned(),
            editable_id: Some("document".to_owned()),
            protected: false,
        }
    }

    fn capabilities() -> PlatformCapabilities {
        PlatformCapabilities {
            native_insertion: Capability::Available,
            clipboard: Capability::Available,
            ..PlatformCapabilities::default()
        }
    }

    #[test]
    fn changed_focus_never_receives_automatic_text() {
        let original = target();
        let mut current = target();
        current.window_id = "terminal".to_owned();
        assert_eq!(
            choose_delivery(
                &original,
                &current,
                capabilities(),
                &InsertionPolicy::default()
            ),
            manual(DeliveryReason::FocusChanged)
        );
    }

    #[test]
    fn protected_and_excluded_targets_use_the_result_panel() {
        let original = target();
        let mut protected = target();
        protected.protected = true;
        assert_eq!(
            choose_delivery(
                &protected,
                &protected,
                capabilities(),
                &InsertionPolicy::default()
            )
            .reason,
            DeliveryReason::ProtectedField
        );

        let policy = InsertionPolicy {
            excluded_applications: BTreeSet::from(["editor".to_owned()]),
            clipboard_fallback_disclosed: true,
        };
        assert_eq!(
            choose_delivery(&original, &original, capabilities(), &policy).reason,
            DeliveryReason::ApplicationExcluded
        );
    }

    #[test]
    fn clipboard_requires_prior_disclosure() {
        let target = target();
        let capabilities = PlatformCapabilities {
            clipboard: Capability::Available,
            ..PlatformCapabilities::default()
        };
        assert_eq!(
            choose_delivery(&target, &target, capabilities, &InsertionPolicy::default()).method,
            DeliveryMethod::ResultPanel
        );
        let policy = InsertionPolicy {
            clipboard_fallback_disclosed: true,
            ..InsertionPolicy::default()
        };
        assert_eq!(
            choose_delivery(&target, &target, capabilities, &policy).method,
            DeliveryMethod::ClipboardFallback
        );
    }

    #[test]
    fn disruptive_lifecycle_events_cancel_without_delivery() {
        for event in [
            LifecycleEvent::ScreenLocked,
            LifecycleEvent::SystemSleeping,
            LifecycleEvent::InputDeviceRemoved,
            LifecycleEvent::MicrophonePermissionRevoked,
            LifecycleEvent::ProcessShuttingDown,
        ] {
            assert_eq!(
                lifecycle_action(event),
                LifecycleAction::CancelWithoutDelivery
            );
        }
    }
}
