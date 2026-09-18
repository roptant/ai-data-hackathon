"""Platform capability model (plan sections 5 and 12, milestone 1).

The plan is emphatic that operating systems do not have identical shortcut and
insertion capabilities, and that a cross-platform launch means a *published
capability matrix* rather than a claim of universal injection.  So capabilities
are reported per platform, with three honest answers available -
``available``, ``unavailable``, ``unknown`` - and ``unknown`` is the default
until a probe has actually run on that machine.

Nothing in this module assumes a capability.  A missing capability degrades
visibly: tray controls and manual paste instead of silent failure.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Capability(StrEnum):
    #: Separate press and release events, needed for hold-to-record.
    GLOBAL_SHORTCUT_PRESS_RELEASE = "global_shortcut_press_release"
    #: A simple registered chord, enough for toggle recording.
    GLOBAL_SHORTCUT_TOGGLE = "global_shortcut_toggle"
    #: Injecting text into the focused editable field.
    TEXT_INSERTION_NATIVE = "text_insertion_native"
    #: Clipboard-based paste, which needs explicit disclosure.
    CLIPBOARD_FALLBACK = "clipboard_fallback"
    #: An overlay that shows recording state without taking focus.
    NONACTIVATING_OVERLAY = "nonactivating_overlay"
    TRAY_INDICATOR = "tray_indicator"
    #: Remembering which application and field was focused at start.
    FOCUS_TARGET_TRACKING = "focus_target_tracking"
    #: Querying microphone permission state.
    MIC_PERMISSION_QUERY = "mic_permission_query"
    #: Detecting a protected password field.  Never claimed as complete.
    PASSWORD_FIELD_DETECTION = "password_field_detection"


class Support(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    #: Present but gated behind a permission the user has not granted.
    REQUIRES_PERMISSION = "requires_permission"
    #: Not probed, or not determinable without trying it on real hardware.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    support: Support
    #: Short machine-readable note.  Never free prose in logs.
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.support is Support.AVAILABLE


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Result of probing one machine."""

    adapter: str
    os_name: str
    os_release: str
    session_type: str
    statuses: dict[Capability, CapabilityStatus] = field(default_factory=dict)
    #: Anything the operator must record before release (plan section 1).
    notes: tuple[str, ...] = ()

    def status(self, capability: Capability) -> CapabilityStatus:
        return self.statuses.get(capability, CapabilityStatus(Support.UNKNOWN, "not_probed"))

    def usable(self, capability: Capability) -> bool:
        return self.status(capability).usable

    def missing(self) -> tuple[Capability, ...]:
        return tuple(
            capability
            for capability in Capability
            if not self.status(capability).usable
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "os_name": self.os_name,
            "os_release": self.os_release,
            "session_type": self.session_type,
            "capabilities": {
                str(capability): {
                    "support": str(self.status(capability).support),
                    "note": self.status(capability).note,
                }
                for capability in Capability
            },
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Capability matrix: {self.adapter}",
            "",
            f"- OS: {self.os_name} {self.os_release}",
            f"- Session type: {self.session_type}",
            "",
            "| Capability | Support | Note |",
            "| --- | --- | --- |",
        ]
        for capability in Capability:
            status = self.status(capability)
            lines.append(f"| {capability} | {status.support} | {status.note or ''} |")
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        lines.extend(
            [
                "",
                "Unknown means not determined on this machine. It is not a claim that the",
                "capability works; record measured results before publishing a matrix.",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FocusTarget:
    """The application and field that was focused when recording started."""

    #: Opaque window handle or identifier.  Never logged.
    handle: str = ""
    #: Process or bundle name, used for per-application exclusions.
    application: str = ""
    editable: bool = False
    password_field: bool | None = None

    def matches(self, other: FocusTarget) -> bool:
        """True when insertion may proceed into ``other``.

        An empty handle never matches: not knowing where the text would land is
        a reason to keep it in the result panel (plan section 5).
        """
        if not self.handle or not other.handle:
            return False
        return self.handle == other.handle and self.application == other.application


@runtime_checkable
class PlatformAdapter(Protocol):
    """Shortcuts, permissions, indicator, focus tracking and insertion."""

    @property
    def name(self) -> str: ...

    def probe(self) -> CapabilityReport: ...

    def current_focus(self) -> FocusTarget: ...

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        """Insert into ``target``.  Returns False when it could not."""

    def copy_to_clipboard(self, text: str) -> bool: ...

    def read_clipboard(self) -> str | None: ...

    def microphone_permission(self) -> Support: ...


def detect_session_type() -> str:
    """Windowing system in use, which determines what is possible on Linux."""
    import os

    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "quartz"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    session = os.environ.get("XDG_SESSION_TYPE", "")
    return session or "headless"


def os_details() -> tuple[str, str]:
    return platform.system(), platform.release()
