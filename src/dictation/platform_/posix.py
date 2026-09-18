"""macOS, X11 and Wayland adapters (plan section 5).

These are capability probes plus the integrations that can honestly be done
through documented user-space tools.  Where a platform cannot do something, the
adapter says ``unavailable`` or ``unknown`` and the application degrades to tray
controls and manual paste - it does not pretend.

* **macOS** needs microphone authorization *and* Accessibility trust, both of
  which can be denied or revoked at any time.  Probing them properly requires
  the native frameworks, so without a PyObjC bridge the answer is ``unknown``,
  and insertion is not attempted.
* **X11** can inject with ``xdotool`` where it is installed; the security
  context is worth stating plainly, because any X11 client can both inject and
  observe input.
* **Wayland** has no general injection path.  Compositors expose global
  shortcuts and remote-desktop input through portals with user consent, and
  support varies, so capabilities are probed rather than assumed and the
  fallback is manual paste.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from dictation.logging_ import events
from dictation.platform_.base import (
    Capability,
    CapabilityReport,
    CapabilityStatus,
    FocusTarget,
    Support,
    detect_session_type,
    os_details,
)


def _status(available: bool, note_yes: str, note_no: str) -> CapabilityStatus:
    return CapabilityStatus(
        Support.AVAILABLE if available else Support.UNAVAILABLE,
        note_yes if available else note_no,
    )


@dataclass
class MacosAdapter:
    dry_run: bool = True
    inserted: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "macos"

    @staticmethod
    def has_pyobjc() -> bool:
        import importlib.util

        return importlib.util.find_spec("AppKit") is not None

    def probe(self) -> CapabilityReport:
        os_name, os_release = os_details()
        native = sys.platform == "darwin"
        bridge = native and self.has_pyobjc()
        statuses = {
            Capability.GLOBAL_SHORTCUT_PRESS_RELEASE: CapabilityStatus(
                Support.UNKNOWN if native else Support.UNAVAILABLE,
                "needs_cgeventtap_with_accessibility_trust",
            ),
            Capability.GLOBAL_SHORTCUT_TOGGLE: CapabilityStatus(
                Support.UNKNOWN if native else Support.UNAVAILABLE, "needs_carbon_hotkey_or_nsevent"
            ),
            Capability.TEXT_INSERTION_NATIVE: CapabilityStatus(
                Support.REQUIRES_PERMISSION if native else Support.UNAVAILABLE,
                "needs_accessibility_trust",
            ),
            Capability.CLIPBOARD_FALLBACK: _status(
                native and shutil.which("pbcopy") is not None, "pbcopy", "pbcopy_missing"
            ),
            Capability.NONACTIVATING_OVERLAY: CapabilityStatus(
                Support.UNKNOWN if native else Support.UNAVAILABLE, "needs_nonactivating_panel"
            ),
            Capability.TRAY_INDICATOR: CapabilityStatus(
                Support.UNKNOWN if native else Support.UNAVAILABLE, "needs_nsstatusitem"
            ),
            Capability.FOCUS_TARGET_TRACKING: CapabilityStatus(
                Support.UNKNOWN if bridge else Support.UNAVAILABLE,
                "needs_accessibility_api" if native else "not_macos",
            ),
            Capability.MIC_PERMISSION_QUERY: CapabilityStatus(
                Support.UNKNOWN if native else Support.UNAVAILABLE,
                "needs_avcapturedevice_authorization_status",
            ),
            Capability.PASSWORD_FIELD_DETECTION: CapabilityStatus(
                Support.UNKNOWN, "ax_secure_text_field_is_not_universal"
            ),
        }
        notes = (
            "microphone authorization and Accessibility trust are separate and both revocable; "
            "show actionable status rather than failing silently",
            "install the PyObjC bridge, or implement a native helper, before claiming support",
        )
        return CapabilityReport(
            adapter=self.name,
            os_name=os_name,
            os_release=os_release,
            session_type="quartz" if native else detect_session_type(),
            statuses=statuses,
            notes=notes,
        )

    def current_focus(self) -> FocusTarget:
        # Without the Accessibility API there is no trustworthy answer, and a
        # guess would risk inserting text into the wrong window.
        return FocusTarget()

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        if self.dry_run:
            self.inserted.append(text)
            return True
        events.warn("platform.insert_unavailable", adapter=self.name)
        return False

    def copy_to_clipboard(self, text: str) -> bool:
        if self.dry_run:
            self.inserted.append(f"clipboard:{len(text)}")
            return True
        return _run_stdin(["pbcopy"], text)

    def read_clipboard(self) -> str | None:
        return _run_stdout(["pbpaste"])

    def microphone_permission(self) -> Support:
        return Support.UNKNOWN


@dataclass
class X11Adapter:
    dry_run: bool = True
    inserted: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "linux-x11"

    def probe(self) -> CapabilityReport:
        os_name, os_release = os_details()
        on_x11 = bool(os.environ.get("DISPLAY")) and sys.platform.startswith("linux")
        xdotool = shutil.which("xdotool") is not None
        statuses = {
            Capability.GLOBAL_SHORTCUT_PRESS_RELEASE: CapabilityStatus(
                Support.UNKNOWN if on_x11 else Support.UNAVAILABLE, "needs_xgrabkey_or_de_binding"
            ),
            Capability.GLOBAL_SHORTCUT_TOGGLE: CapabilityStatus(
                Support.UNKNOWN if on_x11 else Support.UNAVAILABLE, "needs_xgrabkey_or_de_binding"
            ),
            Capability.TEXT_INSERTION_NATIVE: _status(
                on_x11 and xdotool, "xdotool_type", "xdotool_missing" if on_x11 else "no_display"
            ),
            Capability.CLIPBOARD_FALLBACK: _status(
                on_x11 and (shutil.which("xclip") or shutil.which("xsel")) is not None,
                "xclip_or_xsel",
                "clipboard_tool_missing",
            ),
            Capability.NONACTIVATING_OVERLAY: CapabilityStatus(
                Support.UNKNOWN if on_x11 else Support.UNAVAILABLE, "needs_override_redirect_window"
            ),
            Capability.TRAY_INDICATOR: CapabilityStatus(Support.UNKNOWN, "needs_status_notifier"),
            Capability.FOCUS_TARGET_TRACKING: _status(
                on_x11 and xdotool, "xdotool_getactivewindow", "xdotool_missing"
            ),
            Capability.MIC_PERMISSION_QUERY: CapabilityStatus(
                Support.UNAVAILABLE, "no_per_app_microphone_permission_on_x11"
            ),
            Capability.PASSWORD_FIELD_DETECTION: CapabilityStatus(
                Support.UNAVAILABLE, "not_exposed_by_x11"
            ),
        }
        notes = (
            "on X11 any client can inject and observe input; document this security context",
            "test against the specific desktop environments you intend to support",
        )
        return CapabilityReport(
            adapter=self.name,
            os_name=os_name,
            os_release=os_release,
            session_type="x11",
            statuses=statuses,
            notes=notes,
        )

    def current_focus(self) -> FocusTarget:
        handle = _run_stdout(["xdotool", "getactivewindow"])
        if not handle:
            return FocusTarget()
        name = _run_stdout(["xdotool", "getactivewindow", "getwindowclassname"]) or ""
        return FocusTarget(handle=handle.strip(), application=name.strip(), editable=True)

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        if not target.matches(self.current_focus()):
            events.warn("platform.focus_changed", adapter=self.name)
            return False
        if self.dry_run:
            self.inserted.append(text)
            return True
        return _run(["xdotool", "type", "--clearmodifiers", "--", text])

    def copy_to_clipboard(self, text: str) -> bool:
        if self.dry_run:
            self.inserted.append(f"clipboard:{len(text)}")
            return True
        if shutil.which("xclip"):
            return _run_stdin(["xclip", "-selection", "clipboard"], text)
        if shutil.which("xsel"):
            return _run_stdin(["xsel", "--clipboard", "--input"], text)
        return False

    def read_clipboard(self) -> str | None:
        if shutil.which("xclip"):
            return _run_stdout(["xclip", "-selection", "clipboard", "-o"])
        if shutil.which("xsel"):
            return _run_stdout(["xsel", "--clipboard", "--output"])
        return None

    def microphone_permission(self) -> Support:
        return Support.UNAVAILABLE


@dataclass
class WaylandAdapter:
    dry_run: bool = True
    inserted: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "linux-wayland"

    def probe(self) -> CapabilityReport:
        os_name, os_release = os_details()
        on_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        portal = _portal_available()
        typer = shutil.which("wtype") or shutil.which("ydotool")
        statuses = {
            Capability.GLOBAL_SHORTCUT_PRESS_RELEASE: CapabilityStatus(
                Support.UNKNOWN if portal else Support.UNAVAILABLE,
                "globalshortcuts_portal_does_not_guarantee_release_events",
            ),
            Capability.GLOBAL_SHORTCUT_TOGGLE: CapabilityStatus(
                Support.REQUIRES_PERMISSION if portal else Support.UNAVAILABLE,
                "globalshortcuts_portal" if portal else "portal_unavailable",
            ),
            Capability.TEXT_INSERTION_NATIVE: CapabilityStatus(
                Support.REQUIRES_PERMISSION if (on_wayland and typer) else Support.UNAVAILABLE,
                "remotedesktop_portal_or_virtual_keyboard" if typer else "no_injection_path",
            ),
            Capability.CLIPBOARD_FALLBACK: _status(
                bool(shutil.which("wl-copy")), "wl_clipboard", "wl_clipboard_missing"
            ),
            Capability.NONACTIVATING_OVERLAY: CapabilityStatus(
                Support.UNKNOWN if on_wayland else Support.UNAVAILABLE, "needs_layer_shell"
            ),
            Capability.TRAY_INDICATOR: CapabilityStatus(Support.UNKNOWN, "needs_status_notifier"),
            Capability.FOCUS_TARGET_TRACKING: CapabilityStatus(
                Support.UNAVAILABLE, "compositors_do_not_expose_focus_to_clients"
            ),
            Capability.MIC_PERMISSION_QUERY: CapabilityStatus(
                Support.UNKNOWN, "depends_on_sandbox_and_portal"
            ),
            Capability.PASSWORD_FIELD_DETECTION: CapabilityStatus(
                Support.UNAVAILABLE, "not_exposed_to_clients"
            ),
        }
        notes = (
            "focus tracking is generally unavailable, so insertion cannot be verified against "
            "the original target; prefer the result panel and manual paste",
            "probe capabilities per compositor; support varies between GNOME and KDE sessions",
        )
        return CapabilityReport(
            adapter=self.name,
            os_name=os_name,
            os_release=os_release,
            session_type="wayland",
            statuses=statuses,
            notes=notes,
        )

    def current_focus(self) -> FocusTarget:
        return FocusTarget()

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        if self.dry_run:
            self.inserted.append(text)
            return True
        # Without focus tracking there is no way to confirm the target, so
        # injection is refused and the text stays in the result panel.
        events.warn("platform.insert_unavailable", adapter=self.name)
        return False

    def copy_to_clipboard(self, text: str) -> bool:
        if self.dry_run:
            self.inserted.append(f"clipboard:{len(text)}")
            return True
        return _run_stdin(["wl-copy"], text)

    def read_clipboard(self) -> str | None:
        return _run_stdout(["wl-paste", "--no-newline"])

    def microphone_permission(self) -> Support:
        return Support.UNKNOWN


@dataclass
class NullAdapter:
    """Headless fallback: everything unavailable, nothing pretended."""

    inserted: list[str] = field(default_factory=list)
    dry_run: bool = True

    @property
    def name(self) -> str:
        return "null"

    def probe(self) -> CapabilityReport:
        os_name, os_release = os_details()
        return CapabilityReport(
            adapter=self.name,
            os_name=os_name,
            os_release=os_release,
            session_type=detect_session_type(),
            statuses={
                capability: CapabilityStatus(Support.UNAVAILABLE, "headless")
                for capability in Capability
            },
            notes=("no windowing system detected; dictation is usable through the CLI only",),
        )

    def current_focus(self) -> FocusTarget:
        return FocusTarget()

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        self.inserted.append(text)
        return False

    def copy_to_clipboard(self, text: str) -> bool:
        self.inserted.append(f"clipboard:{len(text)}")
        return False

    def read_clipboard(self) -> str | None:
        return None

    def microphone_permission(self) -> Support:
        return Support.UNAVAILABLE


def _portal_available() -> bool:
    """Look for a desktop portal on the session bus."""
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return False
    return bool(shutil.which("busctl") or shutil.which("dbus-send") or shutil.which("gdbus"))


def _run(command: list[str]) -> bool:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _run_stdin(command: list[str], text: str) -> bool:
    try:
        completed = subprocess.run(
            command, input=text.encode("utf-8"), check=False, capture_output=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _run_stdout(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace")
