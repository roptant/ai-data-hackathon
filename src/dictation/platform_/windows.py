"""Windows adapter (plan section 5).

Implemented here: capability probing, focus-target tracking through
``GetForegroundWindow``, clipboard access, and Unicode text insertion with
``SendInput`` and ``KEYEVENTF_UNICODE``.

Deliberately reported as ``unknown`` rather than implemented:

* **Press/release global shortcuts.**  ``RegisterHotKey`` delivers a press
  only; hold-to-record needs a low-level keyboard hook with a message loop,
  which belongs in the desktop shell rather than in this core.
* **Password-field detection.**  There is no reliable, general way to know a
  focused control is a protected field, so it is never claimed.

``SendInput`` is subject to User Interface Privilege Isolation: a normal-
integrity process cannot inject into an elevated window.  The adapter reports
that as an insertion failure so the caller falls back to the result panel; it
does not elevate the whole application to work around it.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
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

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1

#: Pause between synthetic keystrokes.  Some applications drop input sent
#: faster than they can process it.
KEYSTROKE_DELAY_S = 0.001


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


@dataclass
class WindowsAdapter:
    """Windows platform integration.

    ``dry_run`` defaults to ``True``: constructing an adapter must never be
    able to type into whatever window happens to be focused.  The desktop shell
    sets it to ``False`` once a real recording delivers text.
    """

    dry_run: bool = True
    inserted: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "windows"

    # -- probing -------------------------------------------------------------

    def probe(self) -> CapabilityReport:
        os_name, os_release = os_details()
        statuses: dict[Capability, CapabilityStatus] = {}
        notes: list[str] = []

        if sys.platform != "win32":
            for capability in Capability:
                statuses[capability] = CapabilityStatus(Support.UNAVAILABLE, "not_windows")
            notes.append("probe ran on a non-Windows host; results are not applicable")
            return CapabilityReport(
                adapter=self.name,
                os_name=os_name,
                os_release=os_release,
                session_type=detect_session_type(),
                statuses=statuses,
                notes=tuple(notes),
            )

        user32 = _user32()
        have_user32 = user32 is not None
        statuses[Capability.TEXT_INSERTION_NATIVE] = CapabilityStatus(
            Support.AVAILABLE if have_user32 else Support.UNAVAILABLE,
            "sendinput_unicode" if have_user32 else "user32_unavailable",
        )
        statuses[Capability.FOCUS_TARGET_TRACKING] = CapabilityStatus(
            Support.AVAILABLE if have_user32 else Support.UNAVAILABLE,
            "getforegroundwindow",
        )
        statuses[Capability.CLIPBOARD_FALLBACK] = CapabilityStatus(
            Support.AVAILABLE, "clip_and_powershell"
        )
        statuses[Capability.GLOBAL_SHORTCUT_TOGGLE] = CapabilityStatus(
            Support.AVAILABLE if have_user32 else Support.UNAVAILABLE, "registerhotkey"
        )
        statuses[Capability.GLOBAL_SHORTCUT_PRESS_RELEASE] = CapabilityStatus(
            Support.UNKNOWN, "needs_low_level_keyboard_hook"
        )
        statuses[Capability.NONACTIVATING_OVERLAY] = CapabilityStatus(
            Support.UNKNOWN, "needs_ws_ex_noactivate_window"
        )
        statuses[Capability.TRAY_INDICATOR] = CapabilityStatus(
            Support.UNKNOWN, "needs_shell_notifyicon"
        )
        statuses[Capability.MIC_PERMISSION_QUERY] = CapabilityStatus(
            Support.UNKNOWN, "needs_capture_attempt_or_settings_api"
        )
        statuses[Capability.PASSWORD_FIELD_DETECTION] = CapabilityStatus(
            Support.UNKNOWN, "no_reliable_general_detection"
        )
        notes.append(
            "SendInput cannot inject into a higher integrity level process; "
            "insertion into an elevated window fails and falls back to the result panel"
        )
        notes.append("record the exact Windows build before publishing a support claim")
        return CapabilityReport(
            adapter=self.name,
            os_name=os_name,
            os_release=os_release,
            session_type="windows",
            statuses=statuses,
            notes=tuple(notes),
        )

    # -- focus ---------------------------------------------------------------

    def current_focus(self) -> FocusTarget:
        user32 = _user32()
        if user32 is None:
            return FocusTarget()
        handle = user32.GetForegroundWindow()
        if not handle:
            return FocusTarget()
        process_id = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
        return FocusTarget(
            handle=str(int(handle)),
            application=f"pid-{process_id.value}",
            editable=True,  # not determinable without UI Automation
            password_field=None,
        )

    # -- insertion -----------------------------------------------------------

    def insert_text(self, text: str, target: FocusTarget) -> bool:
        """Type text into the focused window as Unicode key events."""
        if not text:
            return True
        current = self.current_focus()
        if not target.matches(current):
            events.warn("platform.focus_changed", adapter=self.name)
            return False
        if self.dry_run:
            self.inserted.append(text)
            events.emit("platform.insert_dry_run", adapter=self.name, chars=len(text))
            return True
        user32 = _user32()
        if user32 is None:
            return False
        for character in text:
            for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
                structure = _Input(
                    type=INPUT_KEYBOARD,
                    union=_InputUnion(
                        ki=_KeyboardInput(
                            wVk=0,
                            wScan=ord(character),
                            dwFlags=flags,
                            time=0,
                            dwExtraInfo=None,
                        )
                    ),
                )
                sent = user32.SendInput(1, ctypes.byref(structure), ctypes.sizeof(structure))
                if sent != 1:
                    # Typically UIPI: the target runs at a higher integrity level.
                    events.warn("platform.insert_blocked", adapter=self.name)
                    return False
            time.sleep(KEYSTROKE_DELAY_S)
        return True

    # -- clipboard -----------------------------------------------------------

    def copy_to_clipboard(self, text: str) -> bool:
        import subprocess

        if self.dry_run:
            self.inserted.append(f"clipboard:{len(text)}")
            return True
        try:
            completed = subprocess.run(
                ["clip"], input=text.encode("utf-16-le"), check=False, capture_output=True
            )
        except OSError:
            return False
        return completed.returncode == 0

    def read_clipboard(self) -> str | None:
        import subprocess

        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-Clipboard -Raw",
                ],
                check=False,
                capture_output=True,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.decode("utf-8", errors="replace").rstrip("\r\n")

    def microphone_permission(self) -> Support:
        # Windows exposes per-app microphone settings, but there is no simple
        # query; the honest answer is that it is determined by trying.
        return Support.UNKNOWN


def _user32() -> ctypes.WinDLL | None:  # type: ignore[name-defined]
    if sys.platform != "win32":
        return None
    try:
        return ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
    except OSError:  # pragma: no cover - present on every supported build
        return None
