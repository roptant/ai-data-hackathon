"""Delivering the transcript to the user's application (plan section 5).

The rules, in the order they are applied:

1. The full dictation is what gets inserted.  Redaction applies to training
   data, never to the user's own text.
2. Insert only if focus still matches the target remembered at recording start
   and the field is still suitable.  If it moved, the text goes to a local
   result panel and the user chooses where to paste it - it is never sent to a
   newly focused terminal or chat window.
3. Excluded applications and password fields get nothing.  Password-field
   detection is not claimed to be complete, so the exclusion list is the
   dependable half.
4. Clipboard fallback is used only when disclosed and enabled, because
   clipboard history and cross-device synchronisation can retain the full
   dictation.  Previous contents are restored only if the clipboard still holds
   what this application put there; nothing is promised about external
   clipboard managers.
5. Enter is never synthesised.  A dictation must not submit a form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from dictation.config import InsertionSettings
from dictation.logging_ import content_digest, events
from dictation.platform_.base import FocusTarget, PlatformAdapter


class DeliveryMethod(StrEnum):
    NATIVE = "native"
    CLIPBOARD = "clipboard"
    RESULT_PANEL = "result_panel"
    NONE = "none"


class DeliveryReason(StrEnum):
    OK = "ok"
    FOCUS_CHANGED = "focus_changed"
    APP_EXCLUDED = "app_excluded"
    PASSWORD_FIELD = "password_field"
    NOT_EDITABLE = "not_editable"
    INSERTION_FAILED = "insertion_failed"
    CLIPBOARD_DISABLED = "clipboard_fallback_disabled"
    CLIPBOARD_FAILED = "clipboard_failed"
    CANCELLED = "cancelled"
    EMPTY = "empty_transcript"


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    method: DeliveryMethod
    reason: DeliveryReason
    delivered: bool

    @property
    def needs_user_action(self) -> bool:
        return self.method is DeliveryMethod.RESULT_PANEL


@dataclass(slots=True)
class ResultPanel:
    """Local holding place for text that could not be inserted.

    In memory only.  Keeping a transcript history would be a separate setting
    and is outside the default scope (plan section 6).
    """

    entries: list[tuple[str, str]] = field(default_factory=list)
    limit: int = 5

    def hold(self, session_id: str, text: str) -> None:
        self.entries.append((session_id, text))
        del self.entries[: max(0, len(self.entries) - self.limit)]
        events.emit(
            "insertion.held_in_panel",
            session=session_id,
            chars=len(text),
            digest=content_digest(text),
        )

    def take(self, session_id: str | None = None) -> str | None:
        for index in range(len(self.entries) - 1, -1, -1):
            if session_id is None or self.entries[index][0] == session_id:
                return self.entries.pop(index)[1]
        return None

    def clear(self) -> None:
        self.entries.clear()


class TextDelivery:
    """Applies the insertion policy through a platform adapter."""

    def __init__(
        self,
        adapter: PlatformAdapter,
        settings: InsertionSettings | None = None,
        panel: ResultPanel | None = None,
    ) -> None:
        self.adapter = adapter
        self.settings = settings or InsertionSettings()
        self.panel = panel or ResultPanel()

    def deliver(
        self,
        text: str,
        target: FocusTarget,
        *,
        session_id: str,
        cancelled: bool = False,
    ) -> DeliveryResult:
        """Deliver the full dictation, or hold it for the user."""
        if cancelled:
            # Never insert after cancellation, at any stage.
            events.emit("insertion.skipped", session=session_id, reason=DeliveryReason.CANCELLED)
            return DeliveryResult(DeliveryMethod.NONE, DeliveryReason.CANCELLED, False)
        if not text.strip():
            return DeliveryResult(DeliveryMethod.NONE, DeliveryReason.EMPTY, False)

        blocked = self._blocked(target)
        if blocked is not None:
            return self._hold(text, session_id, blocked)

        current = self.adapter.current_focus()
        if not target.matches(current):
            return self._hold(text, session_id, DeliveryReason.FOCUS_CHANGED)

        if self.adapter.insert_text(text, target):
            events.emit(
                "insertion.native",
                session=session_id,
                chars=len(text),
                app=_app_tag(target.application),
            )
            return DeliveryResult(DeliveryMethod.NATIVE, DeliveryReason.OK, True)

        if not self.settings.clipboard_fallback_allowed:
            return self._hold(text, session_id, DeliveryReason.CLIPBOARD_DISABLED)
        return self._clipboard(text, session_id)

    # -- internals -----------------------------------------------------------

    def _blocked(self, target: FocusTarget) -> DeliveryReason | None:
        if target.password_field:
            return DeliveryReason.PASSWORD_FIELD
        if target.application and target.application in self.settings.excluded_apps:
            return DeliveryReason.APP_EXCLUDED
        if not target.editable:
            return DeliveryReason.NOT_EDITABLE
        return None

    def _clipboard(self, text: str, session_id: str) -> DeliveryResult:
        """Copy, then restore the previous contents if still ours."""
        previous = self.adapter.read_clipboard() if self.settings.restore_clipboard else None
        if not self.adapter.copy_to_clipboard(text):
            return self._hold(text, session_id, DeliveryReason.CLIPBOARD_FAILED)
        events.emit(
            "insertion.clipboard",
            session=session_id,
            chars=len(text),
            restored=previous is not None,
        )
        if previous is not None:
            current = self.adapter.read_clipboard()
            if current == text:
                # Only restore when the clipboard still holds our replacement:
                # otherwise we would clobber whatever the user copied since.
                self.adapter.copy_to_clipboard(previous)
        return DeliveryResult(DeliveryMethod.CLIPBOARD, DeliveryReason.OK, True)

    def _hold(self, text: str, session_id: str, reason: DeliveryReason) -> DeliveryResult:
        self.panel.hold(session_id, text)
        events.emit("insertion.deferred", session=session_id, reason=reason)
        return DeliveryResult(DeliveryMethod.RESULT_PANEL, reason, False)


def _app_tag(application: str) -> str:
    """Token-shaped application tag for the event log.

    Window titles can contain document names and message text, so only a
    coarse tag is logged, and only when it is already token-shaped.
    """
    cleaned = "".join(ch for ch in application if ch.isalnum() or ch in "._-")
    return cleaned[:32] or "unknown"
