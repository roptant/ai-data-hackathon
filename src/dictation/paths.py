"""Filesystem locations (plan section 6).

Payloads live in the OS application-data directory, never in a shared temporary
directory and never in the source repository.  The logical subdirectories are
the ones named in the plan: ``models/``, ``sessions/``, ``queue/`` and
``metadata/``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "LocalDictation"

#: Marker file read by the CLI to warn when a payload root sits inside a git
#: checkout, which would be a data-leak footgun during development.
REPO_MARKERS = (".git", "pyproject.toml")


def default_data_root() -> Path:
    """Per-user application-data directory for this platform."""
    override = os.environ.get("DICTATION_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / APP_DIR_NAME


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Resolved payload locations.  Create with :meth:`create`."""

    root: Path

    @classmethod
    def create(cls, root: Path | None = None) -> DataPaths:
        paths = cls(Path(root) if root is not None else default_data_root())
        paths.ensure()
        return paths

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def queue(self) -> Path:
        return self.root / "queue"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def database(self) -> Path:
        return self.metadata / "state.sqlite3"

    @property
    def settings_file(self) -> Path:
        return self.metadata / "settings.json"

    @property
    def capability_matrix(self) -> Path:
        return self.metadata / "capability-matrix.json"

    def session_dir(self, session_id: str) -> Path:
        _check_id(session_id)
        return self.sessions / session_id

    def queue_dir(self, job_id: str) -> Path:
        _check_id(job_id)
        return self.queue / job_id

    def all_dirs(self) -> tuple[Path, ...]:
        return (self.root, self.models, self.sessions, self.queue, self.metadata)

    def ensure(self) -> None:
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
            _restrict_permissions(directory)
        self._mark_no_backup()

    def inside_repository(self) -> bool:
        """True when the payload root is inside a source checkout."""
        for parent in [self.root, *self.root.parents]:
            if any((parent / marker).exists() for marker in REPO_MARKERS):
                return True
        return False

    def _mark_no_backup(self) -> None:
        """Best-effort exclusion from backups and search indexing.

        This is advisory only.  It does not cover database journals, crash
        dumps, swap or third-party backup tools, all of which stay in the threat
        model (plan section 6).
        """
        tag = self.root / "CACHEDIR.TAG"
        if not tag.exists():
            try:
                tag.write_text(
                    "Signature: 8a477f597d28d172789f06886806bc55\n"
                    "# Encrypted dictation payloads. Excluded from backups by request;\n"
                    "# exclusion is advisory and not enforced by this application.\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        if sys.platform == "darwin":
            marker = self.root / ".metadata_never_index"
            try:
                marker.touch(exist_ok=True)
            except OSError:
                pass


def _check_id(value: str) -> None:
    """Reject anything that could escape the payload root."""
    if not value or "/" in value or "\\" in value or value.startswith("."):
        raise ValueError(f"unsafe path component: {value!r}")


def _restrict_permissions(path: Path) -> None:
    """Narrow POSIX permissions to the owner.  A no-op on Windows, where the
    per-user application-data ACL already applies."""
    if sys.platform == "win32":
        return
    try:
        path.chmod(0o700)
    except OSError:
        pass
