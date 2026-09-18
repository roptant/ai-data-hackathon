"""Adapter selection and the capability-matrix artifact (plan milestone 1).

Milestone 1 exits with "a documented platform matrix and resource budget".
This module produces the matrix half: it picks the adapter for the current
session type, probes it, and writes both JSON and Markdown so the result is a
file that can be reviewed and published rather than a claim in a README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dictation.platform_.base import Capability, CapabilityReport, PlatformAdapter, detect_session_type
from dictation.platform_.posix import MacosAdapter, NullAdapter, WaylandAdapter, X11Adapter
from dictation.platform_.windows import WindowsAdapter

ADAPTERS: dict[str, type] = {
    "windows": WindowsAdapter,
    "macos": MacosAdapter,
    "linux-x11": X11Adapter,
    "linux-wayland": WaylandAdapter,
    "null": NullAdapter,
}


def adapter_name_for_session(session_type: str | None = None) -> str:
    session = session_type or detect_session_type()
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if session == "wayland":
        return "linux-wayland"
    if session == "x11":
        return "linux-x11"
    return "null"


def select_adapter(name: str | None = None, *, dry_run: bool = True) -> PlatformAdapter:
    """Instantiate the adapter for this machine, or a named one."""
    chosen = name or adapter_name_for_session()
    factory = ADAPTERS.get(chosen, NullAdapter)
    return factory(dry_run=dry_run)  # type: ignore[call-arg]


def probe_all() -> dict[str, CapabilityReport]:
    """Probe every adapter.

    Running all of them on one machine is still useful: the off-platform
    reports come back ``unavailable`` with the reason, which is what the
    published matrix should say until that platform has been tested on real
    hardware.
    """
    return {name: factory().probe() for name, factory in ADAPTERS.items()}  # type: ignore[call-arg]


def write_matrix(report: CapabilityReport, directory: Path) -> tuple[Path, Path]:
    """Write ``capability-matrix.json`` and ``.md`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "capability-matrix.json"
    markdown_path = directory / "capability-matrix.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, markdown_path


def summarize(report: CapabilityReport) -> str:
    """One-screen summary for the CLI."""
    lines = [
        f"adapter: {report.adapter}",
        f"os: {report.os_name} {report.os_release}",
        f"session: {report.session_type}",
        "",
    ]
    width = max(len(str(capability)) for capability in Capability)
    for capability in Capability:
        status = report.status(capability)
        note = f"  ({status.note})" if status.note else ""
        lines.append(f"  {str(capability):<{width}}  {status.support}{note}")
    if report.notes:
        lines.append("")
        lines.extend(f"  note: {note}" for note in report.notes)
    return "\n".join(lines)
