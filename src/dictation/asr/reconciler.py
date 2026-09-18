"""Reconciliation of overlapping partial hypotheses (plan sections 6 and 8).

Partials arrive from overlapping audio windows, so the same segment is emitted
repeatedly with a rising revision and possibly different text.  Clients must not
append partials blindly, and neither may the indicator: the reconciler keeps the
highest revision per segment identity and orders segments by start time.

A stale revision is dropped rather than applied.  Out-of-order delivery is
normal on a loaded machine, and applying an old hypothesis would make text
visibly regress.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dictation.types import Segment


@dataclass(slots=True)
class PartialReconciler:
    """Current best hypothesis per segment."""

    _segments: dict[str, Segment] = field(default_factory=dict)
    #: Monotonic sequence number handed to API clients so they can detect gaps.
    _seq: int = 0

    def apply(self, segment: Segment) -> bool:
        """Apply a hypothesis.  Returns False when it was stale or redundant."""
        existing = self._segments.get(segment.id)
        if existing is not None:
            if segment.revision < existing.revision:
                return False
            if segment.revision == existing.revision and segment.text == existing.text:
                return False
            if existing.is_final and not segment.is_final:
                # A final segment is not revised by a later partial.
                return False
        self._segments[segment.id] = segment
        self._seq += 1
        return True

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def seq(self) -> int:
        return self._seq

    def ordered(self) -> tuple[Segment, ...]:
        return tuple(sorted(self._segments.values(), key=lambda s: (s.start_sample, s.id)))

    def revision_of(self, segment_id: str) -> int:
        segment = self._segments.get(segment_id)
        return segment.revision if segment else -1

    def text(self) -> str:
        """Current display text: segments joined in time order."""
        return " ".join(segment.text.strip() for segment in self.ordered() if segment.text.strip())

    def reset(self) -> None:
        self._segments.clear()
        self._seq = 0
