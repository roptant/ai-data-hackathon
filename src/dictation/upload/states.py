"""Training job states (plan section 9).

    LOCAL_PENDING -> ANALYZING -> BUILDING -> ELIGIBLE -> UPLOADING -> ACKNOWLEDGED
           |             |           |          |           |
           +-------------+-----------+----------+-----------+-> DELETED / REJECTED

The transition table is data so that both the queue and its tests read the same
rules, and so an unlisted transition is impossible rather than merely unusual.
``UPLOADING -> ELIGIBLE`` is the retry edge; every state can end in ``DELETED``,
which is what withdrawal uses.
"""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    LOCAL_PENDING = "local_pending"
    ANALYZING = "analyzing"
    BUILDING = "building"
    ELIGIBLE = "eligible"
    UPLOADING = "uploading"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    DELETED = "deleted"


TERMINAL_STATES = frozenset({JobState.ACKNOWLEDGED, JobState.REJECTED, JobState.DELETED})

#: States from which withdrawal must cancel work.
CANCELLABLE_STATES = frozenset(
    {
        JobState.LOCAL_PENDING,
        JobState.ANALYZING,
        JobState.BUILDING,
        JobState.ELIGIBLE,
        JobState.UPLOADING,
    }
)

ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.LOCAL_PENDING: frozenset({JobState.ANALYZING, JobState.REJECTED, JobState.DELETED}),
    JobState.ANALYZING: frozenset({JobState.BUILDING, JobState.REJECTED, JobState.DELETED}),
    JobState.BUILDING: frozenset({JobState.ELIGIBLE, JobState.REJECTED, JobState.DELETED}),
    JobState.ELIGIBLE: frozenset({JobState.UPLOADING, JobState.REJECTED, JobState.DELETED}),
    JobState.UPLOADING: frozenset(
        {JobState.ACKNOWLEDGED, JobState.ELIGIBLE, JobState.REJECTED, JobState.DELETED}
    ),
    JobState.ACKNOWLEDGED: frozenset({JobState.DELETED}),
    JobState.REJECTED: frozenset({JobState.DELETED}),
    JobState.DELETED: frozenset(),
}


def may_transition(current: JobState, target: JobState) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def predecessors(target: JobState) -> frozenset[JobState]:
    """States a guarded update may accept when moving to ``target``."""
    return frozenset(
        state for state, allowed in ALLOWED_TRANSITIONS.items() if target in allowed
    )
