"""Local dictation with privacy-filtered personalization.

Reference implementation of ``IMPLEMENTATION_PLAN.md``.  Module layout mirrors
the plan's module table (plan section 3); see ``docs/ARCHITECTURE.md`` for the
mapping and ``docs/DEVIATIONS.md`` for what is deliberately not implemented.

Nothing in this package establishes that its privacy behaviour has been
validated.  Automatic upload stays disabled until the gates in plan section 12
are measured on real data; see :mod:`dictation.consent` and
:data:`dictation.version.UPLOAD_GATES_MET`.
"""

from dictation.version import APP_VERSION, POLICY_VERSION

__all__ = ["APP_VERSION", "POLICY_VERSION"]
