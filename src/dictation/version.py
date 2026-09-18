"""Versions recorded alongside every artifact (plan sections 7.1, 9, 10)."""

from __future__ import annotations

APP_VERSION = "0.1.0"

#: Version of the privacy detection policy: rule set, categories, sentence
#: expansion behaviour and padding.  Bump on any change that could alter which
#: audio is removed, so datasets stay attributable to a policy revision.
POLICY_VERSION = "policy-2026.09.1"

#: Version of the eligibility contract between client and server (plan 9).
#: The server admits only packages whose eligibility version it knows.
ELIGIBILITY_VERSION = 1

#: Version of the consent text presented before opt-in (plan 9).  Consent
#: recorded under an older version does not authorise uploads under a newer one.
CONSENT_VERSION = "consent-2026.09.1"

#: Local API contract version (plan 8).
API_VERSION = 1

#: Languages whose recordings may become eligible for automatic upload.
#: English only until other languages get their own privacy evaluation
#: (plan section 1, provisional).
VALIDATED_UPLOAD_LANGUAGES = frozenset({"en"})

#: Whether the evaluation gates of plan section 12 have been measured and met.
#: This stays ``False`` in the reference implementation: no recall benchmark has
#: been run.  Automatic upload refuses to enable while it is ``False`` unless an
#: operator explicitly overrides it for a non-production server.
UPLOAD_GATES_MET = False
