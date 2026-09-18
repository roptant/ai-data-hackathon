"""Privacy-classifier output contract and its independent validator.

Plan sections 4 and 7.4.  Grammar-constrained decoding improves structural
reliability but proves nothing about correctness, so the output is validated
here against the *frozen transcript* rather than trusted:

* only known keys, types and enum values,
* every word ID must exist in the window that was actually shown,
* spans must be non-empty, in bounds, ordered and non-absurd in count,
* ``drop_words`` is rejected unless word-level cuts are explicitly enabled,
* the empty-span case still requires a complete, valid response - a missing or
  truncated answer is a failure, never "nothing sensitive found".
"""

from __future__ import annotations

import json
from typing import Any, Collection, Iterable

from dictation.errors import SchemaViolation
from dictation.types import DetectionSource, RemovalAction, SensitiveCategory, Span

#: Guard against a degenerate answer that marks hundreds of tiny spans.
MAX_SPANS_PER_WINDOW = 64

CATEGORY_VALUES = tuple(str(category) for category in SensitiveCategory)
ACTION_VALUES = tuple(str(action) for action in RemovalAction)

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["spans", "uncertain"],
    "properties": {
        "spans": {
            "type": "array",
            "maxItems": MAX_SPANS_PER_WINDOW,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start_word_id", "end_word_id_exclusive", "category", "action"],
                "properties": {
                    "start_word_id": {"type": "integer", "minimum": 0},
                    "end_word_id_exclusive": {"type": "integer", "minimum": 1},
                    "category": {"type": "string", "enum": list(CATEGORY_VALUES)},
                    "action": {"type": "string", "enum": list(ACTION_VALUES)},
                },
            },
        },
        "uncertain": {"type": "boolean"},
    },
}

#: GBNF grammar for llama.cpp, kept in sync with JSON_SCHEMA above.  It
#: constrains structure only; :func:`validate_output` still runs on the result.
GBNF_GRAMMAR = r"""
root   ::= "{" ws "\"spans\"" ws ":" ws spans ws "," ws "\"uncertain\"" ws ":" ws bool ws "}"
spans  ::= "[" ws "]" | "[" ws span (ws "," ws span)* ws "]"
span   ::= "{" ws "\"start_word_id\"" ws ":" ws int ws ","
           ws "\"end_word_id_exclusive\"" ws ":" ws int ws ","
           ws "\"category\"" ws ":" ws category ws ","
           ws "\"action\"" ws ":" ws action ws "}"
category ::= "\"direct_identifier\"" | "\"contact\"" | "\"address\"" |
             "\"account_identifier\"" | "\"government_id\"" | "\"financial\"" |
             "\"credential_secret\"" | "\"customer_confidential\"" | "\"health\"" |
             "\"political_religious\"" | "\"sexuality\"" | "\"sensitive_narrative\"" |
             "\"contextual_combination\"" | "\"user_defined\""
action   ::= "\"drop_sentence\"" | "\"drop_words\""
bool     ::= "true" | "false"
int      ::= [0-9] | [1-9] [0-9]*
ws       ::= [ \t\n]*
"""


def parse_output(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Decode the worker's answer.

    A parse failure is a worker failure.  It must not be smoothed over: the
    caller rejects the session rather than treating unparsed text as clean.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    text = raw.strip()
    if not text:
        raise SchemaViolation("classifier returned an empty response")
    # Tolerate a single fenced block, which small instruct models often emit.
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise SchemaViolation(f"classifier output is not valid JSON: {error.msg}") from None
    if not isinstance(parsed, dict):
        raise SchemaViolation("classifier output must be a JSON object")
    return parsed


def validate_output(
    raw: str | bytes | dict[str, Any],
    *,
    allowed_word_ids: Collection[int],
    allow_word_level_cuts: bool = False,
    max_spans: int = MAX_SPANS_PER_WINDOW,
) -> tuple[tuple[Span, ...], bool]:
    """Validate one window's answer and convert it to spans.

    Returns ``(spans, uncertain)``.  Raises
    :class:`~dictation.errors.SchemaViolation` on anything unexpected.
    """
    payload = parse_output(raw)
    unknown = set(payload) - {"spans", "uncertain"}
    if unknown:
        raise SchemaViolation(f"unexpected keys in classifier output: {sorted(unknown)}")
    if "spans" not in payload or "uncertain" not in payload:
        raise SchemaViolation("classifier output must contain both 'spans' and 'uncertain'")

    uncertain = payload["uncertain"]
    if not isinstance(uncertain, bool):
        raise SchemaViolation("'uncertain' must be a boolean")

    items = payload["spans"]
    if not isinstance(items, list):
        raise SchemaViolation("'spans' must be an array")
    if len(items) > max_spans:
        raise SchemaViolation(f"too many spans: {len(items)} > {max_spans}")

    ids = set(allowed_word_ids)
    if not ids and items:
        raise SchemaViolation("classifier reported spans for an empty window")

    lowest, highest = (min(ids), max(ids)) if ids else (0, -1)
    spans: list[Span] = []
    previous_start = -1
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SchemaViolation(f"span {index} is not an object")
        unknown_keys = set(item) - {
            "start_word_id",
            "end_word_id_exclusive",
            "category",
            "action",
        }
        if unknown_keys:
            raise SchemaViolation(f"span {index} has unexpected keys: {sorted(unknown_keys)}")
        start = item.get("start_word_id")
        end = item.get("end_word_id_exclusive")
        if not isinstance(start, int) or isinstance(start, bool):
            raise SchemaViolation(f"span {index}: start_word_id must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise SchemaViolation(f"span {index}: end_word_id_exclusive must be an integer")
        if end <= start:
            raise SchemaViolation(f"span {index}: empty or inverted range [{start}, {end})")
        if start not in ids:
            raise SchemaViolation(f"span {index}: start_word_id {start} was not in the window")
        if end - 1 not in ids:
            raise SchemaViolation(f"span {index}: end_word_id_exclusive {end} was not in the window")
        if start < lowest or end - 1 > highest:
            raise SchemaViolation(f"span {index}: range [{start}, {end}) is outside the window")
        if start < previous_start:
            raise SchemaViolation(f"span {index}: spans must be ordered by start_word_id")
        previous_start = start

        category_value = item.get("category")
        if category_value not in CATEGORY_VALUES:
            raise SchemaViolation(f"span {index}: unknown category {category_value!r}")
        action_value = item.get("action")
        if action_value not in ACTION_VALUES:
            raise SchemaViolation(f"span {index}: unknown action {action_value!r}")
        action = RemovalAction(action_value)
        if action is RemovalAction.DROP_WORDS and not allow_word_level_cuts:
            # Word-level cuts need validated alignment; fall back to the safer
            # action rather than honouring a finer cut than we can prove.
            action = RemovalAction.DROP_SENTENCE

        spans.append(
            Span(
                start_word_id=start,
                end_word_id_exclusive=end,
                category=SensitiveCategory(category_value),
                action=action,
                source=DetectionSource.MODEL,
                detector="privacy_model",
            )
        )
    return tuple(spans), uncertain


def spans_to_payload(spans: Iterable[Span], uncertain: bool = False) -> dict[str, Any]:
    """Serialise spans in the wire shape, used by fixtures and tests."""
    return {
        "spans": [
            {
                "start_word_id": span.start_word_id,
                "end_word_id_exclusive": span.end_word_id_exclusive,
                "category": str(span.category),
                "action": str(span.action),
            }
            for span in spans
        ],
        "uncertain": uncertain,
    }
