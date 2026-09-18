"""Independent validation of classifier output (plan section 7.4)."""

from __future__ import annotations

import pytest

from dictation.errors import SchemaViolation
from dictation.privacy.schema import (
    GBNF_GRAMMAR,
    JSON_SCHEMA,
    MAX_SPANS_PER_WINDOW,
    parse_output,
    spans_to_payload,
    validate_output,
)
from dictation.types import DetectionSource, RemovalAction, SensitiveCategory, Span

IDS = range(10, 20)


def validate(payload: object, **kwargs):
    return validate_output(payload, allowed_word_ids=IDS, **kwargs)


def span_payload(start: int, end: int, category: str = "contact", action: str = "drop_sentence"):
    return {
        "spans": [
            {
                "start_word_id": start,
                "end_word_id_exclusive": end,
                "category": category,
                "action": action,
            }
        ],
        "uncertain": False,
    }


# -- accepted ----------------------------------------------------------------


def test_empty_span_list_is_a_complete_answer() -> None:
    spans, uncertain = validate({"spans": [], "uncertain": False})
    assert spans == ()
    assert uncertain is False


def test_valid_span_is_converted() -> None:
    spans, uncertain = validate(span_payload(12, 17))
    assert len(spans) == 1
    assert spans[0].start_word_id == 12
    assert spans[0].end_word_id_exclusive == 17
    assert spans[0].category is SensitiveCategory.CONTACT
    assert spans[0].source is DetectionSource.MODEL
    assert uncertain is False


def test_json_string_is_parsed() -> None:
    spans, _ = validate('{"spans": [], "uncertain": true}')
    assert spans == ()


def test_fenced_json_block_is_tolerated() -> None:
    raw = '```json\n{"spans": [], "uncertain": false}\n```'
    spans, uncertain = validate(raw)
    assert spans == () and uncertain is False


def test_uncertain_true_is_carried_through() -> None:
    _, uncertain = validate({"spans": [], "uncertain": True})
    assert uncertain is True


def test_word_level_action_is_downgraded_unless_enabled() -> None:
    spans, _ = validate(span_payload(12, 14, action="drop_words"))
    assert spans[0].action is RemovalAction.DROP_SENTENCE

    spans, _ = validate(span_payload(12, 14, action="drop_words"), allow_word_level_cuts=True)
    assert spans[0].action is RemovalAction.DROP_WORDS


def test_round_trip_through_spans_to_payload() -> None:
    original = (
        Span(12, 15, SensitiveCategory.ADDRESS, RemovalAction.DROP_SENTENCE),
        Span(16, 18, SensitiveCategory.HEALTH, RemovalAction.DROP_SENTENCE),
    )
    spans, uncertain = validate(spans_to_payload(original))
    assert [(s.start_word_id, s.end_word_id_exclusive) for s in spans] == [(12, 15), (16, 18)]
    assert uncertain is False


# -- rejected ----------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "I found nothing sensitive.",
        '{"spans": [',
        "[]",
        '{"spans": []}',
        '{"uncertain": false}',
        '{"spans": [], "uncertain": "no"}',
        '{"spans": {}, "uncertain": false}',
        '{"spans": [], "uncertain": false, "extra": 1}',
    ],
)
def test_structurally_invalid_output_is_refused(payload: str) -> None:
    with pytest.raises(SchemaViolation):
        validate(payload)


@pytest.mark.parametrize(
    "start,end",
    [(9, 12), (18, 25), (100, 101), (12, 12), (15, 13)],
)
def test_out_of_window_or_empty_ranges_are_refused(start: int, end: int) -> None:
    with pytest.raises(SchemaViolation):
        validate(span_payload(start, end))


def test_unknown_category_is_refused() -> None:
    with pytest.raises(SchemaViolation):
        validate(span_payload(12, 14, category="vibes"))


def test_unknown_action_is_refused() -> None:
    with pytest.raises(SchemaViolation):
        validate(span_payload(12, 14, action="paraphrase"))


def test_extra_span_keys_are_refused() -> None:
    payload = span_payload(12, 14)
    payload["spans"][0]["replacement_text"] = "[REDACTED]"
    with pytest.raises(SchemaViolation):
        validate(payload)


def test_non_integer_ids_are_refused() -> None:
    payload = span_payload(12, 14)
    payload["spans"][0]["start_word_id"] = "12"
    with pytest.raises(SchemaViolation):
        validate(payload)


def test_boolean_is_not_accepted_as_an_id() -> None:
    payload = span_payload(12, 14)
    payload["spans"][0]["start_word_id"] = True
    with pytest.raises(SchemaViolation):
        validate(payload)


def test_unordered_spans_are_refused() -> None:
    payload = {
        "spans": [
            {
                "start_word_id": 16,
                "end_word_id_exclusive": 18,
                "category": "contact",
                "action": "drop_sentence",
            },
            {
                "start_word_id": 12,
                "end_word_id_exclusive": 14,
                "category": "contact",
                "action": "drop_sentence",
            },
        ],
        "uncertain": False,
    }
    with pytest.raises(SchemaViolation):
        validate(payload)


def test_absurd_span_count_is_refused() -> None:
    payload = {
        "spans": [
            {
                "start_word_id": 10,
                "end_word_id_exclusive": 11,
                "category": "contact",
                "action": "drop_sentence",
            }
        ]
        * (MAX_SPANS_PER_WINDOW + 1),
        "uncertain": False,
    }
    with pytest.raises(SchemaViolation):
        validate(payload)


def test_spans_for_an_empty_window_are_refused() -> None:
    with pytest.raises(SchemaViolation):
        validate_output(span_payload(12, 14), allowed_word_ids=())


# -- contract shape ----------------------------------------------------------


def test_schema_and_grammar_agree_on_enums() -> None:
    categories = JSON_SCHEMA["properties"]["spans"]["items"]["properties"]["category"]["enum"]
    actions = JSON_SCHEMA["properties"]["spans"]["items"]["properties"]["action"]["enum"]
    for value in [*categories, *actions]:
        # The grammar spells each literal as an escaped JSON string.
        assert f'\\"{value}\\"' in GBNF_GRAMMAR


def test_schema_forbids_additional_properties() -> None:
    assert JSON_SCHEMA["additionalProperties"] is False
    assert JSON_SCHEMA["properties"]["spans"]["items"]["additionalProperties"] is False


def test_parse_output_accepts_a_dict_unchanged() -> None:
    assert parse_output({"spans": [], "uncertain": False}) == {"spans": [], "uncertain": False}
