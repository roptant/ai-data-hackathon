"""Interval algebra, destination mapping and resampling (plan sections 6-7)."""

from __future__ import annotations

import pytest

from dictation.time_map import (
    DestinationMap,
    ResampleMap,
    complement,
    contains_interval,
    ms_to_samples,
    normalize,
    pad,
    pad_all,
    samples_to_ms,
    total_length,
    validate_alignment,
)
from dictation.types import SampleInterval, Word


def interval(start: int, end: int) -> SampleInterval:
    return SampleInterval(start, end)


# -- basics ------------------------------------------------------------------


def test_half_open_intervals_do_not_share_boundary_samples() -> None:
    first, second = interval(0, 10), interval(10, 20)
    assert not first.intersects(second)
    assert first.touches(second)
    assert not first.contains(10)
    assert second.contains(10)


def test_inverted_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        SampleInterval(10, 5)


def test_negative_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        SampleInterval(-1, 5)


def test_ms_and_sample_conversion_round_trips() -> None:
    assert ms_to_samples(250, 16_000) == 4_000
    assert samples_to_ms(4_000, 16_000) == 250.0


# -- normalize and merge -----------------------------------------------------


def test_normalize_merges_overlapping_and_abutting() -> None:
    merged = normalize([interval(10, 20), interval(15, 30), interval(30, 40), interval(60, 70)])
    assert merged == (interval(10, 40), interval(60, 70))


def test_normalize_drops_empty_intervals() -> None:
    assert normalize([interval(5, 5), interval(7, 9)]) == (interval(7, 9),)


def test_normalize_is_order_independent() -> None:
    a = normalize([interval(60, 70), interval(10, 20), interval(15, 30)])
    b = normalize([interval(15, 30), interval(60, 70), interval(10, 20)])
    assert a == b


# -- padding -----------------------------------------------------------------


def test_padding_is_clipped_to_the_available_audio() -> None:
    bound = interval(0, 100)
    assert pad(interval(5, 20), 10, bound) == interval(0, 30)
    assert pad(interval(80, 95), 10, bound) == interval(70, 100)


def test_padding_merges_intervals_that_grow_together() -> None:
    padded = pad_all([interval(10, 20), interval(25, 30)], 5, interval(0, 100))
    assert padded == (interval(5, 35),)


def test_negative_padding_is_rejected() -> None:
    with pytest.raises(ValueError):
        pad(interval(0, 10), -1, interval(0, 100))


# -- complement --------------------------------------------------------------


def test_complement_returns_the_gaps() -> None:
    retained = complement([interval(10, 20), interval(50, 60)], interval(0, 100))
    assert retained == (interval(0, 10), interval(20, 50), interval(60, 100))


def test_complement_of_everything_is_empty() -> None:
    assert complement([interval(0, 100)], interval(0, 100)) == ()


def test_complement_of_nothing_is_the_whole_bound() -> None:
    assert complement([], interval(0, 100)) == (interval(0, 100),)


def test_complement_and_removal_partition_the_audio() -> None:
    bound = interval(0, 1000)
    removed = normalize([interval(100, 250), interval(240, 300), interval(900, 1000)])
    retained = complement(removed, bound)
    assert total_length(removed) + total_length(retained) == bound.length
    for kept in retained:
        for cut in removed:
            assert not kept.intersects(cut)


def test_complement_clips_removals_outside_the_bound() -> None:
    retained = complement([interval(0, 50), interval(990, 2000)], interval(0, 1000))
    assert retained == (interval(50, 990),)


def test_contains_interval() -> None:
    assert contains_interval([interval(0, 100)], interval(10, 20))
    assert not contains_interval([interval(0, 100)], interval(90, 110))


# -- destination mapping -----------------------------------------------------


def test_destination_intervals_follow_the_plan_formula() -> None:
    """For retained ``[a, b)`` after total length ``L``: ``[L, L + b - a)``."""
    retained = [interval(0, 100), interval(300, 450), interval(800, 900)]
    mapping = DestinationMap.build(retained)
    assert mapping.destination_of(0) == interval(0, 100)
    assert mapping.destination_of(1) == interval(100, 250)
    assert mapping.destination_of(2) == interval(250, 350)
    assert mapping.total_samples == 350


def test_retained_sample_maps_by_offset() -> None:
    mapping = DestinationMap.build([interval(0, 100), interval(300, 450)])
    assert mapping.map_sample(50) == 50
    assert mapping.map_sample(300) == 100
    assert mapping.map_sample(449) == 249


def test_removed_sample_has_no_destination() -> None:
    mapping = DestinationMap.build([interval(0, 100), interval(300, 450)])
    with pytest.raises(KeyError):
        mapping.map_sample(200)


def test_word_crossing_a_boundary_has_no_destination() -> None:
    mapping = DestinationMap.build([interval(0, 100)])
    straddling = Word(id=0, text="hello", start_sample=90, end_sample=110)
    with pytest.raises(KeyError):
        mapping.map_word(straddling)


def test_boundary_manifest_is_monotonic_and_contiguous() -> None:
    mapping = DestinationMap.build([interval(0, 100), interval(300, 450)])
    manifest = mapping.boundary_manifest()
    assert [entry["destination_start"] for entry in manifest] == [0, 100]
    assert manifest[0]["destination_end"] == manifest[1]["destination_start"]


# -- resampling --------------------------------------------------------------


def test_removal_intervals_round_outward() -> None:
    mapping = ResampleMap(source_rate=48_000)
    removal = mapping.removal_to_source(interval(1, 2))
    assert removal.start <= 3 and removal.end >= 6


def test_retained_intervals_round_inward() -> None:
    mapping = ResampleMap(source_rate=44_100)
    retained = mapping.retained_to_source(interval(1, 2))
    canonical_start, canonical_end = 1 * 44_100 / 16_000, 2 * 44_100 / 16_000
    assert retained.start >= canonical_start
    assert retained.end <= canonical_end


def test_rounding_never_leaves_removed_audio_inside_a_retained_interval() -> None:
    mapping = ResampleMap(source_rate=48_000)
    removed_source = mapping.removal_to_source(interval(1_000, 2_000))
    for canonical in (interval(0, 1_000), interval(2_000, 3_000)):
        retained_source = mapping.retained_to_source(canonical)
        assert not retained_source.intersects(removed_source)


def test_zero_sample_rate_is_rejected() -> None:
    with pytest.raises(ValueError):
        ResampleMap(source_rate=0)


# -- alignment validation ----------------------------------------------------


def words(*specs: tuple[int, int, int, float]) -> tuple[Word, ...]:
    return tuple(
        Word(id=i, text=f"w{i}", start_sample=start, end_sample=end, confidence=confidence)
        for i, start, end, confidence in specs
    )


def test_clean_alignment_has_no_problems() -> None:
    sequence = words((0, 0, 100, 0.9), (1, 120, 220, 0.9))
    assert validate_alignment(sequence, interval(0, 300), min_confidence=0.5) == ()


def test_overlapping_words_are_reported() -> None:
    sequence = words((0, 0, 200, 0.9), (1, 100, 300, 0.9))
    assert "overlapping_or_nonmonotonic" in validate_alignment(
        sequence, interval(0, 400), min_confidence=0.5
    )


def test_word_outside_the_audio_is_reported() -> None:
    sequence = words((0, 0, 100, 0.9), (1, 200, 900, 0.9))
    assert "word_outside_audio" in validate_alignment(
        sequence, interval(0, 500), min_confidence=0.5
    )


def test_zero_length_word_is_reported() -> None:
    sequence = words((0, 100, 100, 0.9),)
    assert "zero_length_word" in validate_alignment(
        sequence, interval(0, 500), min_confidence=0.5
    )


def test_low_confidence_word_is_reported() -> None:
    sequence = words((0, 0, 100, 0.1),)
    assert "low_confidence_word" in validate_alignment(
        sequence, interval(0, 500), min_confidence=0.5
    )


def test_unreliable_timing_flag_is_reported() -> None:
    flagged = (
        Word(id=0, text="x", start_sample=0, end_sample=100, timing_unreliable=True),
    )
    assert "unreliable_timing" in validate_alignment(
        flagged, interval(0, 500), min_confidence=0.5
    )
