"""Log-leak guards, audio buffers, VAD and streaming reconciliation."""

from __future__ import annotations

import json
import logging

import pytest

from dictation.asr.mock import ScriptedAsr, script_from_text, script_with_mistranscription
from dictation.asr.reconciler import PartialReconciler
from dictation.asr.whisper_cpp import TOKEN_PROBABILITY_FLOOR, find_binary, parse_whisper_json
from dictation.capture.audio_buffer import BoundedPcmBuffer, read_wav, write_wav
from dictation.capture.devices import (
    FileSource,
    MemorySource,
    MicrophoneSource,
    SilenceSource,
    chunks_of,
    synthetic_speech,
)
from dictation.capture.vad import EnergyVad
from dictation.errors import BufferOverflow, CapabilityUnavailable, WorkerError
from dictation.logging_ import EventLog, ForbiddenLogField, content_digest, sanitize
from dictation.privacy import rules
from dictation.types import CANONICAL_SAMPLE_WIDTH, Segment, SampleInterval

# -- logging -----------------------------------------------------------------


def test_reason_codes_and_numbers_are_loggable() -> None:
    assert sanitize({"session": "session-abc123", "clips": 3, "eligible": True})


def test_free_text_is_refused() -> None:
    with pytest.raises(ForbiddenLogField):
        sanitize({"detail": "Send the parcel to Jane at 14 Oak Street"})


@pytest.mark.parametrize("key", ["text", "transcript", "prompt", "spans", "token", "password"])
def test_content_shaped_field_names_are_refused(key: str) -> None:
    with pytest.raises(ForbiddenLogField):
        sanitize({key: "anything"})


def test_unloggable_types_are_refused() -> None:
    with pytest.raises(ForbiddenLogField):
        sanitize({"payload": {"nested": "object"}})


def test_emitted_record_contains_only_declared_fields(caplog) -> None:
    log = EventLog(logging.getLogger("dictation-test"))
    with caplog.at_level(logging.INFO, logger="dictation-test"):
        record = log.emit("dataset.rejected", session="session-1", reason="uncertain_analysis")
    assert set(record) == {"event", "ts", "session", "reason"}
    payload = json.loads(caplog.records[-1].message)
    assert payload["reason"] == "uncertain_analysis"


def test_digest_is_stable_and_does_not_reveal_the_text() -> None:
    first = content_digest("Send the parcel to Jane")
    assert first == content_digest("Send the parcel to Jane")
    assert first != content_digest("Send the parcel to Joan")
    assert "Jane" not in first


# -- bounded buffers ---------------------------------------------------------


def test_buffer_tracks_samples_and_duration() -> None:
    buffer = BoundedPcmBuffer.for_seconds(1.0)
    buffer.append(bytes(1600 * CANONICAL_SAMPLE_WIDTH))
    assert buffer.sample_count == 1600
    assert buffer.duration_ms == 100


def test_overflow_is_an_error_not_a_silent_drop() -> None:
    buffer = BoundedPcmBuffer(max_samples=10)
    with pytest.raises(BufferOverflow):
        buffer.append(bytes(40))
    assert buffer.sample_count == 0


def test_odd_length_payload_is_rejected() -> None:
    buffer = BoundedPcmBuffer(max_samples=10)
    with pytest.raises(ValueError):
        buffer.append(b"\x00")


def test_reading_beyond_the_buffer_is_refused() -> None:
    buffer = BoundedPcmBuffer(max_samples=10)
    buffer.append_silence(4)
    with pytest.raises(ValueError):
        buffer.read(SampleInterval(0, 8))


def test_reading_intervals_concatenates_in_order() -> None:
    buffer = BoundedPcmBuffer(max_samples=100)
    buffer.append(b"".join(index.to_bytes(2, "little") for index in range(10)))
    joined = buffer.read_intervals([SampleInterval(0, 2), SampleInterval(5, 7)])
    assert joined == (
        b"".join(index.to_bytes(2, "little") for index in (0, 1, 5, 6))
    )


def test_clear_drops_the_samples() -> None:
    buffer = BoundedPcmBuffer(max_samples=100)
    buffer.append_silence(10)
    buffer.clear()
    assert buffer.sample_count == 0
    assert buffer.read() == b""


def test_warmup_buffer_drains_into_the_session_buffer() -> None:
    warmup = BoundedPcmBuffer(max_samples=100)
    session = BoundedPcmBuffer(max_samples=1000)
    warmup.append_silence(50)
    moved = warmup.drain_into(session)
    assert moved == 50
    assert warmup.sample_count == 0


def test_wav_round_trip(tmp_path) -> None:
    pcm = synthetic_speech([(0.1, 0.4)], total_seconds=0.5)
    path = tmp_path / "audio.wav"
    write_wav(path, pcm)
    read_back, rate = read_wav(path)
    assert read_back == pcm
    assert rate == 16_000


def test_file_source_reads_canonical_audio(tmp_path) -> None:
    pcm = synthetic_speech([(0.1, 0.4)], total_seconds=0.5)
    path = tmp_path / "audio.wav"
    write_wav(path, pcm)
    source = FileSource(path)
    source.open()
    collected = b""
    while chunk := source.read_chunk(320):
        collected += chunk
    assert collected == pcm


def test_microphone_source_reports_the_missing_backend() -> None:
    source = MicrophoneSource()
    if MicrophoneSource.available():
        pytest.skip("a capture backend is installed on this machine")
    with pytest.raises(CapabilityUnavailable, match="backend"):
        source.open()


def test_chunks_cover_the_whole_signal() -> None:
    pcm = synthetic_speech([(0.0, 0.5)], total_seconds=0.5)
    assert b"".join(chunks_of(pcm, 320)) == pcm


# -- voice activity ----------------------------------------------------------


def test_silence_is_detected_as_silence() -> None:
    vad = EnergyVad()
    silent = bytes(16_000 * CANONICAL_SAMPLE_WIDTH)
    assert vad.speech_ratio(silent) == 0.0
    assert vad.is_effectively_silent(silent)


def test_speech_bursts_are_detected() -> None:
    vad = EnergyVad()
    pcm = synthetic_speech([(0.5, 1.5), (2.0, 2.8)], total_seconds=3.0)
    assert not vad.is_effectively_silent(pcm)
    intervals = vad.speech_intervals(pcm)
    assert intervals
    assert all(interval.length > 0 for interval in intervals)


def test_speech_intervals_are_inside_the_bursts() -> None:
    vad = EnergyVad()
    pcm = synthetic_speech([(1.0, 2.0)], total_seconds=3.0)
    intervals = vad.speech_intervals(pcm)
    assert intervals
    first, last = intervals[0], intervals[-1]
    assert first.start >= int(0.8 * 16_000)
    assert last.end <= int(2.4 * 16_000)


def test_empty_audio_has_no_speech() -> None:
    vad = EnergyVad()
    assert vad.frame_flags(b"") == []
    assert vad.speech_ratio(b"") == 0.0


def test_silence_source_produces_only_silence() -> None:
    source = SilenceSource(0.5)
    source.open()
    pcm = source.read_chunk(16_000)
    assert set(pcm) == {0}


# -- streaming reconciliation ------------------------------------------------


def test_later_revision_replaces_earlier_text() -> None:
    reconciler = PartialReconciler()
    assert reconciler.apply(Segment("seg-1", 1, "the meeting", 0, 100))
    assert reconciler.apply(Segment("seg-1", 2, "the meeting starts", 0, 200))
    assert reconciler.text() == "the meeting starts"


def test_stale_revision_is_dropped() -> None:
    reconciler = PartialReconciler()
    reconciler.apply(Segment("seg-1", 5, "final text", 0, 100))
    assert not reconciler.apply(Segment("seg-1", 4, "older text", 0, 100))
    assert reconciler.text() == "final text"


def test_identical_revision_is_not_reapplied() -> None:
    reconciler = PartialReconciler()
    reconciler.apply(Segment("seg-1", 1, "same", 0, 100))
    assert not reconciler.apply(Segment("seg-1", 1, "same", 0, 100))


def test_final_segment_is_not_revised_by_a_partial() -> None:
    reconciler = PartialReconciler()
    reconciler.apply(Segment("seg-1", 1, "final", 0, 100, is_final=True))
    assert not reconciler.apply(Segment("seg-1", 2, "partial", 0, 100))
    assert reconciler.text() == "final"


def test_segments_are_ordered_by_start_time() -> None:
    reconciler = PartialReconciler()
    reconciler.apply(Segment("seg-2", 1, "second", 200, 300))
    reconciler.apply(Segment("seg-1", 1, "first", 0, 100))
    assert reconciler.text() == "first second"


def test_scripted_streaming_emits_revisions() -> None:
    script = script_from_text("one two three four five six seven eight")
    asr = ScriptedAsr(script, words_per_segment=4)
    pcm = synthetic_speech([(0.0, 3.0)], total_seconds=3.0)
    reconciler = PartialReconciler()
    revisions: list[int] = []
    for update in asr.stream(chunks_of(pcm, 4_000), session_id="s1"):
        if reconciler.apply(update.segment):
            revisions.append(update.segment.revision)
    assert revisions
    assert max(revisions) > 1


# -- whisper.cpp output parsing ---------------------------------------------


def test_token_level_json_becomes_words() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 1000},
                "text": " Hello world",
                "tokens": [
                    {"text": " Hello", "offsets": {"from": 0, "to": 400}, "p": 0.9},
                    {"text": " world", "offsets": {"from": 400, "to": 900}, "p": 0.8},
                ],
            }
        ],
        "result": {"language": "en"},
    }
    transcript = parse_whisper_json(payload, session_id="s1")
    assert [word.text for word in transcript.words] == ["hello", "world"]
    assert transcript.words[0].start_sample == 0
    assert transcript.words[1].end_sample == int(0.9 * 16_000)
    assert transcript.language == "en"


def test_low_probability_token_is_marked_unreliable() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 500},
                "text": " maybe",
                "tokens": [
                    {
                        "text": " maybe",
                        "offsets": {"from": 0, "to": 500},
                        "p": TOKEN_PROBABILITY_FLOOR / 2,
                    }
                ],
            }
        ]
    }
    transcript = parse_whisper_json(payload, session_id="s1")
    assert transcript.words[0].timing_unreliable


def test_special_tokens_are_skipped() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 500},
                "text": " hi",
                "tokens": [
                    {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
                    {"text": " hi", "offsets": {"from": 0, "to": 500}, "p": 0.9},
                ],
            }
        ]
    }
    transcript = parse_whisper_json(payload, session_id="s1")
    assert [word.text for word in transcript.words] == ["hi"]


def test_segment_without_token_offsets_is_flagged_unreliable() -> None:
    payload = {
        "transcription": [
            {"offsets": {"from": 0, "to": 1000}, "text": " two words here"}
        ]
    }
    transcript = parse_whisper_json(payload, session_id="s1")
    assert len(transcript.words) == 3
    assert all(word.timing_unreliable for word in transcript.words)


def test_malformed_whisper_output_is_a_worker_error() -> None:
    with pytest.raises(WorkerError):
        parse_whisper_json({"result": {}}, session_id="s1")
    with pytest.raises(WorkerError):
        parse_whisper_json({"transcription": ["not an object"]}, session_id="s1")


def test_transcript_word_ids_are_strictly_increasing() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 900},
                "text": " a b",
                "tokens": [
                    {"text": " a", "offsets": {"from": 0, "to": 400}, "p": 0.9},
                    {"text": " b", "offsets": {"from": 400, "to": 900}, "p": 0.9},
                ],
            }
        ]
    }
    transcript = parse_whisper_json(payload, session_id="s1")
    assert transcript.word_ids() == (0, 1)


def test_binary_lookup_does_not_raise_when_absent() -> None:
    assert find_binary() is None or find_binary().exists()


# -- the mistranscription case the plan warns about --------------------------


def test_a_mistranscribed_identifier_can_escape_the_rules() -> None:
    """Plan section 2: the privacy model only sees ASR output.

    Here the recogniser turned "Oak" into "oh" and dropped the number, so the
    street-address rule no longer matches.  The test documents the gap rather
    than pretending it is closed; it is why upload stays gated on measured
    recall over real recordings.
    """
    clean_script = script_from_text("Send the parcel to 14 Oak Street tomorrow.")
    mangled = script_with_mistranscription(
        "Send the parcel to 14 Oak Street tomorrow.",
        {"14": "for", "Oak": "oh", "Street": "streak"},
    )
    pcm = synthetic_speech([(0.0, 3.0)], total_seconds=3.0)

    good = ScriptedAsr(clean_script).transcribe(pcm, session_id="s1")
    bad = ScriptedAsr(mangled).transcribe(pcm, session_id="s2")

    assert rules.detect(good).spans
    assert not rules.detect(bad).spans
