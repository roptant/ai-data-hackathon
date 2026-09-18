"""Shared fixtures: synthetic sessions with known word/audio alignment.

The audio is tone bursts placed exactly where the scripted words are, so every
interval assertion in these tests has a ground truth.  No real speech, no real
model: the point is to pin down the deterministic machinery around them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dictation.asr.mock import ScriptedAsr, ScriptedWord, script_from_text
from dictation.capture.devices import synthetic_speech
from dictation.types import CANONICAL_SAMPLE_RATE, Transcript


@dataclass(frozen=True, slots=True)
class Session:
    """A synthetic recording: audio plus the transcript that matches it."""

    pcm: bytes
    transcript: Transcript
    script: tuple[ScriptedWord, ...]

    @property
    def sample_rate(self) -> int:
        return self.transcript.sample_rate

    def text_of(self, start: int, end: int) -> str:
        return " ".join(word.text for word in self.transcript.words[start:end])


def make_session(
    text: str,
    *,
    session_id: str = "session-test",
    word_ms: int = 320,
    gap_ms: int = 60,
    sentence_gap_ms: int = 900,
    tail_ms: int = 400,
    confidence: float = 0.95,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
) -> Session:
    """Build a session whose audio has speech exactly under each word."""
    script = script_from_text(
        text,
        word_ms=word_ms,
        gap_ms=gap_ms,
        sentence_gap_ms=sentence_gap_ms,
        confidence=confidence,
    )
    total_ms = (script[-1].end_ms if script else 0) + tail_ms
    pcm = synthetic_speech(
        [(word.start_ms / 1000, word.end_ms / 1000) for word in script],
        total_seconds=total_ms / 1000,
        sample_rate=sample_rate,
    )
    asr = ScriptedAsr(script)
    asr.warm_up()
    transcript = asr.transcribe(pcm, session_id=session_id, sample_rate=sample_rate)
    return Session(pcm=pcm, transcript=transcript, script=script)


@pytest.fixture
def parcel_session() -> Session:
    """The plan's own example (section 7).

    The first sentence carries a name and a street address and must go; the
    rest is ordinary dictation and should survive.
    """
    return make_session(
        "Send the parcel to Jane at 14 Oak Street. "
        "The meeting starts tomorrow and I will bring the printed agenda. "
        "Remind me to water the plants before we leave the office."
    )


@pytest.fixture
def clean_session() -> Session:
    """Nothing sensitive: two plain sentences of dictation."""
    return make_session(
        "The meeting starts tomorrow and I will bring the printed agenda. "
        "Remind me to water the plants before we leave the office."
    )


@pytest.fixture
def secret_session() -> Session:
    """Spoken credentials and a spoken digit run."""
    return make_session(
        "The password is hunter seven seven seven and the account number is "
        "four nine two seven one three eight. "
        "Otherwise the deployment went fine and the logs look healthy."
    )
