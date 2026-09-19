//! Speech-recognition results: live partial reconciliation and freezing.
//!
//! Partial hypotheses come from repeated passes over the not-yet-committed
//! tail of the recording. Segments that end well before the audio edge are
//! committed with a fixed identity; the remaining tail is one provisional
//! segment whose revision rises with each pass. Clients replace a segment's
//! text by `(segment_id, revision)` rather than appending (plan §6, §8).
//!
//! The canonical transcript comes from one final pass over the whole
//! recording and supersedes every partial.

use crate::{
    time_map::SampleInterval,
    transcript::{FrozenTranscript, TranscriptError, Word},
    vad::SessionActivity,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RecognizedWord {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub probability: f32,
    pub timing_unreliable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecognizedSegment {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub no_speech_probability: f32,
    pub words: Vec<RecognizedWord>,
}

impl RecognizedSegment {
    /// Shifts sample positions by `offset` (the pass's start in the session).
    #[must_use]
    pub fn offset(mut self, offset: u64) -> Self {
        self.start_sample += offset;
        self.end_sample += offset;
        for word in &mut self.words {
            word.start_sample += offset;
            word.end_sample += offset;
        }
        self
    }
}

/// Spoken-form label: lowercase with surrounding punctuation removed.
/// Punctuation and casing stay in the display form only.
#[must_use]
pub fn normalize_spoken(display: &str) -> String {
    display
        .trim_matches(|character: char| !character.is_alphanumeric())
        .to_lowercase()
}

/// Segments that should never become text: no detected speech in their
/// interval, or whisper's own no-speech estimate is high.
#[must_use]
pub fn is_hallucination_risk(
    segment: &RecognizedSegment,
    activity: &SessionActivity,
    max_no_speech_probability: f32,
) -> bool {
    let Ok(interval) = SampleInterval::new(segment.start_sample, segment.end_sample.max(segment.start_sample)) else {
        return true;
    };
    segment.no_speech_probability > max_no_speech_probability
        || activity.speech_samples(interval) == 0
}

/// Freezes recognized segments into immutable, sequentially numbered words.
///
/// # Errors
///
/// Propagates invalid word timing or confidence.
pub fn freeze(
    session_id: &str,
    revision: u64,
    segments: &[RecognizedSegment],
    sample_rate: u32,
    total_samples: u64,
    language: &str,
) -> Result<FrozenTranscript, TranscriptError> {
    let mut words = Vec::new();
    for segment in segments {
        for recognized in &segment.words {
            let spoken = normalize_spoken(&recognized.text);
            if spoken.is_empty() {
                continue;
            }
            let end = recognized.end_sample.min(total_samples);
            let start = recognized.start_sample.min(end);
            let id = words.len() as u64;
            let probability = if recognized.probability.is_finite() {
                recognized.probability.clamp(0.0, 1.0)
            } else {
                0.0
            };
            words.push(
                Word::new(id, spoken, start, end, probability)?
                    .with_display(recognized.text.trim())
                    .with_unreliable_timing(recognized.timing_unreliable),
            );
        }
    }
    FrozenTranscript::new(
        session_id,
        revision,
        words,
        sample_rate,
        total_samples,
        language,
    )
}

/// Display text for insertion: recognized segment text in time order.
#[must_use]
pub fn display_text(segments: &[RecognizedSegment]) -> String {
    segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A replaceable caption segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentUpdate {
    pub segment_id: String,
    pub revision: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveSettings {
    /// Segments ending this long before the audio edge are committed.
    pub stability_margin_samples: u64,
    /// Force commitment when the uncommitted tail exceeds this length, so a
    /// pass never grows beyond whisper's comfortable window.
    pub max_tail_samples: u64,
}

impl Default for LiveSettings {
    fn default() -> Self {
        Self {
            stability_margin_samples: 16_000,
            max_tail_samples: 16_000 * 20,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LiveReconciler {
    settings: LiveSettings,
    committed_until: u64,
    committed: Vec<SegmentUpdate>,
    provisional: Option<SegmentUpdate>,
    next_index: u64,
    seq: u64,
}

impl LiveReconciler {
    #[must_use]
    pub fn new(settings: LiveSettings) -> Self {
        Self {
            settings,
            ..Self::default()
        }
    }

    /// Where the next partial pass should begin.
    #[must_use]
    pub const fn committed_until(&self) -> u64 {
        self.committed_until
    }

    /// Applies one pass over `[committed_until, audio_end)`. Segments are in
    /// session sample positions. Returns the updates clients must apply.
    pub fn apply_pass(
        &mut self,
        segments: &[RecognizedSegment],
        audio_end: u64,
    ) -> Vec<SegmentUpdate> {
        let mut updates = Vec::new();
        let stable_before = audio_end.saturating_sub(self.settings.stability_margin_samples);
        let force = audio_end.saturating_sub(self.committed_until) > self.settings.max_tail_samples;
        let mut remaining: &[RecognizedSegment] = segments;
        // Commit every segment but the last that ended before the margin; when
        // forced, commit all but the last regardless.
        while remaining.len() > 1 {
            let first = &remaining[0];
            if !(force || first.end_sample <= stable_before) {
                break;
            }
            let id = self.provisional.take().map_or_else(
                || {
                    let id = format!("segment-{}", self.next_index);
                    self.next_index += 1;
                    (id, 1)
                },
                |previous| (previous.segment_id, previous.revision + 1),
            );
            let update = SegmentUpdate {
                segment_id: id.0,
                revision: id.1,
                start_sample: first.start_sample,
                end_sample: first.end_sample,
                text: first.text.trim().to_owned(),
            };
            self.committed_until = first.end_sample.max(self.committed_until);
            self.committed.push(update.clone());
            updates.push(update);
            remaining = &remaining[1..];
        }
        let tail_text = display_text(remaining);
        if tail_text.is_empty() {
            return updates;
        }
        let start = remaining.first().map_or(self.committed_until, |s| s.start_sample);
        let end = remaining.last().map_or(audio_end, |s| s.end_sample);
        let next = match self.provisional.take() {
            Some(previous) if previous.text == tail_text => {
                self.provisional = Some(previous);
                return updates;
            }
            Some(previous) => SegmentUpdate {
                segment_id: previous.segment_id,
                revision: previous.revision + 1,
                start_sample: start,
                end_sample: end,
                text: tail_text,
            },
            None => {
                let id = format!("segment-{}", self.next_index);
                self.next_index += 1;
                SegmentUpdate {
                    segment_id: id,
                    revision: 1,
                    start_sample: start,
                    end_sample: end,
                    text: tail_text,
                }
            }
        };
        self.provisional = Some(next.clone());
        updates.push(next);
        updates
    }

    /// Monotonic sequence number for API events, so clients detect gaps.
    pub const fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Current best text: committed segments then the provisional tail.
    #[must_use]
    pub fn text(&self) -> String {
        self.committed
            .iter()
            .chain(self.provisional.iter())
            .map(|segment| segment.text.as_str())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Snapshot for a reconnecting authorized client.
    #[must_use]
    pub fn snapshot(&self) -> Vec<SegmentUpdate> {
        self.committed
            .iter()
            .chain(self.provisional.iter())
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::VadSettings;

    fn segment(text: &str, start: u64, end: u64) -> RecognizedSegment {
        RecognizedSegment {
            text: text.to_owned(),
            start_sample: start,
            end_sample: end,
            no_speech_probability: 0.01,
            words: text
                .split_whitespace()
                .enumerate()
                .map(|(index, word)| RecognizedWord {
                    text: word.to_owned(),
                    start_sample: start + index as u64 * 100,
                    end_sample: start + index as u64 * 100 + 90,
                    probability: 0.9,
                    timing_unreliable: false,
                })
                .collect(),
        }
    }

    #[test]
    fn spoken_form_strips_punctuation_and_case() {
        assert_eq!(normalize_spoken("Americans,"), "americans");
        assert_eq!(normalize_spoken("don't"), "don't");
        assert_eq!(normalize_spoken("—"), "");
        assert_eq!(normalize_spoken("14"), "14");
    }

    #[test]
    fn freezing_numbers_words_and_keeps_display() {
        let transcript = freeze(
            "s",
            1,
            &[segment("Hello, world.", 0, 1_000)],
            16_000,
            1_000,
            "en",
        )
        .unwrap();
        let words = transcript.words();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text(), "hello");
        assert_eq!(words[0].display_text(), "Hello,");
        assert_eq!(words[1].id(), 1);
    }

    #[test]
    fn provisional_tail_is_revised_not_appended() {
        let mut live = LiveReconciler::new(LiveSettings::default());
        let first = live.apply_pass(&[segment("the meet", 0, 8_000)], 8_000);
        assert_eq!(first.len(), 1);
        assert_eq!((first[0].segment_id.as_str(), first[0].revision), ("segment-0", 1));
        let second = live.apply_pass(&[segment("the meeting starts", 0, 12_000)], 12_000);
        assert_eq!((second[0].segment_id.as_str(), second[0].revision), ("segment-0", 2));
        assert_eq!(live.text(), "the meeting starts");
        assert!(live.apply_pass(&[segment("the meeting starts", 0, 12_000)], 13_000).is_empty());
    }

    #[test]
    fn stable_segments_commit_and_advance_the_pass_start() {
        let mut live = LiveReconciler::new(LiveSettings::default());
        live.apply_pass(&[segment("hello there", 0, 16_000)], 16_000);
        let updates = live.apply_pass(
            &[segment("hello there.", 0, 16_000), segment("next bit", 20_000, 40_000)],
            40_000,
        );
        assert_eq!(updates[0].segment_id, "segment-0");
        assert_eq!(updates[0].text, "hello there.");
        assert_eq!(updates[1].segment_id, "segment-1");
        assert_eq!(live.committed_until(), 16_000);
        assert_eq!(live.text(), "hello there. next bit");
    }

    #[test]
    fn segments_without_detected_speech_are_suppressed() {
        let silence = SessionActivity::analyze(&[0; 32_000], 16_000, VadSettings::default());
        assert!(is_hallucination_risk(&segment("Thank you.", 0, 32_000), &silence, 0.6));
    }
}
