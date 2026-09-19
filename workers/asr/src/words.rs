//! Conversion from whisper.cpp tokens to sample-aligned words.
//!
//! whisper.cpp word timing is experimental, so this module is conservative:
//! each word extends until the next word begins, which keeps inter-word gaps
//! inside some word rather than in unowned audio, and any token without a DTW
//! onset marks its word's timing unreliable. Privacy filtering discards
//! utterances whose timing is unreliable.

use dictation_worker::messages::AsrWord;

/// Samples per whisper.cpp timestamp unit (10 ms at 16 kHz).
pub const SAMPLES_PER_CENTISECOND: u64 = 160;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub text: String,
    /// Heuristic start/end in centiseconds relative to the request audio.
    pub t0: i64,
    pub t1: i64,
    /// DTW onset in centiseconds, or negative when unavailable.
    pub t_dtw: i64,
    pub probability: f32,
}

fn to_samples(centiseconds: i64) -> u64 {
    u64::try_from(centiseconds.max(0)).unwrap_or(0) * SAMPLES_PER_CENTISECOND
}

fn is_punctuation_only(text: &str) -> bool {
    !text.trim().is_empty()
        && text
            .trim()
            .chars()
            .all(|character| !character.is_alphanumeric())
}

/// Groups text tokens (special tokens already removed) into words.
///
/// A token starting with whitespace begins a new word; punctuation-only tokens
/// attach to the previous word. `segment_end_sample` bounds the final word and
/// `audio_samples` bounds everything.
#[must_use]
pub fn group_words(
    tokens: &[Token],
    segment_start_sample: u64,
    segment_end_sample: u64,
    audio_samples: u64,
) -> Vec<AsrWord> {
    struct Pending {
        text: String,
        onset: u64,
        probability: f32,
        unreliable: bool,
    }
    let mut pending: Vec<Pending> = Vec::new();
    for token in tokens {
        if token.text.is_empty() {
            continue;
        }
        let starts_word = token.text.starts_with(char::is_whitespace) || pending.is_empty();
        let attaches = is_punctuation_only(&token.text) && !pending.is_empty();
        let has_dtw = token.t_dtw >= 0;
        let onset = if has_dtw {
            to_samples(token.t_dtw)
        } else {
            to_samples(token.t0)
        };
        if starts_word && !attaches {
            pending.push(Pending {
                text: token.text.trim_start().to_owned(),
                onset,
                probability: token.probability,
                unreliable: !has_dtw,
            });
        } else if let Some(last) = pending.last_mut() {
            last.text.push_str(&token.text);
            last.probability = last.probability.min(token.probability);
            last.unreliable |= !has_dtw;
        }
    }
    let upper = segment_end_sample.min(audio_samples);
    let mut words = Vec::with_capacity(pending.len());
    let mut previous_start = segment_start_sample.min(upper);
    for (index, word) in pending.iter().enumerate() {
        let start = word.onset.clamp(previous_start, upper);
        let next_onset = pending
            .get(index + 1)
            .map_or(upper, |next| next.onset.clamp(start, upper));
        let end = next_onset.max(start);
        // A zero-length word cannot be cut precisely: flag it.
        let unreliable = word.unreliable || end == start || word.onset < previous_start;
        previous_start = start;
        let text = word.text.trim().to_owned();
        if text.is_empty() {
            continue;
        }
        words.push(AsrWord {
            text,
            start_sample: start,
            end_sample: end,
            probability: if word.probability.is_finite() {
                word.probability.clamp(0.0, 1.0)
            } else {
                0.0
            },
            timing_unreliable: unreliable,
        });
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str, t_dtw: i64, probability: f32) -> Token {
        Token {
            text: text.to_owned(),
            t0: t_dtw,
            t1: t_dtw + 10,
            t_dtw,
            probability,
        }
    }

    #[test]
    fn words_tile_the_segment_and_absorb_punctuation() {
        let words = group_words(
            &[
                token(" Hello", 10, 0.9),
                token(" wor", 50, 0.8),
                token("ld", 60, 0.95),
                token(".", 70, 0.99),
            ],
            0,
            16_000,
            32_000,
        );
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(
            (words[0].start_sample, words[0].end_sample),
            (1_600, 8_000)
        );
        assert_eq!(words[1].text, "world.");
        assert_eq!(
            (words[1].start_sample, words[1].end_sample),
            (8_000, 16_000)
        );
        assert!((words[1].probability - 0.8).abs() < f32::EPSILON);
        assert!(words.iter().all(|word| !word.timing_unreliable));
    }

    #[test]
    fn missing_dtw_marks_timing_unreliable() {
        let words = group_words(&[token(" a", -1, 0.9)], 0, 1_600, 1_600);
        assert!(words[0].timing_unreliable);
    }

    #[test]
    fn nonmonotonic_onsets_are_clamped_and_flagged() {
        let words = group_words(
            &[token(" one", 50, 0.9), token(" two", 20, 0.9)],
            0,
            16_000,
            16_000,
        );
        assert!(words[1].start_sample >= words[0].start_sample);
        assert!(words[1].timing_unreliable);
    }

    #[test]
    fn timings_never_exceed_the_audio() {
        let words = group_words(&[token(" late", 500, 0.9)], 0, 99_999, 4_000);
        assert!(words[0].end_sample <= 4_000);
        assert!(words[0].start_sample <= words[0].end_sample);
    }
}
