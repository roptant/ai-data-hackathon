//! Privacy-preserving conversion from word spans to synchronized audio cuts.
//!
//! Sensitive spans expand to utterances by default, convert through frozen word
//! alignment, receive outward padding, and then pull in every word touched by
//! that padding. Pull-in repeats to a fixed point so retained text never names
//! audio that has been cut.

use std::collections::BTreeSet;

use crate::{
    time_map::{SampleInterval, milliseconds_to_samples, normalize, pad_all},
    transcript::{FrozenTranscript, RemovalAction, SensitiveSpan, Word},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sentence {
    pub index: usize,
    pub start_word_id: u64,
    pub end_word_id_exclusive: u64,
    pub interval: SampleInterval,
}

impl Sentence {
    #[must_use]
    pub const fn contains_word(self, word_id: u64) -> bool {
        self.start_word_id <= word_id && word_id < self.end_word_id_exclusive
    }

    #[must_use]
    pub const fn overlaps_span(self, span: &SensitiveSpan) -> bool {
        self.start_word_id < span.end_word_id_exclusive
            && span.start_word_id < self.end_word_id_exclusive
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovalPlan {
    pub intervals: Vec<SampleInterval>,
    pub removed_word_ids: BTreeSet<u64>,
    pub removed_sentence_indices: BTreeSet<usize>,
    pub padding_samples: u64,
    pub iterations: usize,
}

impl RemovalPlan {
    #[must_use]
    pub fn removed_samples(&self) -> u64 {
        self.intervals.iter().map(|interval| interval.len()).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemovalSettings {
    pub padding_ms: u64,
    pub expand_to_sentence: bool,
}

impl Default for RemovalSettings {
    fn default() -> Self {
        Self {
            padding_ms: 250,
            expand_to_sentence: true,
        }
    }
}

/// Splits frozen words at terminal punctuation or a sufficiently long pause.
#[must_use]
pub fn sentences_from_words(words: &[Word], sample_rate: u32, pause_ms: u64) -> Vec<Sentence> {
    if words.is_empty() {
        return Vec::new();
    }
    let pause_samples = milliseconds_to_samples(pause_ms, sample_rate);
    let mut sentences = Vec::new();
    let mut sentence_start = 0;

    for (index, word) in words.iter().enumerate() {
        let terminal = word.display_text().trim_end().ends_with(['.', '!', '?']);
        let long_pause = words.get(index + 1).is_some_and(|next| {
            next.interval()
                .start()
                .saturating_sub(word.interval().end())
                >= pause_samples
        });
        if terminal || long_pause {
            push_sentence(&mut sentences, &words[sentence_start..=index]);
            sentence_start = index + 1;
        }
    }
    if sentence_start < words.len() {
        push_sentence(&mut sentences, &words[sentence_start..]);
    }
    sentences
}

fn push_sentence(sentences: &mut Vec<Sentence>, words: &[Word]) {
    let (Some(first), Some(last)) = (words.first(), words.last()) else {
        return;
    };
    sentences.push(Sentence {
        index: sentences.len(),
        start_word_id: first.id(),
        end_word_id_exclusive: last.id().saturating_add(1),
        interval: SampleInterval::new(first.interval().start(), last.interval().end())
            .expect("a validated word sequence cannot invert its outer bounds"),
    });
}

#[must_use]
pub fn expand_to_sentences(
    spans: &[SensitiveSpan],
    sentences: &[Sentence],
    expand_all: bool,
) -> (BTreeSet<u64>, BTreeSet<usize>) {
    let mut word_ids = BTreeSet::new();
    let mut sentence_indices = BTreeSet::new();
    for span in spans {
        let sentence_scope = expand_all || span.action == RemovalAction::DropSentence;
        let mut touched = false;
        if sentence_scope {
            for sentence in sentences
                .iter()
                .copied()
                .filter(|sentence| sentence.overlaps_span(span))
            {
                touched = true;
                sentence_indices.insert(sentence.index);
                word_ids.extend(sentence.start_word_id..sentence.end_word_id_exclusive);
            }
        }
        if !touched {
            word_ids.extend(span.start_word_id..span.end_word_id_exclusive);
        }
    }
    (word_ids, sentence_indices)
}

/// Builds synchronized word and sample removals to a fixed point.
#[must_use]
pub fn plan_removals(
    transcript: &FrozenTranscript,
    spans: &[SensitiveSpan],
    sentences: &[Sentence],
    settings: RemovalSettings,
    extra_sentence_indices: &BTreeSet<usize>,
    bound: Option<SampleInterval>,
) -> RemovalPlan {
    let audio_bound = bound.unwrap_or_else(|| transcript.span());
    let padding_samples = milliseconds_to_samples(settings.padding_ms, transcript.sample_rate());
    let (candidate_ids, mut removed_sentence_indices) =
        expand_to_sentences(spans, sentences, settings.expand_to_sentence);
    removed_sentence_indices.extend(extra_sentence_indices);
    let whole_sentence_indices = removed_sentence_indices.clone();

    let mut selected: BTreeSet<u64> = candidate_ids
        .into_iter()
        .filter(|word_id| transcript.word_by_id(*word_id).is_some())
        .collect();
    for sentence_index in extra_sentence_indices {
        if let Some(sentence) = sentences.get(*sentence_index) {
            selected.extend(
                transcript
                    .words()
                    .iter()
                    .filter(|word| sentence.contains_word(word.id()))
                    .map(Word::id),
            );
        }
    }

    if selected.is_empty() {
        return RemovalPlan {
            intervals: Vec::new(),
            removed_word_ids: selected,
            removed_sentence_indices,
            padding_samples,
            iterations: 0,
        };
    }

    let mut iterations = 0;
    let mut intervals = padded_intervals(
        transcript,
        &selected,
        sentences,
        &whole_sentence_indices,
        padding_samples,
        audio_bound,
    );
    loop {
        iterations += 1;
        let mut grown = selected.clone();
        for word in transcript.words() {
            if !selected.contains(&word.id())
                && intervals
                    .iter()
                    .any(|interval| word.interval().intersects(*interval))
            {
                grown.insert(word.id());
            }
        }
        if grown == selected {
            break;
        }
        selected = grown;
        intervals = padded_intervals(
            transcript,
            &selected,
            sentences,
            &whole_sentence_indices,
            padding_samples,
            audio_bound,
        );
    }

    for sentence in sentences {
        if transcript
            .words()
            .iter()
            .any(|word| sentence.contains_word(word.id()) && selected.contains(&word.id()))
        {
            removed_sentence_indices.insert(sentence.index);
        }
    }

    RemovalPlan {
        intervals,
        removed_word_ids: selected,
        removed_sentence_indices,
        padding_samples,
        iterations,
    }
}

fn padded_intervals(
    transcript: &FrozenTranscript,
    selected: &BTreeSet<u64>,
    sentences: &[Sentence],
    whole_sentence_indices: &BTreeSet<usize>,
    padding_samples: u64,
    bound: SampleInterval,
) -> Vec<SampleInterval> {
    let word_intervals = selected
        .iter()
        .filter_map(|word_id| transcript.word_by_id(*word_id))
        .map(Word::interval);
    let sentence_intervals = whole_sentence_indices
        .iter()
        .filter_map(|index| sentences.get(*index))
        .map(|sentence| sentence.interval);
    pad_all(
        normalize(word_intervals.chain(sentence_intervals)),
        padding_samples,
        bound,
    )
}

#[must_use]
pub fn words_fully_inside(
    transcript: &FrozenTranscript,
    interval: SampleInterval,
    excluded: &BTreeSet<u64>,
) -> Vec<u64> {
    transcript
        .words()
        .iter()
        .filter(|word| {
            !excluded.contains(&word.id())
                && interval.start() <= word.interval().start()
                && word.interval().end() <= interval.end()
        })
        .map(Word::id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{DetectionSource, SensitiveCategory};

    fn transcript(words: Vec<Word>, total_samples: u64) -> FrozenTranscript {
        FrozenTranscript::new("session", 1, words, 1_000, total_samples, "en").unwrap()
    }

    fn word(id: u64, text: &str, start: u64, end: u64) -> Word {
        Word::new(id, text, start, end, 1.0).unwrap()
    }

    fn span(start: u64, end: u64, action: RemovalAction) -> SensitiveSpan {
        SensitiveSpan::new(
            start,
            end,
            SensitiveCategory::Contact,
            action,
            DetectionSource::Model,
        )
        .unwrap()
    }

    #[test]
    fn sentence_split_uses_punctuation_and_pauses() {
        let words = vec![
            word(0, "first", 0, 100).with_display("First"),
            word(1, "sentence", 110, 200).with_display("sentence."),
            word(2, "after", 210, 300),
            word(3, "pause", 1_100, 1_200),
        ];
        let sentences = sentences_from_words(&words, 1_000, 700);
        assert_eq!(sentences.len(), 3);
        assert_eq!(
            (
                sentences[0].start_word_id,
                sentences[0].end_word_id_exclusive
            ),
            (0, 2)
        );
        assert_eq!(
            (
                sentences[1].start_word_id,
                sentences[1].end_word_id_exclusive
            ),
            (2, 3)
        );
    }

    #[test]
    fn default_action_expands_to_the_whole_sentence() {
        let words = vec![
            word(0, "send", 0, 100),
            word(1, "jane", 110, 200),
            word(2, "now", 210, 300),
        ];
        let frozen = transcript(words, 400);
        let sentences = sentences_from_words(frozen.words(), frozen.sample_rate(), 700);
        let plan = plan_removals(
            &frozen,
            &[span(1, 2, RemovalAction::DropSentence)],
            &sentences,
            RemovalSettings {
                padding_ms: 0,
                expand_to_sentence: false,
            },
            &BTreeSet::new(),
            None,
        );
        assert_eq!(plan.removed_word_ids, BTreeSet::from([0, 1, 2]));
        assert_eq!(plan.intervals, vec![SampleInterval::new(0, 300).unwrap()]);
    }

    #[test]
    fn validated_word_cut_does_not_expand_without_padding() {
        let words = vec![
            word(0, "send", 0, 100),
            word(1, "jane", 110, 200),
            word(2, "now", 210, 300),
        ];
        let frozen = transcript(words, 400);
        let sentences = sentences_from_words(frozen.words(), frozen.sample_rate(), 700);
        let plan = plan_removals(
            &frozen,
            &[span(1, 2, RemovalAction::DropWords)],
            &sentences,
            RemovalSettings {
                padding_ms: 0,
                expand_to_sentence: false,
            },
            &BTreeSet::new(),
            None,
        );
        assert_eq!(plan.removed_word_ids, BTreeSet::from([1]));
        assert_eq!(plan.intervals, vec![SampleInterval::new(110, 200).unwrap()]);
    }

    #[test]
    fn padding_pull_in_reaches_a_true_fixed_point() {
        let words = (0..70)
            .map(|id| word(id, "word", id * 100, id * 100 + 100))
            .collect();
        let frozen = transcript(words, 7_000);
        let plan = plan_removals(
            &frozen,
            &[span(35, 36, RemovalAction::DropWords)],
            &[],
            RemovalSettings {
                padding_ms: 1,
                expand_to_sentence: false,
            },
            &BTreeSet::new(),
            None,
        );
        assert_eq!(plan.removed_word_ids.len(), 70);
        assert!(plan.iterations > 32);
        assert_eq!(plan.intervals, vec![SampleInterval::new(0, 7_000).unwrap()]);
    }

    #[test]
    fn retained_word_must_fit_entirely_inside_the_clip() {
        let frozen = transcript(
            vec![word(0, "edge", 50, 150), word(1, "inside", 200, 300)],
            400,
        );
        assert_eq!(
            words_fully_inside(
                &frozen,
                SampleInterval::new(100, 350).unwrap(),
                &BTreeSet::new(),
            ),
            vec![1]
        );
    }
}
