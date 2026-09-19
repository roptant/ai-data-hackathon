//! Word error rate for model evaluation (plan §10, §12).

use crate::asr::normalize_spoken;

/// Normalized spoken words: lowercase, punctuation stripped.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalize_spoken)
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word-level Levenshtein distance.
#[must_use]
pub fn edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0; hypothesis.len() + 1];
    for (row, reference_word) in reference.iter().enumerate() {
        current[0] = row + 1;
        for (column, hypothesis_word) in hypothesis.iter().enumerate() {
            let substitution = previous[column] + usize::from(reference_word != hypothesis_word);
            current[column + 1] = substitution
                .min(previous[column + 1] + 1)
                .min(current[column] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

/// Corpus WER: total edits over total reference words.
#[must_use]
pub fn corpus_wer<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<f64> {
    let (mut edits, mut total) = (0_usize, 0_usize);
    for (reference, hypothesis) in pairs {
        let reference = words(reference);
        edits += edit_distance(&reference, &words(hypothesis));
        total += reference.len();
    }
    #[allow(clippy::cast_precision_loss)]
    (total > 0).then(|| edits as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_counts_substitutions_insertions_and_deletions() {
        assert_eq!(corpus_wer([("the cat sat", "The cat sat.")]), Some(0.0));
        assert_eq!(corpus_wer([("the cat sat", "the bat sat")]), Some(1.0 / 3.0));
        assert_eq!(corpus_wer([("the cat sat", "the cat")]), Some(1.0 / 3.0));
        assert_eq!(corpus_wer([("a b", "a x b y")]), Some(1.0));
        assert_eq!(corpus_wer([("", "noise")]), None);
    }
}
