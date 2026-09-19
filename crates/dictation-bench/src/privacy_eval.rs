//! Span-recall evaluation of the full privacy pipeline (plan §12).
//!
//! Each case is a synthetic transcript with annotated sensitive phrases. A
//! span counts as caught only if every one of its words is removed from the
//! training copy (after sentence expansion and padding pull-in). A session
//! the pipeline refuses entirely also catches its spans, but is reported
//! separately because it retains nothing useful. Over-redaction is measured
//! on words outside every annotated span.

use std::collections::{BTreeMap, BTreeSet};

use dictation_core::{
    privacy::{Classifier, ClassifierFailure, PrivacySettings, analyze},
    removal::{RemovalSettings, plan_removals},
    transcript::{FrozenTranscript, Word},
};
use serde_json::{Value, json};

#[derive(Debug)]
pub struct Case {
    pub id: String,
    pub suite: String,
    pub text: String,
    pub spans: Vec<(String, String)>,
}

/// Parses the JSONL corpus.
///
/// # Errors
///
/// Returns a message for malformed lines.
pub fn load(text: &str) -> Result<Vec<Case>, String> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
            Ok(Case {
                id: value["id"].as_str().unwrap_or_default().to_owned(),
                suite: value["suite"].as_str().unwrap_or_default().to_owned(),
                text: value["text"].as_str().unwrap_or_default().to_owned(),
                spans: value["sensitive"]
                    .as_array()
                    .map(|spans| {
                        spans
                            .iter()
                            .map(|span| (span["phrase"].as_str().unwrap_or_default().to_owned(), span["category"].as_str().unwrap_or_default().to_owned()))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        })
        .collect()
}

/// Synthetic timing: 0.4 s per word, 0.8 s pause after sentence punctuation.
fn transcript(case: &Case) -> Result<FrozenTranscript, String> {
    let mut words = Vec::new();
    let mut cursor = 1_600_u64;
    for (index, token) in case.text.split_whitespace().enumerate() {
        let spoken = dictation_core::asr::normalize_spoken(token);
        let end = cursor + 6_400;
        words.push(Word::new(index as u64, spoken, cursor, end, 0.95).map_err(|error| error.to_string())?.with_display(token));
        cursor = end + if token.ends_with(['.', '?', '!']) { 12_800 } else { 800 };
    }
    FrozenTranscript::new(case.id.clone(), 1, words, 16_000, cursor + 1_600, "en").map_err(|error| error.to_string())
}

fn phrase_ids(case: &Case, phrase: &str) -> Result<BTreeSet<u64>, String> {
    let tokens: Vec<&str> = case.text.split_whitespace().collect();
    let wanted: Vec<&str> = phrase.split_whitespace().collect();
    let strip = |token: &str| token.trim_end_matches([',', '.', '?', '!']).to_owned();
    for start in 0..=tokens.len().saturating_sub(wanted.len()) {
        if tokens[start..start + wanted.len()].iter().zip(&wanted).all(|(a, b)| strip(a) == strip(b)) {
            return Ok((start as u64..(start + wanted.len()) as u64).collect());
        }
    }
    Err(format!("{}: phrase {phrase:?} not found", case.id))
}

/// Wilson score interval for `hits / total` at 95%.
#[must_use]
pub fn wilson(hits: usize, total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let (n, p) = (total as f64, hits as f64 / total as f64);
    let z = 1.959_964_f64;
    let center = (p + z * z / (2.0 * n)) / (1.0 + z * z / n);
    let margin = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / (1.0 + z * z / n);
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// A classifier that finds nothing, to measure the rules alone.
pub struct NoModel;

impl Classifier for NoModel {
    fn model_id(&self) -> String {
        "rules-only".to_owned()
    }
    fn classify(&mut self, _: &str, _: &str, _: &str, _: u32) -> Result<String, ClassifierFailure> {
        Ok(r#"{"spans":[],"uncertain":false}"#.to_owned())
    }
}

#[derive(Default)]
struct Tally {
    hits: usize,
    total: usize,
}

/// Runs every case and returns the report.
///
/// # Errors
///
/// Returns a message when the corpus is inconsistent.
#[allow(clippy::too_many_lines)]
pub fn evaluate(cases: &[Case], classifier: &mut impl Classifier, progress: impl Fn(usize, &str)) -> Result<Value, String> {
    let mut by_suite: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_category: BTreeMap<String, Tally> = BTreeMap::new();
    let mut misses = Vec::new();
    let mut refused_sessions = Vec::new();
    let (mut benign_words, mut benign_removed) = (0_usize, 0_usize);
    let mut benign_sessions_refused = 0;
    for (index, case) in cases.iter().enumerate() {
        progress(index, &case.id);
        let transcript = transcript(case)?;
        let analysis = analyze(&transcript, classifier, PrivacySettings::default(), &[]);
        let refused = analysis.rejected_reason.is_some() || analysis.uncertain;
        let removed: BTreeSet<u64> = if refused {
            transcript.words().iter().map(Word::id).collect()
        } else {
            plan_removals(&transcript, &analysis.spans, &analysis.sentences, RemovalSettings::default(), &BTreeSet::new(), None).removed_word_ids
        };
        if refused {
            refused_sessions.push(json!({ "id": case.id, "reason": analysis.rejected_reason.unwrap_or("uncertain") }));
        }
        for (phrase, category) in &case.spans {
            let ids = phrase_ids(case, phrase)?;
            let caught = ids.is_subset(&removed);
            for tally in [by_suite.entry(case.suite.clone()).or_default(), by_category.entry(category.clone()).or_default()] {
                tally.total += 1;
                tally.hits += usize::from(caught);
            }
            if !caught {
                misses.push(json!({ "id": case.id, "suite": case.suite, "category": category }));
            }
        }
        if case.suite == "benign" {
            benign_words += transcript.words().len();
            benign_removed += removed.len();
            benign_sessions_refused += usize::from(refused);
        }
    }
    let summarize = |tallies: &BTreeMap<String, Tally>| -> Value {
        tallies
            .iter()
            .map(|(name, tally)| {
                let (low, high) = wilson(tally.hits, tally.total);
                #[allow(clippy::cast_precision_loss)]
                let recall = tally.hits as f64 / tally.total.max(1) as f64;
                (name.clone(), json!({ "caught": tally.hits, "total": tally.total, "recall": recall, "ci95": [low, high] }))
            })
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    #[allow(clippy::cast_precision_loss)]
    let over_redaction = benign_removed as f64 / benign_words.max(1) as f64;
    let high_risk = by_suite.get("high_risk").map_or((0, 0), |tally| (tally.hits, tally.total));
    let secrets = by_suite.get("secrets").map_or((0, 0), |tally| (tally.hits, tally.total));
    Ok(json!({
        "cases": cases.len(),
        "by_suite": summarize(&by_suite),
        "by_category": summarize(&by_category),
        "misses": misses,
        "refused_sessions": refused_sessions,
        "benign_over_redaction_word_fraction": over_redaction,
        "benign_sessions_refused": benign_sessions_refused,
        "gates": {
            "high_risk_recall_at_least_99pct": high_risk.1 > 0 && high_risk.0 * 100 >= high_risk.1 * 99,
            "high_risk_lower_ci_bound": wilson(high_risk.0, high_risk.1).0,
            "secrets_zero_misses": secrets.1 > 0 && secrets.0 == secrets.1,
            "note": "Synthetic text only: no ASR errors, accents, noise, or listening tests. Passing here is necessary, not sufficient.",
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_phrases_all_resolve_and_rules_catch_every_secret() {
        let cases = load(include_str!("../../../eval/privacy_benchmark.jsonl")).unwrap();
        for case in &cases {
            for (phrase, _) in &case.spans {
                phrase_ids(case, phrase).unwrap();
            }
        }
        let report = evaluate(&cases, &mut NoModel, |_, _| {}).unwrap();
        let secrets = &report["by_suite"]["secrets"];
        assert_eq!(secrets["caught"], secrets["total"], "{}", report["misses"]);
    }

    #[test]
    fn wilson_interval_is_sane() {
        let (low, high) = wilson(99, 100);
        assert!(low > 0.94 && low < 0.99 && high > 0.99);
    }
}
