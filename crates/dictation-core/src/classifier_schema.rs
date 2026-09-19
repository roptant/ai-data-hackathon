//! Strict validation for privacy-classifier JSON output.
//!
//! Grammar-constrained decoding improves structure, but this module still
//! validates every field against the exact frozen transcript window shown to
//! the worker. Any parse or schema failure rejects the training copy.

use std::{collections::BTreeSet, fmt, str::FromStr};

use serde_json::{Map, Value};

use crate::transcript::{DetectionSource, RemovalAction, SensitiveCategory, SensitiveSpan};

pub const MAX_SPANS_PER_WINDOW: usize = 64;

pub const GBNF_GRAMMAR: &str = r#"
root ::= "{" ws "\"spans\"" ws ":" ws spans ws "," ws "\"uncertain\"" ws ":" ws bool ws "}"
spans ::= "[" ws "]" | "[" ws span (ws "," ws span)* ws "]"
span ::= ( "{" ws "\"start_word_id\"" ws ":" ws int ws ","
  ws "\"end_word_id_exclusive\"" ws ":" ws int ws ","
  ws "\"category\"" ws ":" ws category ws ","
  ws "\"action\"" ws ":" ws action ws "}" )
category ::= ( "\"direct_identifier\"" | "\"contact\"" | "\"address\"" |
  "\"account_identifier\"" | "\"government_id\"" | "\"financial\"" |
  "\"credential_secret\"" | "\"customer_confidential\"" | "\"health\"" |
  "\"political_religious\"" | "\"sexuality\"" | "\"sensitive_narrative\"" |
  "\"contextual_combination\"" | "\"user_defined\"" )
action ::= "\"drop_sentence\"" | "\"drop_words\""
bool ::= "true" | "false"
int ::= [0-9] | [1-9] [0-9]{0,8}
ws ::= [ \t\n]{0,8}
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifierOutput {
    pub spans: Vec<SensitiveSpan>,
    pub uncertain: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaViolation(String);

impl SchemaViolation {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for SchemaViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SchemaViolation {}

/// Parses and validates one classifier window response.
///
/// # Errors
///
/// Returns [`SchemaViolation`] for malformed JSON, unknown fields or enum
/// values, out-of-window IDs, unordered spans, and excessive span counts.
pub fn validate_output(
    raw: &str,
    allowed_word_ids: &BTreeSet<u64>,
    allow_word_level_cuts: bool,
    max_spans: usize,
) -> Result<ClassifierOutput, SchemaViolation> {
    let payload = parse_object(raw)?;
    reject_unknown_keys(&payload, &["spans", "uncertain"], "classifier output")?;

    let uncertain = payload
        .get("uncertain")
        .and_then(Value::as_bool)
        .ok_or_else(|| SchemaViolation::new("'uncertain' must be present and boolean"))?;
    let items = payload
        .get("spans")
        .and_then(Value::as_array)
        .ok_or_else(|| SchemaViolation::new("'spans' must be present and an array"))?;
    if items.len() > max_spans {
        return Err(SchemaViolation::new(format!(
            "too many spans: {} > {max_spans}",
            items.len()
        )));
    }
    if allowed_word_ids.is_empty() && !items.is_empty() {
        return Err(SchemaViolation::new(
            "classifier reported spans for an empty window",
        ));
    }

    let mut spans = Vec::with_capacity(items.len());
    let mut previous_start = None;
    for (index, value) in items.iter().enumerate() {
        let item = value
            .as_object()
            .ok_or_else(|| SchemaViolation::new(format!("span {index} is not an object")))?;
        reject_unknown_keys(
            item,
            &[
                "start_word_id",
                "end_word_id_exclusive",
                "category",
                "action",
            ],
            &format!("span {index}"),
        )?;
        let start = required_u64(item, "start_word_id", index)?;
        let end = required_u64(item, "end_word_id_exclusive", index)?;
        if end <= start {
            return Err(SchemaViolation::new(format!(
                "span {index}: empty or inverted range [{start}, {end})"
            )));
        }
        if previous_start.is_some_and(|previous| start < previous) {
            return Err(SchemaViolation::new(format!(
                "span {index}: spans must be ordered by start_word_id"
            )));
        }
        let width = end - start;
        let available = u64::try_from(allowed_word_ids.len()).unwrap_or(u64::MAX);
        if width > available || !(start..end).all(|word_id| allowed_word_ids.contains(&word_id)) {
            return Err(SchemaViolation::new(format!(
                "span {index}: range [{start}, {end}) contains an ID not shown in the window"
            )));
        }
        previous_start = Some(start);

        let category_value = required_string(item, "category", index)?;
        let category = SensitiveCategory::from_str(category_value).map_err(|_| {
            SchemaViolation::new(format!("span {index}: unknown category {category_value:?}"))
        })?;
        let action_value = required_string(item, "action", index)?;
        let mut action = RemovalAction::from_str(action_value).map_err(|_| {
            SchemaViolation::new(format!("span {index}: unknown action {action_value:?}"))
        })?;
        if action == RemovalAction::DropWords && !allow_word_level_cuts {
            action = RemovalAction::DropSentence;
        }
        spans.push(
            SensitiveSpan::new(start, end, category, action, DetectionSource::Model)
                .map_err(SchemaViolation::new)?,
        );
    }
    Ok(ClassifierOutput { spans, uncertain })
}

fn parse_object(raw: &str) -> Result<Map<String, Value>, SchemaViolation> {
    let mut text = raw.trim();
    if text.is_empty() {
        return Err(SchemaViolation::new(
            "classifier returned an empty response",
        ));
    }
    let unfenced;
    if text.starts_with("```") {
        unfenced = text
            .lines()
            .filter(|line| !line.trim().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n");
        text = unfenced.trim();
    }
    let value: Value = serde_json::from_str(text).map_err(|error| {
        SchemaViolation::new(format!("classifier output is not valid JSON: {error}"))
    })?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| SchemaViolation::new("classifier output must be a JSON object"))
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    location: &str,
) -> Result<(), SchemaViolation> {
    let unknown: Vec<_> = object
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(SchemaViolation::new(format!(
            "{location} has unexpected keys: {unknown:?}"
        )))
    }
}

fn required_u64(
    item: &Map<String, Value>,
    name: &str,
    index: usize,
) -> Result<u64, SchemaViolation> {
    item.get(name).and_then(Value::as_u64).ok_or_else(|| {
        SchemaViolation::new(format!("span {index}: {name} must be an unsigned integer"))
    })
}

fn required_string<'a>(
    item: &'a Map<String, Value>,
    name: &str,
    index: usize,
) -> Result<&'a str, SchemaViolation> {
    item.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaViolation::new(format!("span {index}: {name} must be a string")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> BTreeSet<u64> {
        (10..20).collect()
    }

    fn validate(raw: &str) -> Result<ClassifierOutput, SchemaViolation> {
        validate_output(raw, &ids(), false, MAX_SPANS_PER_WINDOW)
    }

    #[test]
    fn empty_complete_answer_is_valid() {
        let output = validate(r#"{"spans":[],"uncertain":false}"#).unwrap();
        assert!(output.spans.is_empty());
        assert!(!output.uncertain);
    }

    #[test]
    fn valid_span_is_converted_and_word_cut_is_downgraded() {
        let output = validate(
            r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":14,"category":"contact","action":"drop_words"}],"uncertain":true}"#,
        )
        .unwrap();
        assert!(output.uncertain);
        assert_eq!(output.spans[0].category, SensitiveCategory::Contact);
        assert_eq!(output.spans[0].action, RemovalAction::DropSentence);
        assert_eq!(output.spans[0].source, DetectionSource::Model);
    }

    #[test]
    fn word_cut_can_be_explicitly_enabled() {
        let output = validate_output(
            r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":14,"category":"contact","action":"drop_words"}],"uncertain":false}"#,
            &ids(),
            true,
            MAX_SPANS_PER_WINDOW,
        )
        .unwrap();
        assert_eq!(output.spans[0].action, RemovalAction::DropWords);
    }

    #[test]
    fn fenced_json_is_accepted() {
        let output = validate("```json\n{\"spans\":[],\"uncertain\":true}\n```").unwrap();
        assert!(output.uncertain);
    }

    #[test]
    fn incomplete_or_extra_shapes_are_rejected() {
        for raw in [
            "",
            "[]",
            r#"{"spans":[]}"#,
            r#"{"uncertain":false}"#,
            r#"{"spans":[],"uncertain":"no"}"#,
            r#"{"spans":{},"uncertain":false}"#,
            r#"{"spans":[],"uncertain":false,"extra":1}"#,
        ] {
            assert!(validate(raw).is_err(), "unexpectedly accepted {raw}");
        }
    }

    #[test]
    fn invalid_ranges_and_unknown_enums_are_rejected() {
        for raw in [
            r#"{"spans":[{"start_word_id":9,"end_word_id_exclusive":12,"category":"contact","action":"drop_sentence"}],"uncertain":false}"#,
            r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":12,"category":"contact","action":"drop_sentence"}],"uncertain":false}"#,
            r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":14,"category":"vibes","action":"drop_sentence"}],"uncertain":false}"#,
            r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":14,"category":"contact","action":"paraphrase"}],"uncertain":false}"#,
        ] {
            assert!(validate(raw).is_err(), "unexpectedly accepted {raw}");
        }
    }

    #[test]
    fn missing_interior_word_id_is_rejected() {
        let sparse_ids = BTreeSet::from([12, 14]);
        let raw = r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":15,"category":"contact","action":"drop_sentence"}],"uncertain":false}"#;
        assert!(validate_output(raw, &sparse_ids, false, MAX_SPANS_PER_WINDOW).is_err());
    }

    #[test]
    fn absurd_range_is_rejected_without_iterating_over_it() {
        let raw = r#"{"spans":[{"start_word_id":12,"end_word_id_exclusive":18446744073709551615,"category":"contact","action":"drop_sentence"}],"uncertain":false}"#;
        assert!(validate(raw).is_err());
    }

    #[test]
    fn boolean_and_string_ids_are_rejected() {
        for value in ["true", r#""12""#] {
            let raw = format!(
                r#"{{"spans":[{{"start_word_id":{value},"end_word_id_exclusive":14,"category":"contact","action":"drop_sentence"}}],"uncertain":false}}"#
            );
            assert!(validate(&raw).is_err());
        }
    }

    #[test]
    fn unordered_and_excessive_spans_are_rejected() {
        let unordered = r#"{"spans":[{"start_word_id":16,"end_word_id_exclusive":18,"category":"contact","action":"drop_sentence"},{"start_word_id":12,"end_word_id_exclusive":14,"category":"contact","action":"drop_sentence"}],"uncertain":false}"#;
        assert!(validate(unordered).is_err());

        let item = r#"{"start_word_id":10,"end_word_id_exclusive":11,"category":"contact","action":"drop_sentence"}"#;
        let raw = format!(
            "{{\"spans\":[{}],\"uncertain\":false}}",
            std::iter::repeat_n(item, MAX_SPANS_PER_WINDOW + 1)
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(validate(&raw).is_err());
    }

    #[test]
    fn grammar_contains_every_enum_literal() {
        for category in SensitiveCategory::ALL {
            assert!(GBNF_GRAMMAR.contains(category.as_str()));
        }
        for action in RemovalAction::ALL {
            assert!(GBNF_GRAMMAR.contains(action.as_str()));
        }
    }
}
