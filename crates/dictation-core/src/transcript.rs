//! Immutable transcript types shared by recognition and privacy filtering.

use std::{fmt, str::FromStr};

use crate::time_map::SampleInterval;

pub const CANONICAL_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitiveCategory {
    DirectIdentifier,
    Contact,
    Address,
    AccountIdentifier,
    GovernmentId,
    Financial,
    CredentialSecret,
    CustomerConfidential,
    Health,
    PoliticalReligious,
    Sexuality,
    SensitiveNarrative,
    ContextualCombination,
    UserDefined,
}

impl SensitiveCategory {
    pub const ALL: [Self; 14] = [
        Self::DirectIdentifier,
        Self::Contact,
        Self::Address,
        Self::AccountIdentifier,
        Self::GovernmentId,
        Self::Financial,
        Self::CredentialSecret,
        Self::CustomerConfidential,
        Self::Health,
        Self::PoliticalReligious,
        Self::Sexuality,
        Self::SensitiveNarrative,
        Self::ContextualCombination,
        Self::UserDefined,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectIdentifier => "direct_identifier",
            Self::Contact => "contact",
            Self::Address => "address",
            Self::AccountIdentifier => "account_identifier",
            Self::GovernmentId => "government_id",
            Self::Financial => "financial",
            Self::CredentialSecret => "credential_secret",
            Self::CustomerConfidential => "customer_confidential",
            Self::Health => "health",
            Self::PoliticalReligious => "political_religious",
            Self::Sexuality => "sexuality",
            Self::SensitiveNarrative => "sensitive_narrative",
            Self::ContextualCombination => "contextual_combination",
            Self::UserDefined => "user_defined",
        }
    }
}

impl FromStr for SensitiveCategory {
    type Err = UnknownEnumValue;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|category| category.as_str() == value)
            .ok_or_else(|| UnknownEnumValue(value.to_owned()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalAction {
    DropSentence,
    DropWords,
}

impl RemovalAction {
    pub const ALL: [Self; 2] = [Self::DropSentence, Self::DropWords];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DropSentence => "drop_sentence",
            Self::DropWords => "drop_words",
        }
    }
}

impl FromStr for RemovalAction {
    type Err = UnknownEnumValue;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|action| action.as_str() == value)
            .ok_or_else(|| UnknownEnumValue(value.to_owned()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionSource {
    Rules,
    Model,
    UserTerms,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownEnumValue(String);

impl fmt::Display for UnknownEnumValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown enum value {:?}", self.0)
    }
}

impl std::error::Error for UnknownEnumValue {}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    id: u64,
    text: String,
    display: String,
    interval: SampleInterval,
    confidence: f32,
    timing_unreliable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptError {
    InvalidSampleRate,
    InvertedWordTiming { word_id: u64 },
    InvalidConfidence { word_id: u64 },
    NonIncreasingWordId { word_id: u64 },
}

impl fmt::Display for TranscriptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleRate => write!(formatter, "sample rate must be positive"),
            Self::InvertedWordTiming { word_id } => {
                write!(formatter, "word {word_id} has inverted timing")
            }
            Self::InvalidConfidence { word_id } => {
                write!(formatter, "word {word_id} confidence is outside [0, 1]")
            }
            Self::NonIncreasingWordId { word_id } => {
                write!(formatter, "word id {word_id} is not strictly increasing")
            }
        }
    }
}

impl std::error::Error for TranscriptError {}

impl Word {
    /// Creates one immutable word aligned to the canonical sample stream.
    ///
    /// # Errors
    ///
    /// Returns an error for inverted timing or a non-finite/out-of-range
    /// confidence value.
    pub fn new(
        id: u64,
        text: impl Into<String>,
        start_sample: u64,
        end_sample: u64,
        confidence: f32,
    ) -> Result<Self, TranscriptError> {
        let interval = SampleInterval::new(start_sample, end_sample)
            .map_err(|_| TranscriptError::InvertedWordTiming { word_id: id })?;
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(TranscriptError::InvalidConfidence { word_id: id });
        }
        Ok(Self {
            id,
            text: text.into(),
            display: String::new(),
            interval,
            confidence,
            timing_unreliable: false,
        })
    }

    #[must_use]
    pub fn with_display(mut self, display: impl Into<String>) -> Self {
        self.display = display.into();
        self
    }

    #[must_use]
    pub const fn with_unreliable_timing(mut self, unreliable: bool) -> Self {
        self.timing_unreliable = unreliable;
        self
    }

    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub fn display_text(&self) -> &str {
        if self.display.is_empty() {
            &self.text
        } else {
            &self.display
        }
    }

    #[must_use]
    pub const fn interval(&self) -> SampleInterval {
        self.interval
    }

    #[must_use]
    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    #[must_use]
    pub const fn timing_unreliable(&self) -> bool {
        self.timing_unreliable
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrozenTranscript {
    session_id: String,
    revision: u64,
    words: Vec<Word>,
    sample_rate: u32,
    total_samples: u64,
    language: String,
}

impl FrozenTranscript {
    /// Freezes a transcript revision after validating stable word identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero sample rate or duplicate/out-of-order word
    /// identifiers.
    pub fn new(
        session_id: impl Into<String>,
        revision: u64,
        words: Vec<Word>,
        sample_rate: u32,
        total_samples: u64,
        language: impl Into<String>,
    ) -> Result<Self, TranscriptError> {
        if sample_rate == 0 {
            return Err(TranscriptError::InvalidSampleRate);
        }
        for pair in words.windows(2) {
            if pair[1].id <= pair[0].id {
                return Err(TranscriptError::NonIncreasingWordId {
                    word_id: pair[1].id,
                });
            }
        }
        Ok(Self {
            session_id: session_id.into(),
            revision,
            words,
            sample_rate,
            total_samples,
            language: language.into(),
        })
    }

    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    #[must_use]
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    #[must_use]
    pub fn span(&self) -> SampleInterval {
        let last_word_end = self.words.last().map_or(0, |word| word.interval.end());
        SampleInterval::from_zero(self.total_samples.max(last_word_end))
    }

    #[must_use]
    pub fn word_by_id(&self, id: u64) -> Option<&Word> {
        self.words
            .binary_search_by_key(&id, |word| word.id)
            .ok()
            .map(|index| &self.words[index])
    }

    #[must_use]
    pub fn spoken_text(&self) -> String {
        self.words
            .iter()
            .map(Word::text)
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[must_use]
    pub fn display_text(&self) -> String {
        let mut output = String::new();
        for word in &self.words {
            let token = word.display_text();
            if !output.is_empty() && !token.starts_with([',', '.', '!', '?', ';', ':']) {
                output.push(' ');
            }
            output.push_str(token);
        }
        output.trim().to_owned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SensitiveSpan {
    pub start_word_id: u64,
    pub end_word_id_exclusive: u64,
    pub category: SensitiveCategory,
    pub action: RemovalAction,
    pub source: DetectionSource,
}

impl SensitiveSpan {
    /// Creates a half-open word-ID range.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is empty or inverted.
    pub fn new(
        start_word_id: u64,
        end_word_id_exclusive: u64,
        category: SensitiveCategory,
        action: RemovalAction,
        source: DetectionSource,
    ) -> Result<Self, &'static str> {
        if end_word_id_exclusive <= start_word_id {
            return Err("sensitive span must be non-empty");
        }
        Ok(Self {
            start_word_id,
            end_word_id_exclusive,
            category,
            action,
            source,
        })
    }

    #[must_use]
    pub const fn is_mandatory(&self) -> bool {
        matches!(
            self.source,
            DetectionSource::Rules | DetectionSource::UserTerms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_transcript_rejects_unstable_ids() {
        let words = vec![
            Word::new(3, "first", 0, 100, 0.9).unwrap(),
            Word::new(3, "again", 100, 200, 0.9).unwrap(),
        ];
        assert_eq!(
            FrozenTranscript::new("session", 1, words, 16_000, 200, "en"),
            Err(TranscriptError::NonIncreasingWordId { word_id: 3 })
        );
    }

    #[test]
    fn spoken_and_display_forms_stay_separate() {
        let words = vec![
            Word::new(1, "hello", 0, 100, 1.0)
                .unwrap()
                .with_display("Hello"),
            Word::new(2, "world", 110, 200, 1.0)
                .unwrap()
                .with_display("world!"),
        ];
        let transcript = FrozenTranscript::new("session", 2, words, 16_000, 220, "en").unwrap();
        assert_eq!(transcript.spoken_text(), "hello world");
        assert_eq!(transcript.display_text(), "Hello world!");
        assert_eq!(transcript.span(), SampleInterval::new(0, 220).unwrap());
    }

    #[test]
    fn invalid_confidence_is_rejected() {
        assert!(Word::new(1, "word", 0, 10, f32::NAN).is_err());
        assert!(Word::new(1, "word", 0, 10, 1.1).is_err());
    }
}
