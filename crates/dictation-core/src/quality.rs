//! Explicit quality gates for privacy-filtered training examples.

use std::collections::BTreeSet;

use crate::transcript::FrozenTranscript;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualitySettings {
    pub min_clip_ms: u64,
    pub max_clip_ms: u64,
    pub min_clip_words: usize,
    pub min_total_ms: u64,
    pub max_total_ms: u64,
    pub min_clip_mean_confidence: f32,
    pub min_clip_speech_ratio: f32,
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            min_clip_ms: 1_000,
            max_clip_ms: 30_000,
            min_clip_words: 3,
            min_total_ms: 1_500,
            max_total_ms: 600_000,
            min_clip_mean_confidence: 0.6,
            min_clip_speech_ratio: 0.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipQuality<'a> {
    pub duration_ms: u64,
    pub word_count: usize,
    pub text: &'a str,
    pub mean_confidence: f32,
    pub speech_ratio: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateReason {
    LanguageNotValidated,
    ClipTooShort,
    ClipTooLong,
    ClipTooFewWords,
    ClipEmptyText,
    ClipLowConfidence,
    ClipMostlySilence,
    NoRetainedClips,
    RetainedAudioTooShort,
    RetainedAudioTooLong,
    EmptyTranscript,
    NoAudio,
    SensitiveAfterRemoval,
    DuplicateExample,
    DailyPackageCapReached,
    DailyDurationCapReached,
    ClipNoSpeechDetected,
    UnexplainedSpeechInClip,
}

impl GateReason {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::LanguageNotValidated => "language_not_validated",
            Self::ClipTooShort => "clip_too_short",
            Self::ClipTooLong => "clip_too_long",
            Self::ClipTooFewWords => "clip_too_few_words",
            Self::ClipEmptyText => "clip_empty_text",
            Self::ClipLowConfidence => "clip_low_confidence",
            Self::ClipMostlySilence => "clip_mostly_silence",
            Self::NoRetainedClips => "no_retained_clips",
            Self::RetainedAudioTooShort => "retained_audio_too_short",
            Self::RetainedAudioTooLong => "retained_audio_too_long",
            Self::EmptyTranscript => "empty_transcript",
            Self::NoAudio => "no_audio",
            Self::SensitiveAfterRemoval => "sensitive_after_removal",
            Self::DuplicateExample => "duplicate_example",
            Self::DailyPackageCapReached => "daily_package_cap_reached",
            Self::DailyDurationCapReached => "daily_duration_cap_reached",
            Self::ClipNoSpeechDetected => "clip_no_speech_detected",
            Self::UnexplainedSpeechInClip => "unexplained_speech_in_clip",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateResult {
    pub passed: bool,
    pub reason: Option<GateReason>,
}

impl GateResult {
    pub const OK: Self = Self {
        passed: true,
        reason: None,
    };

    #[must_use]
    pub const fn rejected(reason: GateReason) -> Self {
        Self {
            passed: false,
            reason: Some(reason),
        }
    }
}

#[must_use]
pub fn check_language(language: &str) -> GateResult {
    if language == "en" {
        GateResult::OK
    } else {
        GateResult::rejected(GateReason::LanguageNotValidated)
    }
}

#[must_use]
pub fn check_clip(clip: ClipQuality<'_>, settings: QualitySettings) -> GateResult {
    if clip.duration_ms < settings.min_clip_ms {
        return GateResult::rejected(GateReason::ClipTooShort);
    }
    if clip.duration_ms > settings.max_clip_ms {
        return GateResult::rejected(GateReason::ClipTooLong);
    }
    if clip.word_count < settings.min_clip_words {
        return GateResult::rejected(GateReason::ClipTooFewWords);
    }
    if clip.text.trim().is_empty() {
        return GateResult::rejected(GateReason::ClipEmptyText);
    }
    if !clip.mean_confidence.is_finite() || clip.mean_confidence < settings.min_clip_mean_confidence
    {
        return GateResult::rejected(GateReason::ClipLowConfidence);
    }
    if !clip.speech_ratio.is_finite() || clip.speech_ratio < settings.min_clip_speech_ratio {
        return GateResult::rejected(GateReason::ClipMostlySilence);
    }
    GateResult::OK
}

#[must_use]
pub fn check_session(durations_ms: &[u64], settings: QualitySettings) -> GateResult {
    if durations_ms.is_empty() {
        return GateResult::rejected(GateReason::NoRetainedClips);
    }
    let total = durations_ms
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    if total < settings.min_total_ms {
        return GateResult::rejected(GateReason::RetainedAudioTooShort);
    }
    if total > settings.max_total_ms {
        return GateResult::rejected(GateReason::RetainedAudioTooLong);
    }
    GateResult::OK
}

#[must_use]
pub fn check_transcript_usable(transcript: &FrozenTranscript) -> GateResult {
    if transcript.words().is_empty() {
        return GateResult::rejected(GateReason::EmptyTranscript);
    }
    if transcript.span().is_empty() {
        return GateResult::rejected(GateReason::NoAudio);
    }
    GateResult::OK
}

#[must_use]
pub const fn check_retained_text_clean(clean: bool) -> GateResult {
    if clean {
        GateResult::OK
    } else {
        GateResult::rejected(GateReason::SensitiveAfterRemoval)
    }
}

#[must_use]
pub fn check_duplicate(content_hash: &str, seen: &BTreeSet<String>) -> GateResult {
    if seen.contains(content_hash) {
        GateResult::rejected(GateReason::DuplicateExample)
    } else {
        GateResult::OK
    }
}

#[must_use]
pub const fn check_volume_cap(
    packages_today: u32,
    seconds_today: u64,
    max_packages: u32,
    max_seconds: u64,
) -> GateResult {
    if packages_today >= max_packages {
        GateResult::rejected(GateReason::DailyPackageCapReached)
    } else if seconds_today >= max_seconds {
        GateResult::rejected(GateReason::DailyDurationCapReached)
    } else {
        GateResult::OK
    }
}

#[must_use]
pub fn check_audio_text_consistency(
    aligned_word_samples: u64,
    detected_speech_samples: u64,
    speech_ratio: f32,
    minimum_speech_ratio: f32,
) -> GateResult {
    if !speech_ratio.is_finite() || speech_ratio < minimum_speech_ratio {
        return GateResult::rejected(GateReason::ClipNoSpeechDetected);
    }
    if detected_speech_samples > 0
        && u128::from(aligned_word_samples) * 100 < u128::from(detected_speech_samples) * 45
    {
        return GateResult::rejected(GateReason::UnexplainedSpeechInClip);
    }
    GateResult::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_clip() -> ClipQuality<'static> {
        ClipQuality {
            duration_ms: 2_000,
            word_count: 5,
            text: "meeting starts tomorrow morning now",
            mean_confidence: 0.9,
            speech_ratio: 0.8,
        }
    }

    #[test]
    fn default_quality_accepts_a_good_clip() {
        assert_eq!(
            check_clip(good_clip(), QualitySettings::default()),
            GateResult::OK
        );
    }

    #[test]
    fn non_finite_metrics_fail_closed() {
        let mut clip = good_clip();
        clip.mean_confidence = f32::NAN;
        assert_eq!(
            check_clip(clip, QualitySettings::default()).reason,
            Some(GateReason::ClipLowConfidence)
        );
        clip = good_clip();
        clip.speech_ratio = f32::INFINITY;
        assert_eq!(
            check_clip(clip, QualitySettings::default()).reason,
            Some(GateReason::ClipMostlySilence)
        );
    }

    #[test]
    fn unexplained_speech_is_rejected_without_float_rounding() {
        assert_eq!(
            check_audio_text_consistency(44, 100, 0.8, 0.2).reason,
            Some(GateReason::UnexplainedSpeechInClip)
        );
        assert!(check_audio_text_consistency(45, 100, 0.8, 0.2).passed);
    }

    #[test]
    fn collection_caps_are_enforced_at_the_boundary() {
        assert!(!check_volume_cap(40, 0, 40, 900).passed);
        assert!(!check_volume_cap(0, 900, 40, 900).passed);
        assert!(check_volume_cap(39, 899, 40, 900).passed);
    }
}
