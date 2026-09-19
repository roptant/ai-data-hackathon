//! Training-copy construction (plan §7 steps 5–10).
//!
//! Given a frozen transcript, its canonical audio, and a completed privacy
//! analysis, this yields either eligible contiguous clips or a rejection code.
//! It never yields "probably fine". Invariants, each covered by tests:
//!
//! * no sample inside a removal interval appears in any clip;
//! * every retained word lies wholly inside its clip's audio;
//! * clips are contiguous source intervals; the spliced export's boundary
//!   manifest stays local;
//! * any failed stage rejects the session.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::{
    privacy::{
        PrivacyAnalysis, PrivacySettings, SESSION_FATAL_ALIGNMENT, sentences_with_unusable_words,
        validate_alignment,
    },
    quality::{
        ClipQuality, QualitySettings, check_audio_text_consistency, check_clip, check_duplicate,
        check_language, check_retained_text_clean, check_session, check_transcript_usable,
    },
    removal::{RemovalSettings, plan_removals, words_fully_inside},
    time_map::{DestinationMap, SampleInterval, complement},
    transcript::{FrozenTranscript, Word},
    vad::SessionActivity,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RetainedClip {
    pub index: usize,
    /// Half-open interval in the session's canonical samples. Local only.
    pub source: SampleInterval,
    /// Interval in the spliced export timeline. Local only.
    pub destination: SampleInterval,
    pub words: Vec<Word>,
    /// Display text of exactly the retained words, in order.
    pub text: String,
    pub duration_ms: u64,
    pub mean_confidence: f32,
    pub speech_ratio: f32,
    pub pcm: Vec<i16>,
}

impl RetainedClip {
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DatasetMetrics {
    pub duration_ms: u64,
    pub word_count: u64,
    pub mean_confidence: f32,
    pub speech_ratio: f32,
    pub removed_interval_count: u64,
    pub removed_duration_ms: u64,
    pub clip_count: u64,
}

/// Maps a retained source interval to the spliced timeline. Local only: it
/// would explain where deletions happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Boundary {
    pub source: SampleInterval,
    pub destination: SampleInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Eligible,
    Rejected(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuildResult {
    pub decision: Decision,
    pub clips: Vec<RetainedClip>,
    pub removed: Vec<SampleInterval>,
    pub boundaries: Vec<Boundary>,
    pub metrics: DatasetMetrics,
    pub content_hash: String,
    pub clip_rejections: Vec<(usize, &'static str)>,
}

impl BuildResult {
    fn rejected(reason: &'static str) -> Self {
        Self {
            decision: Decision::Rejected(reason),
            clips: Vec::new(),
            removed: Vec::new(),
            boundaries: Vec::new(),
            metrics: DatasetMetrics::default(),
            content_hash: String::new(),
            clip_rejections: Vec::new(),
        }
    }

    #[must_use]
    pub const fn eligible(&self) -> bool {
        matches!(self.decision, Decision::Eligible)
    }

    #[must_use]
    pub const fn rejection(&self) -> Option<&'static str> {
        match self.decision {
            Decision::Eligible => None,
            Decision::Rejected(reason) => Some(reason),
        }
    }

    /// Concatenated retained audio for local playback. Nothing is inserted
    /// between clips: no beeps, silence, or synthetic bridge audio.
    #[must_use]
    pub fn spliced_pcm(&self) -> Vec<i16> {
        self.clips
            .iter()
            .flat_map(|clip| clip.pcm.iter().copied())
            .collect()
    }

    /// One line per clip, so unrelated sentences are not presented as one.
    #[must_use]
    pub fn spliced_text(&self) -> String {
        self.clips
            .iter()
            .map(|clip| clip.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Re-verifies the core invariant against the object about to persist.
    ///
    /// # Errors
    ///
    /// Returns a rejection code when any clip intersects a removal interval or
    /// its audio length disagrees with its source interval.
    pub fn assert_no_removed_audio(&self) -> Result<(), &'static str> {
        for clip in &self.clips {
            if self
                .removed
                .iter()
                .any(|removed| clip.source.intersects(*removed))
            {
                return Err("removed_audio_in_output");
            }
            if clip.pcm.len() as u64 != clip.source.len() {
                return Err("clip_length_mismatch");
            }
            if clip.words.iter().any(|word| {
                word.interval().start() < clip.source.start()
                    || word.interval().end() > clip.source.end()
            }) {
                return Err("word_outside_clip");
            }
        }
        Ok(())
    }
}

/// Post-removal recheck over retained text only (plan §7.9).
pub trait Recheck {
    /// # Errors
    ///
    /// Returns a rejection code when retained text is not clean or the check
    /// could not complete.
    fn verify_clean(&mut self, retained: &FrozenTranscript) -> Result<(), &'static str>;
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DatasetSettings {
    pub quality: QualitySettings,
    pub privacy: PrivacySettings,
    pub removal: RemovalSettings,
}

/// Rounded duration exactly as the package validator computes it.
#[must_use]
pub fn duration_ms(samples: u64, sample_rate: u32) -> u64 {
    samples
        .saturating_mul(1_000)
        .saturating_add(u64::from(sample_rate) / 2)
        / u64::from(sample_rate)
}

fn content_hash(clips: &[RetainedClip]) -> String {
    let mut digest = Sha256::new();
    for clip in clips {
        digest.update(clip.text.as_bytes());
        digest.update([0]);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn display_text(words: &[Word]) -> String {
    words
        .iter()
        .map(Word::display_text)
        .collect::<Vec<_>>()
        .join(" ")
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn mean(values: impl Iterator<Item = (f64, f64)>) -> f32 {
    let (weighted, weight) = values.fold((0.0, 0.0), |(sum, total), (value, weight)| {
        (sum + value * weight, total + weight)
    });
    if weight <= 0.0 {
        0.0
    } else {
        (weighted / weight) as f32
    }
}

/// Builds the training copy for one session.
#[allow(clippy::too_many_lines)]
pub fn build_dataset(
    transcript: &FrozenTranscript,
    analysis: &PrivacyAnalysis,
    pcm: &[i16],
    activity: &SessionActivity,
    settings: DatasetSettings,
    seen_hashes: &BTreeSet<String>,
    recheck: Option<&mut dyn Recheck>,
) -> BuildResult {
    if let Some(reason) = analysis.rejected_reason {
        return BuildResult::rejected(reason);
    }
    if analysis.uncertain {
        return BuildResult::rejected("uncertain_analysis");
    }
    if let Some(reason) = check_transcript_usable(transcript).reason {
        return BuildResult::rejected(reason.code());
    }
    if let Some(reason) = check_language(transcript.language()).reason {
        return BuildResult::rejected(reason.code());
    }
    if pcm.is_empty() {
        return BuildResult::rejected("no_audio");
    }
    if activity.is_effectively_silent() {
        return BuildResult::rejected("session_silent");
    }
    let bound = SampleInterval::from_zero(pcm.len() as u64);
    let problems = validate_alignment(
        transcript.words(),
        bound,
        settings.privacy.min_word_confidence,
    );
    if SESSION_FATAL_ALIGNMENT
        .iter()
        .any(|problem| problems.contains(problem))
    {
        return BuildResult::rejected("alignment_unusable");
    }
    let extra_sentences = sentences_with_unusable_words(
        transcript.words(),
        &analysis.sentences,
        settings.privacy.min_word_confidence,
    );
    let removal = plan_removals(
        transcript,
        &analysis.spans,
        &analysis.sentences,
        settings.removal,
        &extra_sentences,
        Some(bound),
    );
    let retained = complement(&removal.intervals, bound);
    if retained.is_empty() {
        return BuildResult::rejected("fully_redacted");
    }
    let sample_rate = transcript.sample_rate();
    let mut clips = Vec::new();
    let mut clip_rejections = Vec::new();
    for (index, interval) in retained.iter().copied().enumerate() {
        if removal
            .intervals
            .iter()
            .any(|removed| interval.intersects(*removed))
        {
            return BuildResult::rejected("retained_interval_overlaps_removal");
        }
        let words: Vec<Word> =
            words_fully_inside(transcript, interval, &removal.removed_word_ids)
                .into_iter()
                .filter_map(|id| transcript.word_by_id(id).cloned())
                .collect();
        let text = display_text(&words);
        let clip_duration = duration_ms(interval.len(), sample_rate);
        let mean_confidence = mean(
            words
                .iter()
                .map(|word| (f64::from(word.confidence()), 1.0)),
        );
        let speech_ratio = activity.speech_ratio(interval);
        let gate = check_clip(
            ClipQuality {
                duration_ms: clip_duration,
                word_count: words.len(),
                text: &text,
                mean_confidence,
                speech_ratio,
            },
            settings.quality,
        );
        if let Some(reason) = gate.reason {
            clip_rejections.push((index, reason.code()));
            continue;
        }
        let aligned: u64 = words.iter().map(|word| word.interval().len()).sum();
        let consistency = check_audio_text_consistency(
            aligned,
            activity.speech_samples(interval),
            speech_ratio,
            settings.quality.min_clip_speech_ratio,
        );
        if let Some(reason) = consistency.reason {
            clip_rejections.push((index, reason.code()));
            continue;
        }
        let start = usize::try_from(interval.start()).unwrap_or(usize::MAX);
        let end = usize::try_from(interval.end()).unwrap_or(usize::MAX);
        let Some(clip_pcm) = pcm.get(start..end) else {
            return BuildResult::rejected("clip_outside_audio");
        };
        clips.push(RetainedClip {
            index: clips.len(),
            source: interval,
            destination: interval,
            words,
            text,
            duration_ms: clip_duration,
            mean_confidence,
            speech_ratio,
            pcm: clip_pcm.to_vec(),
        });
    }
    let durations: Vec<u64> = clips.iter().map(|clip| clip.duration_ms).collect();
    if let Some(reason) = check_session(&durations, settings.quality).reason {
        return BuildResult::rejected(reason.code());
    }
    // Destination timeline over surviving clips only, so the spliced export's
    // timestamps match the audio it contains.
    let map = DestinationMap::build(clips.iter().map(|clip| clip.source));
    let mut boundaries = Vec::with_capacity(clips.len());
    for (index, clip) in clips.iter_mut().enumerate() {
        let Some(destination) = map.destination_of(index) else {
            return BuildResult::rejected("destination_map_failed");
        };
        clip.destination = destination;
        boundaries.push(Boundary {
            source: clip.source,
            destination,
        });
    }
    if let Some(recheck) = recheck {
        let Some(retained_transcript) = renumbered_transcript(transcript, &clips) else {
            return BuildResult::rejected("recheck_transcript_invalid");
        };
        let verdict = recheck.verify_clean(&retained_transcript);
        if let Err(reason) = verdict {
            return BuildResult::rejected(reason);
        }
        if let Some(reason) = check_retained_text_clean(verdict.is_ok()).reason {
            return BuildResult::rejected(reason.code());
        }
    }
    let digest = content_hash(&clips);
    if let Some(reason) = check_duplicate(&digest, seen_hashes).reason {
        return BuildResult::rejected(reason.code());
    }
    let metrics = DatasetMetrics {
        duration_ms: durations.iter().sum(),
        word_count: clips.iter().map(|clip| clip.word_count() as u64).sum(),
        mean_confidence: mean(clips.iter().map(|clip| {
            #[allow(clippy::cast_precision_loss)]
            let weight = clip.words.len() as f64;
            (f64::from(clip.mean_confidence), weight)
        })),
        speech_ratio: mean(clips.iter().map(|clip| {
            #[allow(clippy::cast_precision_loss)]
            let weight = clip.duration_ms as f64;
            (f64::from(clip.speech_ratio), weight)
        })),
        removed_interval_count: removal.intervals.len() as u64,
        removed_duration_ms: duration_ms(removal.removed_samples(), sample_rate),
        clip_count: clips.len() as u64,
    };
    let result = BuildResult {
        decision: Decision::Eligible,
        clips,
        removed: removal.intervals,
        boundaries,
        metrics,
        content_hash: digest,
        clip_rejections,
    };
    if let Err(reason) = result.assert_no_removed_audio() {
        return BuildResult::rejected(reason);
    }
    result
}

/// Retained words renumbered from zero on the spliced timeline. Original IDs
/// would leak the positions of removed words to the recheck.
fn renumbered_transcript(
    original: &FrozenTranscript,
    clips: &[RetainedClip],
) -> Option<FrozenTranscript> {
    let mut words = Vec::new();
    for clip in clips {
        for word in &clip.words {
            let offset = clip.destination.start();
            let start = word.interval().start() - clip.source.start() + offset;
            let end = word.interval().end() - clip.source.start() + offset;
            let id = words.len() as u64;
            words.push(
                Word::new(id, word.text(), start, end, word.confidence())
                    .ok()?
                    .with_display(word.display_text())
                    .with_unreliable_timing(word.timing_unreliable()),
            );
        }
    }
    let total = clips.iter().map(|clip| clip.source.len()).sum();
    FrozenTranscript::new(
        format!("{}-retained", original.session_id()),
        1,
        words,
        original.sample_rate(),
        total,
        original.language(),
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        privacy::{Classifier, ClassifierFailure, analyze},
        rules::detect,
        vad::VadSettings,
    };

    const RATE: u32 = 16_000;

    /// Loud "speech" inside each word, quiet room elsewhere.
    fn session(text: &str) -> (FrozenTranscript, Vec<i16>) {
        let mut words = Vec::new();
        let mut pcm = vec![20_i16; 8_000];
        for (index, token) in text.split_whitespace().enumerate() {
            let start = pcm.len() as u64;
            pcm.extend((0..6_400).map(|n| if n % 2 == 0 { 9_000 } else { -9_000 }));
            let end = pcm.len() as u64;
            pcm.extend(std::iter::repeat_n(20_i16, 1_600));
            if token.ends_with('.') {
                pcm.extend(std::iter::repeat_n(20_i16, 12_000));
            }
            let spoken = token
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_owned();
            words.push(
                Word::new(index as u64, spoken, start, end, 0.95)
                    .unwrap()
                    .with_display(token),
            );
        }
        pcm.extend(std::iter::repeat_n(20_i16, 8_000));
        let total = pcm.len() as u64;
        (
            FrozenTranscript::new("s", 1, words, RATE, total, "en").unwrap(),
            pcm,
        )
    }

    struct Clean;
    impl Classifier for Clean {
        fn model_id(&self) -> String {
            "clean".to_owned()
        }
        fn classify(&mut self, _: &str, _: &str, _: &str, _: u32) -> Result<String, ClassifierFailure> {
            Ok(r#"{"spans":[],"uncertain":false}"#.to_owned())
        }
    }

    struct RulesOnlyRecheck;
    impl Recheck for RulesOnlyRecheck {
        fn verify_clean(&mut self, retained: &FrozenTranscript) -> Result<(), &'static str> {
            if detect(retained.words(), &[]).is_clean() {
                Ok(())
            } else {
                Err("sensitive_after_removal")
            }
        }
    }

    fn build(text: &str) -> (FrozenTranscript, BuildResult) {
        let (transcript, pcm) = session(text);
        let analysis = analyze(&transcript, &mut Clean, PrivacySettings::default(), &[]);
        let activity = SessionActivity::analyze(&pcm, RATE, VadSettings::default());
        let result = build_dataset(
            &transcript,
            &analysis,
            &pcm,
            &activity,
            DatasetSettings::default(),
            &BTreeSet::new(),
            Some(&mut RulesOnlyRecheck),
        );
        (transcript, result)
    }

    #[test]
    fn plan_example_drops_the_address_sentence_only() {
        let (transcript, result) =
            build("Send the parcel to Jane at 14 Oak Street. The meeting starts tomorrow.");
        assert!(result.eligible(), "{:?}", result.decision);
        assert_eq!(result.spliced_text(), "The meeting starts tomorrow.");
        let first_sentence_end = transcript.word_by_id(8).unwrap().interval().end();
        for clip in &result.clips {
            assert!(clip.source.start() >= first_sentence_end);
        }
        result.assert_no_removed_audio().unwrap();
        assert_eq!(result.metrics.clip_count, 1);
    }

    #[test]
    fn fully_sensitive_session_is_rejected() {
        let (_, result) = build("My password is hunter2 secret.");
        // Only word-free silence survives the cut, and it fails the clip gates.
        assert!(matches!(
            result.rejection(),
            Some("fully_redacted" | "no_retained_clips")
        ));
        assert!(result.clips.is_empty());
    }

    #[test]
    fn uncertain_or_failed_analysis_rejects_before_any_cut() {
        let (transcript, pcm) = session("The meeting starts tomorrow morning.");
        let activity = SessionActivity::analyze(&pcm, RATE, VadSettings::default());
        let mut analysis = analyze(&transcript, &mut Clean, PrivacySettings::default(), &[]);
        analysis.uncertain = true;
        let result = build_dataset(
            &transcript,
            &analysis,
            &pcm,
            &activity,
            DatasetSettings::default(),
            &BTreeSet::new(),
            None,
        );
        assert_eq!(result.rejection(), Some("uncertain_analysis"));
    }

    #[test]
    fn silent_audio_never_becomes_an_example() {
        let (transcript, _) = session("The meeting starts tomorrow morning.");
        let silence = vec![0_i16; transcript.total_samples() as usize];
        let analysis = analyze(&transcript, &mut Clean, PrivacySettings::default(), &[]);
        let activity = SessionActivity::analyze(&silence, RATE, VadSettings::default());
        let result = build_dataset(
            &transcript,
            &analysis,
            &silence,
            &activity,
            DatasetSettings::default(),
            &BTreeSet::new(),
            None,
        );
        assert_eq!(result.rejection(), Some("session_silent"));
    }

    #[test]
    fn duplicates_are_rejected() {
        let (transcript, pcm) = session("The meeting starts tomorrow morning.");
        let analysis = analyze(&transcript, &mut Clean, PrivacySettings::default(), &[]);
        let activity = SessionActivity::analyze(&pcm, RATE, VadSettings::default());
        let first = build_dataset(
            &transcript,
            &analysis,
            &pcm,
            &activity,
            DatasetSettings::default(),
            &BTreeSet::new(),
            None,
        );
        assert!(first.eligible());
        let seen = BTreeSet::from([first.content_hash.clone()]);
        let second = build_dataset(
            &transcript,
            &analysis,
            &pcm,
            &activity,
            DatasetSettings::default(),
            &seen,
            None,
        );
        assert_eq!(second.rejection(), Some("duplicate_example"));
    }

    #[test]
    fn a_failing_recheck_rejects() {
        struct Dirty;
        impl Recheck for Dirty {
            fn verify_clean(&mut self, _: &FrozenTranscript) -> Result<(), &'static str> {
                Err("sensitive_after_removal")
            }
        }
        let (transcript, pcm) = session("The meeting starts tomorrow morning.");
        let analysis = analyze(&transcript, &mut Clean, PrivacySettings::default(), &[]);
        let activity = SessionActivity::analyze(&pcm, RATE, VadSettings::default());
        let result = build_dataset(
            &transcript,
            &analysis,
            &pcm,
            &activity,
            DatasetSettings::default(),
            &BTreeSet::new(),
            Some(&mut Dirty),
        );
        assert_eq!(result.rejection(), Some("sensitive_after_removal"));
    }

    #[test]
    fn low_confidence_words_drop_their_sentence() {
        let (transcript, pcm) =
            session("The budget review went well. The meeting starts tomorrow morning.");
        let words: Vec<Word> = transcript
            .words()
            .iter()
            .map(|word| {
                let confidence = if word.id() == 1 { 0.1 } else { word.confidence() };
                Word::new(
                    word.id(),
                    word.text(),
                    word.interval().start(),
                    word.interval().end(),
                    confidence,
                )
                .unwrap()
                .with_display(word.display_text())
            })
            .collect();
        let transcript =
            FrozenTranscript::new("s", 1, words, RATE, transcript.total_samples(), "en").unwrap();
        let analysis = analyze(&transcript, &mut Clean, PrivacySettings::default(), &[]);
        let activity = SessionActivity::analyze(&pcm, RATE, VadSettings::default());
        let result = build_dataset(
            &transcript,
            &analysis,
            &pcm,
            &activity,
            DatasetSettings::default(),
            &BTreeSet::new(),
            None,
        );
        assert!(result.eligible(), "{:?}", result.decision);
        assert_eq!(result.spliced_text(), "The meeting starts tomorrow morning.");
    }

    #[test]
    fn built_package_passes_server_validation_and_carries_no_removed_text() {
        use crate::package::{PackageHeader, PackageLimits, build_package};
        let (_, result) =
            build("Send the parcel to Jane at 14 Oak Street. The meeting starts tomorrow.");
        let header = PackageHeader {
            eligibility_version: 1,
            sample_id: "sample-0123456789abcdef0123456789abcdef".to_owned(),
            language: "en".to_owned(),
            asr_model: "whisper-base-q5_1".to_owned(),
            asr_model_revision: "r".to_owned(),
            privacy_model: "qwen3-4b".to_owned(),
            privacy_model_revision: "r".to_owned(),
            policy_version: "p".to_owned(),
            consent_version: "c1".to_owned(),
            consent_reference: "grant".to_owned(),
        };
        let package = build_package(&result, &header, RATE, PackageLimits::default()).unwrap();
        let everything = format!(
            "{}{}",
            package.manifest,
            package
                .payloads
                .values()
                .map(|payload| String::from_utf8_lossy(payload).into_owned())
                .collect::<String>()
        );
        for removed in ["Jane", "Oak", "parcel"] {
            assert!(!everything.contains(removed), "{removed} leaked");
        }
        assert_eq!(package.payloads["clip-000.txt"], b"The meeting starts tomorrow.");
    }

    #[test]
    fn spliced_timeline_is_contiguous() {
        let (_, result) = build(
            "The budget review went well. Call me on four nine two seven one three eight. The meeting starts tomorrow morning.",
        );
        assert!(result.eligible(), "{:?}", result.decision);
        assert_eq!(result.clips.len(), 2);
        assert_eq!(result.clips[0].destination.start(), 0);
        assert_eq!(
            result.clips[1].destination.start(),
            result.clips[0].destination.end()
        );
        assert_eq!(
            result.spliced_pcm().len() as u64,
            result.clips.iter().map(|clip| clip.source.len()).sum::<u64>()
        );
    }
}
