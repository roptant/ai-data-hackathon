//! Non-neural voice activity detection (plan §3, §6).
//!
//! The plan ships exactly two learned model roles, so activity detection is
//! deterministic signal processing: framed RMS energy against an adaptive
//! noise floor with hysteresis. It answers whether anyone spoke at all (so an
//! empty session cannot become hallucinated text) and how much of an interval
//! is speech rather than room noise.
//!
//! Unlike the Python reference, the noise floor is estimated once over the
//! whole session. Estimating it per retained clip placed the floor inside
//! continuous speech and rejected good clips as silence.

use crate::time_map::{SampleInterval, normalize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadSettings {
    pub frame_ms: u32,
    /// Speech must exceed the noise floor by this many decibels.
    pub threshold_db: f32,
    /// Absolute floor so digital silence never counts as speech.
    pub absolute_floor_db: f32,
    /// Loud frames required to open, quiet frames required to close.
    pub open_frames: u32,
    pub close_frames: u32,
    /// Sessions whose speech ratio is below this produce no text or example.
    pub min_session_speech_ratio: f32,
}

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            frame_ms: 20,
            threshold_db: 9.0,
            absolute_floor_db: -55.0,
            open_frames: 2,
            close_frames: 8,
            min_session_speech_ratio: 0.08,
        }
    }
}

/// Per-frame speech decisions for one session's canonical PCM.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionActivity {
    frame_samples: u64,
    total_samples: u64,
    flags: Vec<bool>,
    settings: VadSettings,
}

fn frame_level_db(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return -120.0;
    }
    let energy: f64 = frame
        .iter()
        .map(|sample| {
            let value = f64::from(*sample) / 32_768.0;
            value * value
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let mean = energy / frame.len() as f64;
    if mean <= 1e-12 {
        -120.0
    } else {
        #[allow(clippy::cast_possible_truncation)]
        let level = (10.0 * mean.log10()) as f32;
        level
    }
}

impl SessionActivity {
    #[must_use]
    pub fn analyze(samples: &[i16], sample_rate: u32, settings: VadSettings) -> Self {
        let frame_samples = (u64::from(sample_rate) * u64::from(settings.frame_ms) / 1_000).max(1);
        let frame = usize::try_from(frame_samples).unwrap_or(usize::MAX);
        let levels: Vec<f32> = samples.chunks(frame).map(frame_level_db).collect();
        let floor = if levels.is_empty() {
            -120.0
        } else {
            let mut ordered = levels.clone();
            ordered.sort_by(f32::total_cmp);
            ordered[(ordered.len() / 5).saturating_sub(1)]
        };
        let threshold = (floor + settings.threshold_db).max(settings.absolute_floor_db);
        let mut flags = Vec::with_capacity(levels.len());
        let mut speaking = false;
        let mut run = 0;
        for level in levels {
            let loud = level >= threshold;
            if speaking {
                run = if loud { 0 } else { run + 1 };
                if run >= settings.close_frames {
                    speaking = false;
                    run = 0;
                }
            } else {
                run = if loud { run + 1 } else { 0 };
                if run >= settings.open_frames {
                    speaking = true;
                    run = 0;
                }
            }
            flags.push(speaking);
        }
        Self {
            frame_samples,
            total_samples: samples.len() as u64,
            flags,
            settings,
        }
    }

    /// Speech samples inside `interval`, counted at frame resolution.
    #[must_use]
    pub fn speech_samples(&self, interval: SampleInterval) -> u64 {
        let mut total = 0;
        for (index, flag) in self.flags.iter().enumerate() {
            if !flag {
                continue;
            }
            let start = index as u64 * self.frame_samples;
            let end = (start + self.frame_samples).min(self.total_samples);
            let overlap_start = start.max(interval.start());
            let overlap_end = end.min(interval.end());
            total += overlap_end.saturating_sub(overlap_start);
        }
        total
    }

    #[must_use]
    pub fn speech_ratio(&self, interval: SampleInterval) -> f32 {
        if interval.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratio = self.speech_samples(interval) as f64 / interval.len() as f64;
        #[allow(clippy::cast_possible_truncation)]
        let ratio = ratio as f32;
        ratio
    }

    #[must_use]
    pub fn session_speech_ratio(&self) -> f32 {
        SampleInterval::new(0, self.total_samples)
            .map_or(0.0, |interval| self.speech_ratio(interval))
    }

    /// True when the session should produce neither text nor an example.
    #[must_use]
    pub fn is_effectively_silent(&self) -> bool {
        self.session_speech_ratio() < self.settings.min_session_speech_ratio
    }

    #[must_use]
    pub fn speech_intervals(&self) -> Vec<SampleInterval> {
        let mut intervals = Vec::new();
        let mut start = None;
        for (index, flag) in self.flags.iter().enumerate() {
            let position = (index as u64 * self.frame_samples).min(self.total_samples);
            match (flag, start) {
                (true, None) => start = Some(position),
                (false, Some(begin)) => {
                    if let Ok(interval) = SampleInterval::new(begin, position) {
                        intervals.push(interval);
                    }
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = start {
            if let Ok(interval) = SampleInterval::new(begin, self.total_samples) {
                intervals.push(interval);
            }
        }
        normalize(intervals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(samples: usize, amplitude: f64) -> Vec<i16> {
        (0..samples)
            .map(|index| {
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let value = (amplitude * (index as f64 * 0.15).sin()) as i16;
                value
            })
            .collect()
    }

    #[test]
    fn digital_silence_is_silent() {
        let activity = SessionActivity::analyze(&vec![0; 32_000], 16_000, VadSettings::default());
        assert!(activity.is_effectively_silent());
        assert!(activity.speech_intervals().is_empty());
    }

    #[test]
    fn continuous_speech_in_a_clip_is_not_mistaken_for_silence() {
        let mut samples = tone(8_000, 30.0); // quiet room
        samples.extend(tone(32_000, 8_000.0)); // two seconds of loud speech
        samples.extend(tone(8_000, 30.0));
        let activity = SessionActivity::analyze(&samples, 16_000, VadSettings::default());
        let speech = SampleInterval::new(8_000, 40_000).unwrap();
        assert!(activity.speech_ratio(speech) > 0.9);
        assert!(!activity.is_effectively_silent());
        let intervals = activity.speech_intervals();
        assert_eq!(intervals.len(), 1);
        assert!(intervals[0].start() >= 8_000 && intervals[0].start() < 9_000);
    }

    #[test]
    fn empty_interval_has_zero_ratio() {
        let activity = SessionActivity::analyze(&tone(1_600, 9_000.0), 16_000, VadSettings::default());
        assert!(activity.speech_ratio(SampleInterval::new(5, 5).unwrap()).abs() < f32::EPSILON);
    }
}
