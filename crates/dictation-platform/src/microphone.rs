//! Default-device microphone capture normalized to canonical PCM.

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use cpal::{
    FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use dictation_core::{
    CANONICAL_SAMPLE_RATE,
    audio::{AudioBufferError, PcmBuffer},
};

#[derive(Debug)]
pub enum CaptureError {
    NoInputDevice,
    DefaultConfig(String),
    UnsupportedFormat(SampleFormat),
    BuildStream(String),
    StartStream(String),
    Runtime(String),
    Buffer(AudioBufferError),
    Poisoned,
}

impl fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoInputDevice => write!(formatter, "no microphone input device is available"),
            Self::DefaultConfig(error) => {
                write!(formatter, "microphone configuration failed: {error}")
            }
            Self::UnsupportedFormat(format) => {
                write!(formatter, "unsupported microphone format {format}")
            }
            Self::BuildStream(error) => {
                write!(formatter, "microphone stream creation failed: {error}")
            }
            Self::StartStream(error) => {
                write!(formatter, "microphone stream start failed: {error}")
            }
            Self::Runtime(error) => write!(formatter, "microphone stream failed: {error}"),
            Self::Buffer(error) => write!(formatter, "microphone buffer failed: {error}"),
            Self::Poisoned => write!(formatter, "microphone state lock was poisoned"),
        }
    }
}

impl std::error::Error for CaptureError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedAudio {
    pub sample_rate: u32,
    pub samples: Vec<i16>,
}

enum Source {
    Device(Stream),
    /// Debug builds only: a 16 kHz mono WAV replayed at real-time pace so the
    /// whole pipeline can be tested without changing system audio routing.
    #[cfg(debug_assertions)]
    File(Arc<std::sync::atomic::AtomicBool>),
}

pub struct MicrophoneCapture {
    source: Source,
    buffer: Arc<Mutex<PcmBuffer>>,
    runtime_error: Arc<Mutex<Option<String>>>,
}

#[cfg(debug_assertions)]
fn start_test_file(path: &str, maximum_seconds: u64) -> Result<MicrophoneCapture, CaptureError> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let bytes = std::fs::read(path).map_err(|error| CaptureError::BuildStream(error.to_string()))?;
    let pcm: Vec<i16> = bytes
        .get(44..)
        .ok_or_else(|| CaptureError::BuildStream("test audio too short".to_owned()))?
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let buffer = Arc::new(Mutex::new(
        PcmBuffer::new(CANONICAL_SAMPLE_RATE, maximum_seconds).map_err(CaptureError::Buffer)?,
    ));
    let stop = Arc::new(AtomicBool::new(false));
    let feed = Arc::clone(&buffer);
    let halt = Arc::clone(&stop);
    std::thread::spawn(move || {
        // 20 ms frames; silence continues after the file ends until stopped.
        let frame = 320;
        let mut offset = 0;
        while !halt.load(Ordering::SeqCst) {
            let chunk: Vec<i16> = if offset < pcm.len() {
                let end = (offset + frame).min(pcm.len());
                let chunk = pcm[offset..end].to_vec();
                offset = end;
                chunk
            } else {
                vec![0; frame]
            };
            if let Ok(mut guard) = feed.lock() {
                if guard.push(&chunk).is_err() {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });
    Ok(MicrophoneCapture {
        source: Source::File(stop),
        buffer,
        runtime_error: Arc::new(Mutex::new(None)),
    })
}

impl MicrophoneCapture {
    /// Starts capture from the operating system's default input device.
    ///
    /// # Errors
    ///
    /// Fails when no input is available, its default format is unsupported, or
    /// the bounded stream cannot be built or started.
    pub fn start_default(maximum_seconds: u64) -> Result<Self, CaptureError> {
        #[cfg(debug_assertions)]
        if let Ok(path) = std::env::var("LOCAL_DICTATION_TEST_AUDIO") {
            return start_test_file(&path, maximum_seconds);
        }
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(CaptureError::NoInputDevice)?;
        let supported = device
            .default_input_config()
            .map_err(|error| CaptureError::DefaultConfig(error.to_string()))?;
        let format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let buffer = Arc::new(Mutex::new(
            PcmBuffer::new(CANONICAL_SAMPLE_RATE, maximum_seconds).map_err(CaptureError::Buffer)?,
        ));
        let runtime_error = Arc::new(Mutex::new(None));
        let stream = match format {
            SampleFormat::I16 => build_stream::<i16>(&device, &config, &buffer, &runtime_error),
            SampleFormat::F32 => build_stream::<f32>(&device, &config, &buffer, &runtime_error),
            SampleFormat::U16 => build_stream::<u16>(&device, &config, &buffer, &runtime_error),
            other => return Err(CaptureError::UnsupportedFormat(other)),
        }?;
        stream
            .play()
            .map_err(|error| CaptureError::StartStream(error.to_string()))?;
        Ok(Self {
            source: Source::Device(stream),
            buffer,
            runtime_error,
        })
    }

    /// Canonical samples captured from `start` onward, without stopping, for
    /// live partial transcription.
    ///
    /// # Errors
    ///
    /// Returns an asynchronous stream failure (e.g. the device was removed).
    pub fn snapshot_since(&self, start: usize) -> Result<Vec<i16>, CaptureError> {
        self.check_runtime()?;
        let buffer = self.buffer.lock().map_err(|_| CaptureError::Poisoned)?;
        Ok(buffer.samples().get(start..).map_or_else(Vec::new, <[i16]>::to_vec))
    }

    /// Samples captured so far.
    #[must_use]
    pub fn captured_samples(&self) -> usize {
        self.buffer.lock().map_or(0, |buffer| buffer.samples().len())
    }

    /// Reports device removal, permission revocation, or overflow promptly.
    ///
    /// # Errors
    ///
    /// Returns the first asynchronous stream failure.
    pub fn check_runtime(&self) -> Result<(), CaptureError> {
        if let Some(error) = self
            .runtime_error
            .lock()
            .map_err(|_| CaptureError::Poisoned)?
            .clone()
        {
            return Err(CaptureError::Runtime(error));
        }
        Ok(())
    }

    /// Stops capture and returns the exact canonical samples accumulated so far.
    ///
    /// # Errors
    ///
    /// Returns any asynchronous stream or bounded-buffer failure instead of a
    /// partial recording.
    pub fn stop(self) -> Result<CapturedAudio, CaptureError> {
        match self.source {
            Source::Device(stream) => drop(stream),
            #[cfg(debug_assertions)]
            Source::File(stop) => stop.store(true, std::sync::atomic::Ordering::SeqCst),
        }
        if let Some(error) = self
            .runtime_error
            .lock()
            .map_err(|_| CaptureError::Poisoned)?
            .take()
        {
            return Err(CaptureError::Runtime(error));
        }
        let buffer = self.buffer.lock().map_err(|_| CaptureError::Poisoned)?;
        if buffer.overflowed() {
            return Err(CaptureError::Runtime(
                "bounded capture buffer overflowed".to_owned(),
            ));
        }
        Ok(CapturedAudio {
            sample_rate: buffer.sample_rate(),
            samples: buffer.samples().to_vec(),
        })
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    buffer: &Arc<Mutex<PcmBuffer>>,
    runtime_error: &Arc<Mutex<Option<String>>>,
) -> Result<Stream, CaptureError>
where
    T: Sample + SizedSample,
    i16: FromSample<T>,
{
    let channels = usize::from(config.channels);
    let source_rate = config.sample_rate;
    let buffer = Arc::clone(buffer);
    let callback_error = Arc::clone(runtime_error);
    let stream_error = Arc::clone(runtime_error);
    let received_audio = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let callback_started = Arc::clone(&received_audio);
    let mut resampler = RateConverter::new(source_rate, CANONICAL_SAMPLE_RATE);
    device
        .build_input_stream(
            *config,
            move |input: &[T], _| {
                if !input.is_empty() {
                    callback_started.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                let mut mono = Vec::with_capacity(input.len() / channels.max(1));
                for frame in input.chunks_exact(channels.max(1)) {
                    let total: i64 = frame
                        .iter()
                        .copied()
                        .map(|sample| i64::from(i16::from_sample(sample)))
                        .sum();
                    let divisor = i64::try_from(frame.len()).unwrap_or(1).max(1);
                    mono.push(i16::try_from(total / divisor).unwrap_or_else(|_| {
                        if total.is_negative() {
                            i16::MIN
                        } else {
                            i16::MAX
                        }
                    }));
                }
                let canonical = resampler.convert(&mono);
                if let Ok(mut guard) = buffer.lock() {
                    if let Err(error) = guard.push(&canonical) {
                        record_error(&callback_error, error.to_string());
                    }
                } else {
                    record_error(&callback_error, "capture buffer lock poisoned".to_owned());
                }
            },
            move |error| {
                if fatal_stream_error(error.kind(), received_audio.load(std::sync::atomic::Ordering::Relaxed)) {
                    record_error(&stream_error, error.to_string());
                }
            },
            None,
        )
        .map_err(|error| CaptureError::BuildStream(error.to_string()))
}

fn fatal_stream_error(kind: cpal::ErrorKind, received_audio: bool) -> bool {
    // WASAPI devices can report a discontinuity before delivering their first
    // packet. No session audio has been lost in that case. Later gaps remain
    // fatal so incomplete audio cannot silently enter a training copy.
    !matches!(kind, cpal::ErrorKind::RealtimeDenied)
        && !(kind == cpal::ErrorKind::Xrun && !received_audio)
}

fn record_error(slot: &Arc<Mutex<Option<String>>>, message: String) {
    if let Ok(mut error) = slot.lock() {
        if error.is_none() {
            *error = Some(message);
        }
    }
}

/// Converts to the canonical rate with an anti-aliasing low-pass filter
/// ahead of fractional-phase decimation. Without the filter, energy above the
/// 8 kHz Nyquist limit folds into the speech band.
#[derive(Debug, Clone)]
struct RateConverter {
    source_rate: u32,
    target_rate: u32,
    accumulator: u64,
    taps: Vec<f32>,
    history: std::collections::VecDeque<f32>,
}

const FILTER_TAPS: usize = 63;

fn lowpass_taps(source_rate: u32, target_rate: u32) -> Vec<f32> {
    // Cutoff at 90% of the target Nyquist frequency, Blackman window.
    let cutoff = 0.45 * f64::from(target_rate) / f64::from(source_rate);
    #[allow(clippy::cast_precision_loss)]
    let middle = (FILTER_TAPS - 1) as f64 / 2.0;
    let mut taps: Vec<f64> = (0..FILTER_TAPS)
        .map(|index| {
            #[allow(clippy::cast_precision_loss)]
            let n = index as f64 - middle;
            let sinc = if n == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * std::f64::consts::PI * cutoff * n).sin() / (std::f64::consts::PI * n)
            };
            #[allow(clippy::cast_precision_loss)]
            let phase = 2.0 * std::f64::consts::PI * index as f64 / (FILTER_TAPS - 1) as f64;
            let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            sinc * window
        })
        .collect();
    let gain: f64 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= gain;
    }
    #[allow(clippy::cast_possible_truncation)]
    taps.into_iter().map(|tap| tap as f32).collect()
}

impl RateConverter {
    fn new(source_rate: u32, target_rate: u32) -> Self {
        let taps = if source_rate > target_rate {
            lowpass_taps(source_rate, target_rate)
        } else {
            vec![1.0]
        };
        Self {
            source_rate,
            target_rate,
            accumulator: 0,
            history: std::collections::VecDeque::from(vec![0.0; taps.len()]),
            taps,
        }
    }

    fn convert(&mut self, input: &[i16]) -> Vec<i16> {
        if self.source_rate == self.target_rate {
            return input.to_vec();
        }
        let capacity = input
            .len()
            .saturating_mul(self.target_rate as usize)
            .saturating_div(self.source_rate as usize)
            .saturating_add(1);
        let mut output = Vec::with_capacity(capacity);
        for sample in input {
            self.history.pop_front();
            self.history.push_back(f32::from(*sample));
            self.accumulator = self.accumulator.saturating_add(u64::from(self.target_rate));
            while self.accumulator >= u64::from(self.source_rate) {
                let filtered: f32 = self
                    .history
                    .iter()
                    .zip(&self.taps)
                    .map(|(value, tap)| value * tap)
                    .sum();
                #[allow(clippy::cast_possible_truncation)]
                output.push(filtered.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
                self.accumulator -= u64::from(self.source_rate);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_discontinuity_does_not_poison_capture() {
        assert!(!fatal_stream_error(cpal::ErrorKind::Xrun, false));
        assert!(fatal_stream_error(cpal::ErrorKind::Xrun, true));
        assert!(fatal_stream_error(cpal::ErrorKind::DeviceNotAvailable, false));
        assert!(fatal_stream_error(cpal::ErrorKind::PermissionDenied, true));
        assert!(!fatal_stream_error(cpal::ErrorKind::RealtimeDenied, true));
    }

    #[test]
    fn converter_has_exact_long_term_sample_count() {
        let input = vec![1_i16; 48_000];
        let mut converter = RateConverter::new(48_000, 16_000);
        assert_eq!(converter.convert(&input).len(), 16_000);
    }

    #[test]
    fn converter_attenuates_content_above_the_canonical_nyquist() {
        let tone = |frequency: f64| -> Vec<i16> {
            (0..48_000)
                .map(|n| {
                    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                    let value = (10_000.0
                        * (2.0 * std::f64::consts::PI * frequency * f64::from(n) / 48_000.0)
                            .sin()) as i16;
                    value
                })
                .collect()
        };
        let rms = |samples: &[i16]| {
            let energy: f64 = samples[1_000..].iter().map(|s| f64::from(*s).powi(2)).sum();
            #[allow(clippy::cast_precision_loss)]
            let mean = energy / (samples.len() - 1_000) as f64;
            mean.sqrt()
        };
        let passband = RateConverter::new(48_000, 16_000).convert(&tone(1_000.0));
        let aliased = RateConverter::new(48_000, 16_000).convert(&tone(12_000.0));
        assert!(rms(&passband) > 6_000.0);
        assert!(rms(&aliased) < 300.0, "12 kHz leaked: {}", rms(&aliased));
    }

    #[test]
    fn converter_preserves_fractional_phase_between_callbacks() {
        let mut converter = RateConverter::new(44_100, 16_000);
        let mut count = 0;
        for chunk in vec![vec![1_i16; 441]; 100] {
            count += converter.convert(&chunk).len();
        }
        assert_eq!(count, 16_000);
    }
}
