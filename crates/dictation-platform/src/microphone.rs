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

pub struct MicrophoneCapture {
    stream: Stream,
    buffer: Arc<Mutex<PcmBuffer>>,
    runtime_error: Arc<Mutex<Option<String>>>,
}

impl MicrophoneCapture {
    /// Starts capture from the operating system's default input device.
    ///
    /// # Errors
    ///
    /// Fails when no input is available, its default format is unsupported, or
    /// the bounded stream cannot be built or started.
    pub fn start_default(maximum_seconds: u64) -> Result<Self, CaptureError> {
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
            stream,
            buffer,
            runtime_error,
        })
    }

    /// Stops capture and returns the exact canonical samples accumulated so far.
    ///
    /// # Errors
    ///
    /// Returns any asynchronous stream or bounded-buffer failure instead of a
    /// partial recording.
    pub fn stop(self) -> Result<CapturedAudio, CaptureError> {
        drop(self.stream);
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
    let mut resampler = RateConverter::new(source_rate, CANONICAL_SAMPLE_RATE);
    device
        .build_input_stream(
            *config,
            move |input: &[T], _| {
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
            move |error| record_error(&stream_error, error.to_string()),
            None,
        )
        .map_err(|error| CaptureError::BuildStream(error.to_string()))
}

fn record_error(slot: &Arc<Mutex<Option<String>>>, message: String) {
    if let Ok(mut error) = slot.lock() {
        if error.is_none() {
            *error = Some(message);
        }
    }
}

#[derive(Debug, Clone)]
struct RateConverter {
    source_rate: u32,
    target_rate: u32,
    accumulator: u64,
}

impl RateConverter {
    const fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            source_rate,
            target_rate,
            accumulator: 0,
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
            self.accumulator = self.accumulator.saturating_add(u64::from(self.target_rate));
            while self.accumulator >= u64::from(self.source_rate) {
                output.push(*sample);
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
    fn converter_has_exact_long_term_sample_count() {
        let input = vec![1_i16; 48_000];
        let mut converter = RateConverter::new(48_000, 16_000);
        assert_eq!(converter.convert(&input).len(), 16_000);
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
