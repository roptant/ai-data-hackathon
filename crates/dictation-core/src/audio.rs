//! Bounded canonical PCM capture buffer.
//!
//! Overflow is explicit and poisons the session; samples are never silently
//! dropped. Platform microphone adapters feed normalized 16 kHz mono `i16`
//! frames into this type.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioBufferError {
    InvalidConfiguration,
    Overflow { attempted: usize, remaining: usize },
    AlreadyOverflowed,
}

impl fmt::Display for AudioBufferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => {
                write!(formatter, "audio buffer configuration is invalid")
            }
            Self::Overflow {
                attempted,
                remaining,
            } => {
                write!(
                    formatter,
                    "audio frame has {attempted} samples but only {remaining} remain"
                )
            }
            Self::AlreadyOverflowed => write!(formatter, "audio buffer previously overflowed"),
        }
    }
}

impl std::error::Error for AudioBufferError {}

#[derive(Debug, Clone)]
pub struct PcmBuffer {
    sample_rate: u32,
    maximum_samples: usize,
    samples: Vec<i16>,
    overflowed: bool,
}

impl PcmBuffer {
    /// Creates a bounded mono PCM buffer.
    ///
    /// # Errors
    ///
    /// Rejects zero values and capacities that do not fit the platform.
    pub fn new(sample_rate: u32, maximum_seconds: u64) -> Result<Self, AudioBufferError> {
        if sample_rate == 0 || maximum_seconds == 0 {
            return Err(AudioBufferError::InvalidConfiguration);
        }
        let maximum_samples = u64::from(sample_rate)
            .checked_mul(maximum_seconds)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(AudioBufferError::InvalidConfiguration)?;
        Ok(Self {
            sample_rate,
            maximum_samples,
            samples: Vec::new(),
            overflowed: false,
        })
    }

    /// Appends one complete callback frame or none of it.
    ///
    /// # Errors
    ///
    /// Returns an explicit overflow error without partially appending the frame.
    pub fn push(&mut self, frame: &[i16]) -> Result<(), AudioBufferError> {
        if self.overflowed {
            return Err(AudioBufferError::AlreadyOverflowed);
        }
        let remaining = self.maximum_samples.saturating_sub(self.samples.len());
        if frame.len() > remaining {
            self.overflowed = true;
            return Err(AudioBufferError::Overflow {
                attempted: frame.len(),
                remaining,
            });
        }
        self.samples.extend_from_slice(frame);
        Ok(())
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[must_use]
    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    #[must_use]
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        let samples = u64::try_from(self.samples.len()).unwrap_or(u64::MAX);
        samples.saturating_mul(1_000) / u64::from(self.sample_rate)
    }

    #[must_use]
    pub fn pcm_s16le(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len().saturating_mul(2));
        for sample in &self.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_never_partially_appends_a_frame() {
        let mut buffer = PcmBuffer::new(4, 1).unwrap();
        buffer.push(&[1, 2, 3]).unwrap();
        assert!(matches!(
            buffer.push(&[4, 5]),
            Err(AudioBufferError::Overflow {
                attempted: 2,
                remaining: 1
            })
        ));
        assert_eq!(buffer.samples(), &[1, 2, 3]);
        assert!(matches!(
            buffer.push(&[4]),
            Err(AudioBufferError::AlreadyOverflowed)
        ));
    }

    #[test]
    fn canonical_bytes_are_little_endian() {
        let mut buffer = PcmBuffer::new(16_000, 1).unwrap();
        buffer.push(&[0x1234, -2]).unwrap();
        assert_eq!(buffer.pcm_s16le(), vec![0x34, 0x12, 0xfe, 0xff]);
    }
}
