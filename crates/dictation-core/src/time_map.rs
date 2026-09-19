//! Half-open sample interval arithmetic used by privacy filtering.
//!
//! Removal intervals always round outward while retained intervals round
//! inward. This ensures rounding cannot leave a fragment of removed audio in a
//! retained clip.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleInterval {
    start: u64,
    end: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvertedInterval {
    pub start: u64,
    pub end: u64,
}

impl fmt::Display for InvertedInterval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "inverted interval: [{}, {})",
            self.start, self.end
        )
    }
}

impl std::error::Error for InvertedInterval {}

impl SampleInterval {
    #[must_use]
    pub const fn from_zero(end: u64) -> Self {
        Self { start: 0, end }
    }

    /// Construct a half-open interval `[start, end)`.
    ///
    /// # Errors
    ///
    /// Returns [`InvertedInterval`] when `end` precedes `start`.
    pub fn new(start: u64, end: u64) -> Result<Self, InvertedInterval> {
        if end < start {
            return Err(InvertedInterval { start, end });
        }
        Ok(Self { start, end })
    }

    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    #[must_use]
    pub const fn end(self) -> u64 {
        self.end
    }

    #[must_use]
    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    #[must_use]
    pub const fn contains(self, sample: u64) -> bool {
        self.start <= sample && sample < self.end
    }

    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    #[must_use]
    pub const fn touches(self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    #[must_use]
    pub fn clamp(self, bound: Self) -> Self {
        let start = self.start.clamp(bound.start, bound.end);
        let end = self.end.clamp(bound.start, bound.end).max(start);
        Self { start, end }
    }
}

#[must_use]
pub fn milliseconds_to_samples(milliseconds: u64, sample_rate: u32) -> u64 {
    let numerator = u128::from(milliseconds)
        .saturating_mul(u128::from(sample_rate))
        .saturating_add(500);
    u64::try_from(numerator / 1_000).unwrap_or(u64::MAX)
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn samples_to_milliseconds(samples: u64, sample_rate: u32) -> f64 {
    samples as f64 * 1_000.0 / f64::from(sample_rate)
}

#[must_use]
pub fn normalize(intervals: impl IntoIterator<Item = SampleInterval>) -> Vec<SampleInterval> {
    let mut ordered: Vec<_> = intervals
        .into_iter()
        .filter(|interval| !interval.is_empty())
        .collect();
    ordered.sort_unstable();

    let mut merged: Vec<SampleInterval> = Vec::with_capacity(ordered.len());
    for interval in ordered {
        if let Some(previous) = merged.last_mut() {
            if interval.start <= previous.end {
                previous.end = previous.end.max(interval.end);
                continue;
            }
        }
        merged.push(interval);
    }
    merged
}

#[must_use]
pub fn pad(
    interval: SampleInterval,
    padding_samples: u64,
    bound: SampleInterval,
) -> SampleInterval {
    let start = interval
        .start
        .saturating_sub(padding_samples)
        .max(bound.start);
    let end = interval
        .end
        .saturating_add(padding_samples)
        .min(bound.end)
        .max(start);
    SampleInterval { start, end }
}

#[must_use]
pub fn pad_all(
    intervals: impl IntoIterator<Item = SampleInterval>,
    padding_samples: u64,
    bound: SampleInterval,
) -> Vec<SampleInterval> {
    normalize(
        intervals
            .into_iter()
            .map(|interval| pad(interval, padding_samples, bound)),
    )
}

#[must_use]
pub fn complement(removed: &[SampleInterval], bound: SampleInterval) -> Vec<SampleInterval> {
    let mut retained = Vec::new();
    let mut cursor = bound.start;
    for interval in normalize(removed.iter().copied()) {
        let clipped = interval.clamp(bound);
        if clipped.is_empty() {
            continue;
        }
        if clipped.start > cursor {
            retained.push(SampleInterval {
                start: cursor,
                end: clipped.start,
            });
        }
        cursor = cursor.max(clipped.end);
    }
    if cursor < bound.end {
        retained.push(SampleInterval {
            start: cursor,
            end: bound.end,
        });
    }
    retained
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationMap {
    retained: Vec<SampleInterval>,
    offsets: Vec<u64>,
}

impl DestinationMap {
    #[must_use]
    pub fn build(retained: impl IntoIterator<Item = SampleInterval>) -> Self {
        let retained = normalize(retained);
        let mut running = 0_u64;
        let offsets = retained
            .iter()
            .map(|interval| {
                let current = running;
                running = running.saturating_add(interval.len());
                current
            })
            .collect();
        Self { retained, offsets }
    }

    #[must_use]
    pub fn retained(&self) -> &[SampleInterval] {
        &self.retained
    }

    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.retained.iter().map(|interval| interval.len()).sum()
    }

    #[must_use]
    pub fn destination_of(&self, index: usize) -> Option<SampleInterval> {
        let source = *self.retained.get(index)?;
        let offset = *self.offsets.get(index)?;
        Some(SampleInterval {
            start: offset,
            end: offset + source.len(),
        })
    }

    #[must_use]
    pub fn map_sample(&self, sample: u64) -> Option<u64> {
        self.retained
            .iter()
            .zip(&self.offsets)
            .find(|(interval, _)| interval.contains(sample))
            .map(|(interval, offset)| offset + sample - interval.start)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResampleMap {
    source_rate: u32,
    canonical_rate: u32,
}

impl ResampleMap {
    /// Construct a mapping between canonical and stored sample rates.
    ///
    /// # Errors
    ///
    /// Returns an error if either sample rate is zero.
    pub fn new(source_rate: u32, canonical_rate: u32) -> Result<Self, &'static str> {
        if source_rate == 0 || canonical_rate == 0 {
            return Err("sample rates must be positive");
        }
        Ok(Self {
            source_rate,
            canonical_rate,
        })
    }

    fn floor_to_source(self, sample: u64) -> u64 {
        let numerator = u128::from(sample).saturating_mul(u128::from(self.source_rate));
        u64::try_from(numerator / u128::from(self.canonical_rate)).unwrap_or(u64::MAX)
    }

    fn ceil_to_source(self, sample: u64) -> u64 {
        let denominator = u128::from(self.canonical_rate);
        let numerator = u128::from(sample)
            .saturating_mul(u128::from(self.source_rate))
            .saturating_add(denominator - 1);
        u64::try_from(numerator / denominator).unwrap_or(u64::MAX)
    }

    #[must_use]
    pub fn removal_to_source(self, interval: SampleInterval) -> SampleInterval {
        SampleInterval {
            start: self.floor_to_source(interval.start),
            end: self.ceil_to_source(interval.end),
        }
    }

    #[must_use]
    pub fn retained_to_source(self, interval: SampleInterval) -> SampleInterval {
        let start = self.ceil_to_source(interval.start);
        let end = self.floor_to_source(interval.end);
        SampleInterval {
            start,
            end: end.max(start),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(start: u64, end: u64) -> SampleInterval {
        SampleInterval::new(start, end).expect("valid fixture")
    }

    #[test]
    fn half_open_intervals_do_not_share_boundary_samples() {
        let first = interval(0, 10);
        let second = interval(10, 20);
        assert!(!first.intersects(second));
        assert!(first.touches(second));
        assert!(!first.contains(10));
        assert!(second.contains(10));
    }

    #[test]
    fn normalize_merges_overlapping_and_abutting_intervals() {
        assert_eq!(
            normalize([
                interval(60, 70),
                interval(10, 20),
                interval(15, 30),
                interval(30, 40)
            ]),
            vec![interval(10, 40), interval(60, 70)]
        );
    }

    #[test]
    fn complement_returns_only_the_retained_gaps() {
        assert_eq!(
            complement(&[interval(10, 20), interval(50, 60)], interval(0, 100)),
            vec![interval(0, 10), interval(20, 50), interval(60, 100)]
        );
    }

    #[test]
    fn destination_map_uses_contiguous_output_offsets() {
        let map = DestinationMap::build([interval(0, 100), interval(300, 450)]);
        assert_eq!(map.destination_of(0), Some(interval(0, 100)));
        assert_eq!(map.destination_of(1), Some(interval(100, 250)));
        assert_eq!(map.map_sample(300), Some(100));
        assert_eq!(map.map_sample(200), None);
    }

    #[test]
    fn removal_rounds_outward_and_retention_rounds_inward() {
        let map = ResampleMap::new(44_100, 16_000).expect("valid rates");
        let removed = map.removal_to_source(interval(1, 2));
        let retained = map.retained_to_source(interval(1, 2));
        assert!(removed.start() <= retained.start());
        assert!(removed.end() >= retained.end());
    }
}
