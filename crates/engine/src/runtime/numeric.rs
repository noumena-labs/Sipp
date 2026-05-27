use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) const MILLIS_PER_SECOND: u64 = 1_000;
pub(crate) const MILLIS_PER_SECOND_F64: f64 = MILLIS_PER_SECOND as f64;
pub(crate) const SECONDS_PER_MINUTE: u64 = 60;
pub(crate) const SECONDS_PER_HOUR: u64 = SECONDS_PER_MINUTE * 60;
pub(crate) const SECONDS_PER_DAY: u64 = SECONDS_PER_HOUR * 24;

#[inline]
pub(crate) fn saturating_usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

#[inline]
pub(crate) fn saturating_u32_to_i32(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

#[inline]
pub(crate) fn saturating_usize_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[inline]
pub(crate) fn saturating_usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[inline]
pub(crate) fn nonnegative_i32(value: i32) -> i32 {
    value.max(0)
}

#[inline]
pub(crate) fn positive_i32(value: i32) -> i32 {
    value.max(1)
}

#[inline]
pub(crate) fn positive_usize(value: usize) -> usize {
    value.max(1)
}

#[inline]
pub(crate) fn positive_i32_to_usize(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}

#[inline]
pub(crate) fn duration_ms(start: Instant, end: Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * MILLIS_PER_SECOND_F64
}

#[inline]
pub(crate) fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[inline]
pub(crate) fn positive_fair_share_i32(budget: i32, participant_count: i32) -> i32 {
    (budget / participant_count.max(1)).max(1)
}

#[inline]
pub(crate) fn unix_time_ms() -> u64 {
    system_time_unix_ms(SystemTime::now())
}

#[inline]
pub(crate) fn system_time_unix_ms(value: SystemTime) -> u64 {
    value
        .duration_since(UNIX_EPOCH)
        .map(duration_millis_u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    mod numeric_tests;
}
