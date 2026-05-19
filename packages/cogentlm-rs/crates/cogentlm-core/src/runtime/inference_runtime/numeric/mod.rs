//! Numeric conversions and small per-tick helpers used across the runtime.
//!
//! Kept private to the runtime: callers outside should not need these.

use std::time::Instant;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};

/// Tracks "first occurrence of this slot index in this tick" with a u64 bitmap.
/// `n_parallel` in practice sits in 1..=8 (max 32), so a u64 covers it; for
/// slot indices ≥64 we conservatively report "already seen" rather than
/// allocating a HashSet per tick.
#[inline(always)]
pub(super) fn unique_slot_first_use(seen: &mut u64, slot_index: usize) -> bool {
    if slot_index >= 64 {
        return false;
    }
    let bit = 1u64 << slot_index;
    let already = (*seen & bit) != 0;
    *seen |= bit;
    !already
}

#[inline]
pub(super) fn clamp_usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

#[inline]
pub(super) fn clamp_usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[inline]
pub(super) fn positive_i32_to_usize(value: i32) -> usize {
    usize::try_from(value.max(1)).unwrap_or(1)
}

#[inline]
pub(super) fn nonnegative_i32_to_usize(value: i32) -> usize {
    usize::try_from(value.max(0)).unwrap_or(0)
}

#[inline]
pub(super) fn ffi_arg_count_len(len: usize) -> Result<i32> {
    i32::try_from(len).map_err(|_| Error::InvalidRequest("too many llama.cpp common arguments"))
}

#[inline]
pub(super) fn usize_to_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

#[inline]
pub(super) fn nonnegative_i32_to_usize_opt(value: i32) -> Option<usize> {
    if value < 0 {
        None
    } else {
        usize::try_from(value).ok()
    }
}

#[inline]
pub(super) fn llama_context_limit_i32(shared_context: *mut ffi::llama_context) -> Option<i32> {
    if shared_context.is_null() {
        return None;
    }
    i32::try_from(unsafe { ffi::llama_n_ctx(shared_context) }).ok()
}

#[inline]
pub(super) fn saturating_i32_delta(after: i32, before: i32) -> i32 {
    after.saturating_sub(before)
}

#[inline]
pub(super) fn saturating_usize_delta_to_i32(after: usize, before: usize) -> i32 {
    clamp_usize_to_i32(after.saturating_sub(before))
}

#[inline]
pub(super) fn duration_ms(start: Instant, end: Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1000.0
}

/// Stable fingerprint of a path. Used as a sticky id for residency / engine identity.
pub(super) fn fingerprint_path(path: &std::path::Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    hasher.finish()
}

/// llama.cpp signals "need this many bytes" with a negative `result`. Return
/// the absolute size when it exceeds the current scratch capacity, else `None`.
pub(super) fn token_piece_growth_capacity(result: i32, capacity: i32) -> Option<i32> {
    let needed = result.checked_abs()?;
    (needed > capacity).then_some(needed)
}

#[cfg(test)]
mod tests;
