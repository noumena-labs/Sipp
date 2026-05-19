//! KV-cache space management and prompt preparation for prefill.
//!
//! Three primitives:
//! - `ensure_context_space` slides the KV window when the sequence would
//!   exceed `n_ctx`, preserving a retained prefix.
//! - `prepare_sequence_for_prompt` runs LCP reuse + optional snapshot
//!   restore and returns the number of cache-hit tokens.
//! - `ensure_decode_step_context_space` is the per-decode-step variant.

use std::cmp;

use cogentlm_sys as ffi;

use crate::runtime::scheduler::SlotState;
use crate::runtime::session::{PrefixCachePolicy, PrefixStateCache, SequenceState, SessionStore};
use crate::runtime::{llama_seq_id, llama_token};

use super::numeric::{llama_context_limit_i32, nonnegative_i32_to_usize_opt, usize_to_i32};

#[inline]
pub(super) fn resolve_initial_decode_context_reservation(
    max_output_tokens: i32,
    decode_reserve: i32,
) -> i32 {
    if max_output_tokens <= 0 {
        0
    } else {
        max_output_tokens.min(decode_reserve.max(1))
    }
}

/// Slides the KV window so `state.n_past + new_tokens_needed <= n_ctx`,
/// preserving `retained_prefix_tokens` at the head. Returns `false` if the
/// shared context is invalid or no amount of trimming can fit the new tokens.
pub(super) fn ensure_context_space(
    shared_context: *mut ffi::llama_context,
    retained_prefix_tokens: i32,
    state: &mut SequenceState,
    seq_id: llama_seq_id,
    new_tokens_needed: i32,
) -> bool {
    if shared_context.is_null() || seq_id < 0 {
        return false;
    }
    if new_tokens_needed <= 0 {
        return true;
    }

    let Some(n_ctx) = llama_context_limit_i32(shared_context) else {
        return false;
    };
    if n_ctx <= 0 || new_tokens_needed > n_ctx {
        return false;
    }
    let Some(total_needed) = state.n_past.checked_add(new_tokens_needed) else {
        return false;
    };
    if total_needed <= n_ctx {
        return true;
    }

    let mem = unsafe { ffi::llama_get_memory(shared_context) };
    let n_keep = retained_prefix_tokens.min(state.n_past).max(0);
    let required_discard = total_needed - n_ctx;
    let max_discard = (state.n_past - n_keep).max(0);
    let n_discard = required_discard.clamp(0, max_discard);

    if n_discard <= 0 {
        if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, 0, -1) } {
            return false;
        }
        state.current_kv_tokens.clear();
        state.n_past = 0;
        return true;
    }

    let Some(discard_end) = n_keep.checked_add(n_discard) else {
        return false;
    };

    if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, n_keep, discard_end) } {
        return false;
    }
    unsafe {
        ffi::llama_memory_seq_add(mem, seq_id, discard_end, -1, -n_discard);
    }

    let Some(n_keep_len) = nonnegative_i32_to_usize_opt(n_keep) else {
        return false;
    };
    let Some(discard_end_len) = nonnegative_i32_to_usize_opt(discard_end) else {
        return false;
    };
    if state.current_kv_tokens.len() > n_keep_len {
        let erase_end = cmp::min(discard_end_len, state.current_kv_tokens.len());
        state.current_kv_tokens.drain(n_keep_len..erase_end);
    } else {
        state.current_kv_tokens.clear();
    }
    let Some(n_past) = usize_to_i32(state.current_kv_tokens.len()) else {
        return false;
    };
    state.n_past = n_past;

    let Some(total_needed) = state.n_past.checked_add(new_tokens_needed) else {
        return false;
    };
    if total_needed <= n_ctx {
        return true;
    }

    if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, 0, -1) } {
        return false;
    }
    state.current_kv_tokens.clear();
    state.n_past = 0;
    true
}

/// Drives prefix reuse for an admitted request:
///   1. live LCP against the session's existing KV tokens,
///   2. optional restore from the snapshot prefix cache,
///   3. trim the KV to the final match length (honoring recurrent/hybrid
///      model constraints),
///   4. ensure room for the missing prompt tokens + decode reservation.
///
/// Writes the prefill cursor (tokens already in KV) into `out_prefill_cursor`
/// and returns the number of cache hits as i32.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_sequence_for_prompt(
    shared_context: *mut ffi::llama_context,
    primary_model: *mut ffi::llama_model,
    retained_prefix_tokens: i32,
    snapshot_prefix_cache: bool,
    decode_token_reserve: i32,
    model_fingerprint: u64,
    session_store: &SessionStore,
    prefix_state_cache: &mut PrefixStateCache,
    prefix_cache_policy: &mut PrefixCachePolicy,
    context_key: &str,
    prompt_tokens: &[llama_token],
    n_tokens_predict: i32,
    state: &mut SequenceState,
    seq_id: llama_seq_id,
    out_prefill_cursor: &mut usize,
) -> Option<i32> {
    *out_prefill_cursor = 0;
    if shared_context.is_null() || primary_model.is_null() || seq_id < 0 || prompt_tokens.is_empty()
    {
        return None;
    }

    let mem = unsafe { ffi::llama_get_memory(shared_context) };
    let has_live_tokens = !state.current_kv_tokens.is_empty();
    let live_match_len = if has_live_tokens {
        session_store.compute_lcp_reuse(state, prompt_tokens)
    } else {
        0
    };
    let mut match_len = live_match_len;
    let mut restored_from_prefix_cache = false;

    // Handle-based lookup avoids cloning the entry's state_bytes (potentially huge).
    if snapshot_prefix_cache {
        if let Some(handle) = prefix_state_cache.find_best_prefix_handle(
            model_fingerprint,
            context_key,
            prompt_tokens,
            prefix_cache_policy,
        ) {
            if handle.token_count > live_match_len
                && prefix_state_cache.restore_by_handle(shared_context, seq_id, handle)
            {
                if let Some(entry) = prefix_state_cache.entry_by_handle(handle) {
                    state.current_kv_tokens.clear();
                    state
                        .current_kv_tokens
                        .extend_from_slice(&entry.prefix_tokens);
                    state.n_past = usize_to_i32(entry.token_count)?;
                    match_len = entry.token_count.min(prompt_tokens.len());
                    restored_from_prefix_cache = true;
                }
            }
        }
    }

    if !restored_from_prefix_cache && !has_live_tokens {
        unsafe {
            ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
        }
        state.current_kv_tokens.clear();
        state.n_past = 0;
        match_len = 0;
    }

    // Re-run LCP — it can grow after a cache restore (but never after `ensure_context_space`).
    match_len = match_len.max(session_store.compute_lcp_reuse(state, prompt_tokens));
    let missing_prompt_tokens = prompt_tokens.len().checked_sub(match_len)?;
    let tokens_to_add = usize_to_i32(missing_prompt_tokens)?;
    let total_needed = tokens_to_add
        + resolve_initial_decode_context_reservation(n_tokens_predict, decode_token_reserve);

    if !ensure_context_space(
        shared_context,
        retained_prefix_tokens,
        state,
        seq_id,
        total_needed,
    ) {
        return None;
    }

    match_len = match_len.min(session_store.compute_lcp_reuse(state, prompt_tokens));
    let allow_partial_kv = !(unsafe { ffi::llama_model_is_recurrent(primary_model) }
        || unsafe { ffi::llama_model_is_hybrid(primary_model) });

    if match_len < state.current_kv_tokens.len() || state.current_kv_tokens.is_empty() {
        if !allow_partial_kv || state.current_kv_tokens.is_empty() {
            unsafe {
                ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
            }
            state.current_kv_tokens.clear();
            state.n_past = 0;
            match_len = 0;
        } else {
            let match_len_i32 = usize_to_i32(match_len)?;
            if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, match_len_i32, -1) } {
                return None;
            }
            state.current_kv_tokens.truncate(match_len);
            state.n_past = match_len_i32;
        }
    }

    // Full-prompt cache hit needs a token to drive decode — trim 1 from KV or invalidate.
    if match_len == prompt_tokens.len() && match_len > 0 {
        if allow_partial_kv {
            let match_len_i32 = usize_to_i32(match_len)?;
            let last_token_position = match_len_i32.checked_sub(1)?;
            if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, last_token_position, -1) } {
                return None;
            }
            state.current_kv_tokens.truncate(match_len - 1);
            state.n_past = last_token_position;
            match_len -= 1;
        } else {
            unsafe {
                ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
            }
            state.current_kv_tokens.clear();
            state.n_past = 0;
            match_len = 0;
        }
    }

    let cache_hits = usize_to_i32(match_len)?;
    *out_prefill_cursor = match_len;
    Some(cache_hits)
}

/// Per-step variant: makes room for one more decode token. For multimodal
/// turns, the additional token must fit strictly within `n_ctx` (no eviction
/// of the multimodal prefix is allowed).
pub(super) fn ensure_decode_step_context_space(
    shared_context: *mut ffi::llama_context,
    retained_prefix_tokens: i32,
    slot: &mut SlotState,
) -> bool {
    if shared_context.is_null() || slot.session.is_none() {
        return false;
    }
    if slot.generated_tokens.is_empty() {
        return true;
    }
    if slot
        .request()
        .is_some_and(|request| request.is_multimodal_turn)
        && llama_context_limit_i32(shared_context).is_none_or(|n_ctx| {
            slot.mirror
                .n_past
                .checked_add(1)
                .is_none_or(|needed| needed > n_ctx)
        })
    {
        return false;
    }
    ensure_context_space(
        shared_context,
        retained_prefix_tokens,
        &mut slot.mirror,
        slot.seq_id,
        1,
    )
}
