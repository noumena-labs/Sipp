//! Slot phase normalization and decode-seed recovery.
//!
//! `normalize_runnable_slot_state` is called at the top of every tick to
//! transition slots between admitted/prefill/decode/streaming/completed.
//! `recover_decode_seed_state` re-anchors a slot that was promoted to Decode
//! but lost its sampled-token seed (e.g. after a snapshot restore).

use std::cmp;

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::llama_seq_id;
use crate::runtime::request::GenerateRequestLifecycle;
use crate::runtime::scheduler::{SlotPhase, SlotState};
use crate::runtime::session::SequenceState;
use crate::runtime::REQUEST_CANCELLED_MESSAGE;

use super::super::numeric::{nonnegative_i32_to_usize, usize_to_i32};

/// Transitions `slot.phase` based on its prompt, generated tokens, and any
/// pending cancel / limit conditions. Returns `false` only when recovery
/// failed and the slot was marked `Failed`.
pub(super) fn normalize_runnable_slot_state(
    slot: &mut SlotState,
    native_runtime: &mut NativeRuntimeHandle,
    retained_prefix_tokens: i32,
) -> bool {
    let (is_multimodal_turn, prompt_tokens_len, cancel_requested, max_output_tokens) =
        if let Some(r) = slot.request() {
            (
                r.is_multimodal_turn,
                r.prompt_tokens.len(),
                r.cancel_requested,
                r.max_output_tokens,
            )
        } else {
            return true;
        };

    if slot.phase == SlotPhase::Admitted {
        slot.phase = SlotPhase::Prefill;
    }

    if slot.phase == SlotPhase::Prefill
        && !is_multimodal_turn
        && slot.prefill_cursor >= prompt_tokens_len
        && slot.mirror.n_past > 0
    {
        slot.phase = SlotPhase::Decode;
    }

    if slot.phase == SlotPhase::Streaming && slot.buffered_output_text.is_empty() {
        let reached_limit = max_output_tokens > 0
            && slot.generated_tokens.len() >= nonnegative_i32_to_usize(max_output_tokens);

        if cancel_requested {
            slot.cancel(REQUEST_CANCELLED_MESSAGE);
            return true;
        }

        if reached_limit {
            slot.phase = SlotPhase::Completed;
            if let Some(request_mut) = slot.request_mut() {
                request_mut.lifecycle = GenerateRequestLifecycle::Completed;
            }
            return true;
        }

        slot.phase = if slot.generated_tokens.is_empty() {
            SlotPhase::Prefill
        } else {
            SlotPhase::Decode
        };
        if let Some(request_mut) = slot.request_mut() {
            request_mut.lifecycle = GenerateRequestLifecycle::Running;
        }
    }

    if slot.phase == SlotPhase::Decode && slot.generated_tokens.is_empty() {
        return recover_decode_seed_state(slot, native_runtime, retained_prefix_tokens);
    }

    true
}

/// Re-anchors a Decode slot that has no sampled token yet. Either falls back
/// to Prefill when KV is missing/shorter than the prompt, or trims the KV by
/// one token so the last prompt token can be re-emitted.
fn recover_decode_seed_state(
    slot: &mut SlotState,
    native_runtime: &mut NativeRuntimeHandle,
    _retained_prefix_tokens: i32,
) -> bool {
    if slot.phase != SlotPhase::Decode || !slot.generated_tokens.is_empty() {
        return true;
    }

    let Some(request) = slot.request() else {
        return true;
    };
    let max_output_tokens = request.max_output_tokens;
    let prompt_len = request.prompt_tokens.len();

    if max_output_tokens <= 0 {
        slot.phase = SlotPhase::Completed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Completed;
        }
        return true;
    }
    if prompt_len == 0 {
        slot.fail("Prompt tokenization produced no tokens, so decode had no seed token.");
        return false;
    }
    if slot.prefill_cursor < prompt_len {
        slot.phase = SlotPhase::Prefill;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
        return true;
    }
    if slot.seq_id < 0 {
        slot.fail("Decode slot lost sequence state before its first sampled token.");
        return false;
    }
    if slot.mirror.n_past <= 0 || slot.mirror.current_kv_tokens.is_empty() {
        slot.prefill_cursor = 0;
        slot.phase = SlotPhase::Prefill;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
        return true;
    }

    let Some(retained_n_past) = slot.mirror.n_past.checked_sub(1) else {
        slot.fail("Decode slot KV length underflowed during seed recovery.");
        return false;
    };
    let retained_tokens = cmp::min(
        slot.mirror.current_kv_tokens.len(),
        nonnegative_i32_to_usize(retained_n_past),
    );
    slot.mirror.current_kv_tokens.truncate(retained_tokens);
    if !reconcile_physical_state(&mut slot.mirror, slot.seq_id, native_runtime) {
        slot.fail("Failed to reconcile shared KV state for decode seed recovery.");
        return false;
    }
    slot.prefill_cursor = cmp::min(prompt_len - 1, retained_tokens);
    slot.phase = SlotPhase::Prefill;
    if let Some(request) = slot.request_mut() {
        request.lifecycle = GenerateRequestLifecycle::Running;
    }
    true
}

/// Trims llama.cpp's view of the sequence to match `state.current_kv_tokens`'s
/// length and records the new `n_past`.
fn reconcile_physical_state(
    state: &mut SequenceState,
    seq_id: llama_seq_id,
    native_runtime: &mut NativeRuntimeHandle,
) -> bool {
    if seq_id < 0 {
        return false;
    }
    let Some(current_len) = usize_to_i32(state.current_kv_tokens.len()) else {
        return false;
    };
    if !native_runtime.clear_sequence(seq_id, current_len, -1) {
        return false;
    }
    state.n_past = current_len;
    true
}
