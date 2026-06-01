//! Multimodal prefill: tokenizes prompt + image buffers via mtmd, evaluates
//! the resulting chunks into the KV cache, and seeds the first sampled token.
//!
//! Only invoked for requests that carry a `MultimodalPayload`. The text-only
//! prefill path lives in `mod.rs` (`prepare_sequence_for_prompt`).

use std::time::Instant;

use crate::native_bridge::{self, NativeRuntimeHandle};
use crate::runtime::numeric::duration_ms;
use crate::runtime::request::{GenerateRequestLifecycle, RequestQueue};
use crate::runtime::scheduler::{SlotPhase, SlotScheduler, SlotState};
use crate::runtime::REQUEST_CANCELLED_MESSAGE;

use super::numeric::{nonnegative_i32_to_usize, nonnegative_i32_to_usize_opt, usize_to_i32};
use super::text::append_token_piece_to_slot;
use super::LLAMA_SAMPLER_SAMPLE_FAILED;

/// Runs the multimodal prefill end-to-end for `slot`:
/// 1. Ensure the prompt has enough media markers; if not, prepend them.
/// 2. Evaluate prompt + image buffers through the CXX mtmd bridge.
/// 3. Sample the first decode token and emit it.
///
/// Returns `false` on any failure and clears the multimodal payload so the
/// slot can be reused without dangling payload state.
pub(super) fn run_multimodal_prefill(
    native_runtime: &mut NativeRuntimeHandle,
    batch_token_budget: i32,
    request_queue: &mut RequestQueue,
    slot: &mut SlotState,
    piece_scratch: &mut Vec<u8>,
) -> bool {
    if slot.seq_id < 0 || slot.sampler.is_none() || slot.request().is_none() {
        return false;
    }

    let (multimodal_exists, mut prompt_text, prompt_tokens_len) =
        if let Some(request) = slot.request() {
            (
                request.multimodal.is_some(),
                request.original_prompt.clone(),
                request.prompt_tokens.len(),
            )
        } else {
            (false, String::new(), 0)
        };
    if !multimodal_exists {
        return false;
    }

    let seq_id = slot.seq_id;
    let prefill_cursor = slot.prefill_cursor;
    let add_special = slot.mirror.n_past == 0;
    if !native_runtime.mtmd_ready() {
        clear_multimodal_payload(slot);
        return false;
    }

    let marker = native_bridge::mtmd_default_marker();
    let image_count = slot
        .request()
        .and_then(|request| request.multimodal.as_ref())
        .map_or(0, |multimodal| multimodal.image_buffers.len());
    if !marker.is_empty() {
        let mut marker_count = prompt_text.matches(marker.as_str()).count();
        if marker_count > image_count {
            clear_multimodal_payload(slot);
            return false;
        }
        while marker_count < image_count {
            prompt_text.insert_str(0, marker.as_str());
            marker_count += 1;
        }
    }

    let (image_bytes, image_sizes) = match flatten_image_buffers(slot) {
        Some(images) => images,
        None => {
            clear_multimodal_payload(slot);
            return false;
        }
    };

    if !native_runtime.clear_sequence(seq_id, 0, -1) {
        clear_multimodal_payload(slot);
        return false;
    }

    let prefill_start = Instant::now();
    let Some(prefill_cursor_i32) = usize_to_i32(prefill_cursor) else {
        clear_multimodal_payload(slot);
        return false;
    };
    let new_n_past = match native_runtime.mtmd_eval_images(
        &prompt_text,
        &image_bytes,
        &image_sizes,
        add_special,
        true,
        prefill_cursor_i32,
        seq_id,
        batch_token_budget,
        true,
    ) {
        Ok(new_n_past) => new_n_past,
        Err(_) => {
            clear_multimodal_payload(slot);
            return false;
        }
    };
    let prefill_end = Instant::now();
    clear_multimodal_payload(slot);

    slot.mirror.n_past = new_n_past;
    let Some(new_n_past_len) = nonnegative_i32_to_usize_opt(new_n_past) else {
        return false;
    };
    slot.mirror.current_kv_tokens.resize(new_n_past_len, 0);
    let multimodal_prefill_ms = duration_ms(prefill_start, prefill_end);
    let multimodal_token_count = new_n_past.max(0);
    let multimodal_processed_tokens = multimodal_token_count
        .saturating_sub(prefill_cursor_i32)
        .max(0);

    if let Some(request) = slot.request_mut() {
        request.input_tokens = multimodal_token_count;
        request.prefill_tokens = request
            .prefill_tokens
            .saturating_add(multimodal_processed_tokens);
        request.prefill_ms += multimodal_prefill_ms;
    }
    slot.prefill_cursor = prompt_tokens_len;

    let Some(sampler) = slot.sampler.as_mut() else {
        slot.fail("Sampler was unavailable after multimodal prefill.");
        return false;
    };
    let next_token = native_runtime.sample_with(sampler, -1);
    if next_token == native_bridge::LLAMA_TOKEN_NULL {
        slot.terminal_error_message = LLAMA_SAMPLER_SAMPLE_FAILED.to_string();
        return false;
    }
    sampler.accept(next_token, true);
    if let Some(request) = slot.request_mut() {
        request.first_sampled_token_id = next_token;
        request.first_token_at.get_or_insert_with(Instant::now);
    }
    if native_runtime.is_eog(next_token) {
        slot.terminal_error_message =
            "Model ended generation immediately after multimodal prefill.".to_string();
        return false;
    }

    slot.generated_tokens.push(next_token);
    append_token_piece_to_slot(native_runtime, next_token, slot, piece_scratch);
    slot.phase = SlotPhase::Streaming;
    if let Some(request) = slot.request_mut() {
        request.lifecycle = GenerateRequestLifecycle::Streaming;
    }
    SlotScheduler::emit_buffered_token_piece(request_queue, slot);

    if slot
        .request()
        .is_some_and(|request| request.cancel_requested)
    {
        slot.cancel(REQUEST_CANCELLED_MESSAGE);
        return true;
    }

    let reached_limit = slot.request().is_some_and(|request| {
        request.max_output_tokens > 0
            && slot.generated_tokens.len() >= nonnegative_i32_to_usize(request.max_output_tokens)
    });
    if reached_limit {
        slot.phase = SlotPhase::Completed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Completed;
        }
    } else {
        slot.phase = SlotPhase::Decode;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
    }

    true
}

fn flatten_image_buffers(slot: &SlotState) -> Option<(Vec<u8>, Vec<i32>)> {
    let multimodal = slot.request()?.multimodal.as_ref()?;
    let byte_capacity = multimodal
        .image_buffers
        .iter()
        .try_fold(0_usize, |total, image| total.checked_add(image.len()))?;
    let mut image_bytes = Vec::with_capacity(byte_capacity);
    let mut image_sizes = Vec::with_capacity(multimodal.image_buffers.len());
    for image in &multimodal.image_buffers {
        image_sizes.push(i32::try_from(image.len()).ok()?);
        image_bytes.extend_from_slice(image);
    }
    Some((image_bytes, image_sizes))
}

/// Drops the request's multimodal payload so the slot can be reused.
pub(super) fn clear_multimodal_payload(slot: &mut SlotState) {
    if let Some(request) = slot.request_mut() {
        request.multimodal = None;
    }
}
