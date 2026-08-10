//! Multimodal prefill: tokenizes prompt + media buffers via mtmd, evaluates
//! the resulting chunks into the KV cache, and seeds the first sampled token.
//!
//! Only invoked for requests that carry a `MultimodalPayload`. The text-only
//! prefill path lives in `mod.rs` (`prepare_sequence_for_prompt`).

use std::borrow::Cow;
use std::time::Instant;

use crate::native_bridge::{self, NativeRuntimeHandle};
use crate::runtime::numeric::duration_ms;
use crate::runtime::request::{GenerateRequestLifecycle, RequestQueue};
use crate::runtime::scheduler::{SlotPhase, SlotScheduler, SlotState};
use crate::runtime::REQUEST_CANCELLED_MESSAGE;

use super::numeric::{nonnegative_i32_to_usize, nonnegative_i32_to_usize_opt, usize_to_i32};
use super::text::append_token_piece_to_slot;
use super::LLAMA_SAMPLER_SAMPLE_FAILED;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/multimodal_tests.rs"]
mod multimodal_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Run multimodal prefill and seed the first decoded token.
pub(super) fn run_multimodal_prefill(
    native_runtime: &mut NativeRuntimeHandle,
    batch_token_budget: i32,
    request_queue: &mut RequestQueue,
    slot: &mut SlotState,
    piece_scratch: &mut Vec<u8>,
) -> bool {
    if slot.seq_id < 0 || slot.sampler.is_none() {
        return false;
    }

    let Some(request) = slot.request_mut() else {
        return false;
    };
    let Some(multimodal) = request.multimodal.take() else {
        return false;
    };
    let mut prompt_text = std::mem::take(&mut request.original_prompt);
    let prompt_tokens_len = request.prompt_tokens.len();
    let media_buffers = multimodal.media_buffers;

    let seq_id = slot.seq_id;
    let prefill_cursor = slot.prefill_cursor;
    let add_special = slot.mirror.n_past == 0;
    if !native_runtime.mtmd_ready() {
        return false;
    }

    let marker = native_bridge::mtmd_default_marker();
    let media_count = media_buffers.len();
    if !marker.is_empty() {
        let mut marker_count = prompt_text.matches(marker.as_str()).count();
        if marker_count > media_count {
            return false;
        }
        while marker_count < media_count {
            prompt_text.insert_str(0, marker.as_str());
            marker_count += 1;
        }
    }
    let (media_bytes, media_sizes) = match media_parts(&media_buffers) {
        Some(media) => media,
        None => return false,
    };

    if !native_runtime.clear_sequence(seq_id, 0, -1) {
        return false;
    }

    let prefill_start = Instant::now();
    let Some(prefill_cursor_i32) = usize_to_i32(prefill_cursor) else {
        return false;
    };
    let new_n_past = match native_runtime.mtmd_eval_media(
        &prompt_text,
        media_bytes.as_ref(),
        &media_sizes,
        add_special,
        true,
        prefill_cursor_i32,
        seq_id,
        batch_token_budget,
        true,
    ) {
        Ok(new_n_past) => new_n_past,
        Err(_) => return false,
    };
    let prefill_end = Instant::now();

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
    slot.phase = SlotPhase::EmitBuffered;
    if let Some(request) = slot.request_mut() {
        request.lifecycle = GenerateRequestLifecycle::Decoding;
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

fn media_parts(media_buffers: &[Vec<u8>]) -> Option<(Cow<'_, [u8]>, Vec<i32>)> {
    let mut media_sizes = Vec::with_capacity(media_buffers.len());
    for media in media_buffers {
        media_sizes.push(i32::try_from(media.len()).ok()?);
    }
    if let [media] = media_buffers {
        return Some((Cow::Borrowed(media), media_sizes));
    }
    let capacity = media_buffers
        .iter()
        .try_fold(0_usize, |total, media| total.checked_add(media.len()))?;
    let mut flattened = Vec::with_capacity(capacity);
    for media in media_buffers {
        flattened.extend_from_slice(media);
    }
    Some((Cow::Owned(flattened), media_sizes))
}
