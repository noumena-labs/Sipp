//! Multimodal prefill: tokenizes prompt + image bitmaps via mtmd, evaluates
//! the resulting chunks into the KV cache, and seeds the first sampled token.
//!
//! Only invoked for requests that carry a `MultimodalPayload`. The text-only
//! prefill path lives in `mod.rs` (`prepare_sequence_for_prompt`).

use std::ffi::{CStr, CString};
use std::time::Instant;

use cogentlm_sys as ffi;

use crate::runtime::numeric::duration_ms;
use crate::runtime::request::{GenerateRequestLifecycle, RequestQueue};
use crate::runtime::scheduler::{SlotPhase, SlotScheduler, SlotState};
use crate::runtime::REQUEST_CANCELLED_MESSAGE;

use super::numeric::{nonnegative_i32_to_usize, nonnegative_i32_to_usize_opt, usize_to_i32};
use super::text::append_token_piece_to_slot;
use super::LLAMA_SAMPLER_SAMPLE_FAILED;

/// RAII guard for `cogent_mtmd_bitmap`. Frees the bitmap on drop.
struct BitmapGuard(*mut ffi::cogent_mtmd_bitmap);

impl Drop for BitmapGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::cogent_mtmd_bitmap_free(self.0) };
        }
    }
}

/// RAII guard for `cogent_mtmd_input_chunks`. Frees the chunks on drop.
struct ChunksGuard(*mut ffi::cogent_mtmd_input_chunks);

impl Drop for ChunksGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::cogent_mtmd_input_chunks_free(self.0) };
        }
    }
}

/// Runs the multimodal prefill end-to-end for `slot`:
/// 1. Build bitmaps from the request's image buffers (RAII-guarded).
/// 2. Ensure the prompt has enough media markers; if not, prepend them.
/// 3. Tokenize via `cogent_mtmd_tokenize` and evaluate into the KV cache.
/// 4. Sample the first decode token and emit it.
///
/// Returns `false` on any failure (and clears the multimodal payload so the
/// slot can be reused without dangling FFI state).
pub(super) fn run_multimodal_prefill(
    mtmd_context: *mut ffi::cogent_mtmd_context,
    shared_context: *mut ffi::llama_context,
    vocab: *const ffi::llama_vocab,
    batch_token_budget: i32,
    request_queue: &mut RequestQueue,
    slot: &mut SlotState,
    piece_scratch: &mut Vec<i8>,
) -> bool {
    if mtmd_context.is_null()
        || shared_context.is_null()
        || vocab.is_null()
        || slot.seq_id < 0
        || slot.sampler.is_none()
        || slot.request().is_none()
    {
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

    let mut bitmaps = Vec::new();
    let mut success = true;
    if let Some(request) = slot.request() {
        if let Some(multimodal) = request.multimodal.as_ref() {
            bitmaps.reserve(multimodal.image_buffers.len());
            for buffer in &multimodal.image_buffers {
                let bitmap = unsafe {
                    ffi::cogent_mtmd_bitmap_init_from_buf(
                        mtmd_context,
                        buffer.as_ptr(),
                        buffer.len(),
                    )
                };
                if bitmap.is_null() {
                    success = false;
                    break;
                }
                bitmaps.push(BitmapGuard(bitmap));
            }
        }
    }
    if !success {
        clear_multimodal_payload(slot);
        return false;
    }
    let marker = unsafe { ffi::cogent_mtmd_default_marker() };
    if !marker.is_null() {
        let marker = unsafe { CStr::from_ptr(marker) }.to_string_lossy();
        if !marker.is_empty() {
            let mut marker_count = prompt_text.matches(marker.as_ref()).count();
            if marker_count > bitmaps.len() {
                clear_multimodal_payload(slot);
                return false;
            }
            while marker_count < bitmaps.len() {
                prompt_text.insert_str(0, marker.as_ref());
                marker_count += 1;
            }
        }
    }

    let Ok(prompt_text) = CString::new(prompt_text) else {
        clear_multimodal_payload(slot);
        return false;
    };
    let chunks = ChunksGuard(unsafe { ffi::cogent_mtmd_input_chunks_init() });
    if chunks.0.is_null() {
        clear_multimodal_payload(slot);
        return false;
    }
    let bitmap_ptrs: Vec<*const ffi::cogent_mtmd_bitmap> =
        bitmaps.iter().map(|bitmap| bitmap.0.cast_const()).collect();
    let tokenized = unsafe {
        ffi::cogent_mtmd_tokenize(
            mtmd_context,
            chunks.0,
            prompt_text.as_ptr(),
            add_special,
            true,
            bitmap_ptrs.as_ptr(),
            bitmap_ptrs.len(),
        )
    };
    if !tokenized {
        clear_multimodal_payload(slot);
        return false;
    }

    let memory = unsafe { ffi::llama_get_memory(shared_context) };
    if !unsafe { ffi::llama_memory_seq_rm(memory, seq_id, 0, -1) } {
        clear_multimodal_payload(slot);
        return false;
    }

    let prefill_start = Instant::now();
    let mut new_n_past = 0_i32;
    let eval_status = unsafe {
        let Some(prefill_cursor) = usize_to_i32(prefill_cursor) else {
            clear_multimodal_payload(slot);
            return false;
        };
        ffi::cogent_mtmd_eval_chunks(
            mtmd_context,
            shared_context,
            chunks.0,
            prefill_cursor,
            seq_id,
            batch_token_budget,
            true,
            &mut new_n_past,
        )
    };
    let sync_ok = unsafe { ffi::cogent_llama_synchronize(shared_context) };
    let prefill_end = Instant::now();
    clear_multimodal_payload(slot);
    if eval_status != 0 || !sync_ok {
        return false;
    }

    slot.mirror.n_past = new_n_past;
    let Some(new_n_past_len) = nonnegative_i32_to_usize_opt(new_n_past) else {
        return false;
    };
    slot.mirror.current_kv_tokens.resize(new_n_past_len, 0);
    let multimodal_prefill_ms = duration_ms(prefill_start, prefill_end);
    let multimodal_token_count = new_n_past.max(0);
    let prefill_cursor_i32 = usize_to_i32(prefill_cursor).unwrap_or(i32::MAX);
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

    let Some(sampler) = slot.sampler else {
        slot.fail("Sampler was unavailable after multimodal prefill.");
        return false;
    };
    let next_token =
        unsafe { ffi::cogent_common_sampler_sample(sampler.as_ptr(), shared_context, -1) };
    if next_token == ffi::LLAMA_TOKEN_NULL {
        slot.terminal_error_message = LLAMA_SAMPLER_SAMPLE_FAILED.to_string();
        return false;
    }
    unsafe {
        ffi::cogent_common_sampler_accept(sampler.as_ptr(), next_token, true);
    }
    if let Some(request) = slot.request_mut() {
        request.first_sampled_token_id = next_token;
        request.first_token_at.get_or_insert_with(Instant::now);
    }
    if unsafe { ffi::llama_vocab_is_eog(vocab, next_token) } {
        slot.terminal_error_message =
            "Model ended generation immediately after multimodal prefill.".to_string();
        return false;
    }

    slot.generated_tokens.push(next_token);
    append_token_piece_to_slot(vocab, next_token, slot, piece_scratch);
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

/// Drops the request's multimodal payload so the slot can be reused.
pub(super) fn clear_multimodal_payload(slot: &mut SlotState) {
    if let Some(request) = slot.request_mut() {
        request.multimodal = None;
    }
}
