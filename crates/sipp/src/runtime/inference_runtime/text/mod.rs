//! Token-to-text decoding, stop-sequence matching, and incremental UTF-8
//! reassembly that drive the slot's output buffer.

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::llama_token;
use crate::runtime::scheduler::{SlotPhase, SlotState};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../../tests/runtime/inference_runtime/text_tests.rs"]
mod text_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Decode `token` into UTF-8 and push it onto the slot's emission/output
/// buffers. Marks the slot as `Failed` on tokenization error.
#[inline]
pub(super) fn append_token_piece_to_slot(
    native_runtime: &NativeRuntimeHandle,
    token: llama_token,
    slot: &mut SlotState,
    piece_scratch: &mut Vec<u8>,
) {
    if !token_to_piece_into(native_runtime, token, false, piece_scratch) {
        slot.fail("Failed to convert sampled token to text piece.");
        return;
    }

    slot.pending_utf8_bytes.extend_from_slice(piece_scratch);
    let tail_len = incomplete_utf8_tail_length(&slot.pending_utf8_bytes);
    let complete_len = slot.pending_utf8_bytes.len().saturating_sub(tail_len);
    if complete_len > 0 {
        let complete = String::from_utf8_lossy(&slot.pending_utf8_bytes[..complete_len]);
        slot.pending_emission_text.push_str(&complete);
        slot.pending_utf8_bytes.drain(..complete_len);
    }

    if !slot.pending_emission_text.is_empty() {
        slot.buffered_output_text
            .push_str(&std::mem::take(&mut slot.pending_emission_text));
    }
}

/// Checks the slot's output for any of the request's stop strings, splitting
/// the search across the already-emitted text and the still-buffered tail.
/// Returns `true` if a stop was found and the slot was transitioned to
/// `Completed`.
pub(super) fn apply_stop_sequences_to_slot(slot: &mut SlotState) -> bool {
    let (stop_index, max_hold) = if let Some(request) = slot.request() {
        if request.stop.is_empty() {
            return false;
        }
        let stops = &request.stop;
        let stop_index =
            earliest_stop_index_split(&slot.output_text, &slot.buffered_output_text, stops);
        let max_hold = if stop_index.is_none() {
            stops
                .iter()
                .map(|stop| stop.len().saturating_sub(1))
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        (stop_index, max_hold)
    } else {
        return false;
    };

    if let Some(stop_index) = stop_index {
        let output_len = slot.output_text.len();
        if stop_index <= output_len {
            slot.output_text.truncate(stop_index);
            slot.buffered_output_text.clear();
        } else {
            truncate_to_char_boundary(&mut slot.buffered_output_text, stop_index - output_len);
        }
        slot.pending_emission_text.clear();
        slot.pending_utf8_bytes.clear();
        slot.phase = SlotPhase::Completed;
        true
    } else {
        if max_hold > 0 && slot.buffered_output_text.len() > max_hold {
            let raw_split = slot.buffered_output_text.len() - max_hold;
            let split = floor_char_boundary(&slot.buffered_output_text, raw_split);
            if split > 0 && split < slot.buffered_output_text.len() {
                let held = slot.buffered_output_text.split_off(split);
                slot.pending_emission_text.insert_str(0, &held);
            }
        }
        false
    }
}

/// Finds the earliest absolute index in `output ++ buffered` where any of
/// `stops` matches. Cross-boundary matches are supported by searching from a
/// short suffix of `output`.
pub(super) fn earliest_stop_index_split(
    output: &str,
    buffered: &str,
    stops: &[String],
) -> Option<usize> {
    let output_len = output.len();
    stops
        .iter()
        .filter_map(|stop| {
            if stop.is_empty() {
                return None;
            }
            let suffix_len = stop.len().saturating_sub(1);
            let mut suffix_start = output_len.saturating_sub(suffix_len);
            while suffix_start > 0 && !output.is_char_boundary(suffix_start) {
                suffix_start -= 1;
            }
            let suffix = &output[suffix_start..];
            let mut search_space = String::with_capacity(suffix.len() + buffered.len());
            search_space.push_str(suffix);
            search_space.push_str(buffered);
            search_space.find(stop).map(|idx| suffix_start + idx)
        })
        .min()
}

pub(super) fn truncate_to_char_boundary(value: &mut String, max_len: usize) {
    let boundary = floor_char_boundary(value, max_len.min(value.len()));
    value.truncate(boundary);
}

pub(super) fn floor_char_boundary(value: &str, mut index: usize) -> usize {
    index = index.min(value.len());
    while index > 0 && !value.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub(super) fn flush_pending_utf8(slot: &mut SlotState) {
    if !slot.pending_emission_text.is_empty() {
        slot.buffered_output_text
            .push_str(&std::mem::take(&mut slot.pending_emission_text));
    }
    if slot.pending_utf8_bytes.is_empty() {
        return;
    }
    let pending = String::from_utf8_lossy(&slot.pending_utf8_bytes);
    slot.buffered_output_text.push_str(&pending);
    slot.pending_utf8_bytes.clear();
}

/// Fills `scratch` with the bytes of `token`'s text piece. Returns `false`
/// on error so callers can act without `Result` boxing. The scratch vector
/// is reused across calls (per-token work is `resize` + `truncate`).
#[inline]
pub(super) fn token_to_piece_into(
    native_runtime: &NativeRuntimeHandle,
    token: llama_token,
    special: bool,
    scratch: &mut Vec<u8>,
) -> bool {
    scratch.clear();
    if token < 0 {
        return false;
    }
    let Ok(piece) = native_runtime.token_to_piece_bytes(token, special) else {
        return false;
    };
    scratch.extend(piece);
    true
}

/// Returns the number of trailing bytes that form an incomplete UTF-8 code
/// point at the end of `data`. Used to hold back partial sequences so token
/// emission only flushes complete characters.
pub(super) fn incomplete_utf8_tail_length(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let max_lookback = data.len().min(4);
    for offset in 1..=max_lookback {
        let byte = data[data.len() - offset];
        if (byte & 0xC0) == 0x80 {
            continue;
        }
        let expected = if (byte & 0x80) == 0 {
            1
        } else if (byte & 0xE0) == 0xC0 {
            2
        } else if (byte & 0xF0) == 0xE0 {
            3
        } else if (byte & 0xF8) == 0xF0 {
            4
        } else {
            return 0;
        };
        if offset >= expected {
            return 0;
        }
        return offset;
    }
    max_lookback
}
