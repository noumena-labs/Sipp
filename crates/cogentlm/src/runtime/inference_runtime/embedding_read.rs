//! Reading and post-processing embedding outputs from the llama context.
//!
//! Two entry points for each model class, called at different points in the
//! slot lifecycle:
//! - Encoder-only slots call this immediately after `cogent_llama_encode` +
//!   sync (admission-time).
//! - Decoder-only slots loaded with `embeddings=true` call this once the
//!   standard decode prefill reaches the end of the prompt.

use crate::engine::protocol::PoolingType;
use crate::error::{Error, Result};
use crate::runtime::scheduler::{SlotEmbeddingOutput, SlotPhase};

use super::InferenceRuntime;

impl InferenceRuntime {
    pub(super) fn read_slot_embedding(&mut self, slot_index: usize) -> Result<()> {
        let (seq_id, normalize_requested) = slot_inputs(self, slot_index)?;
        let values = self.read_pooled_embedding(seq_id)?;
        let normalized =
            apply_normalization(values, self.capabilities.pooling_type, normalize_requested);
        self.complete_slot_with_embedding(slot_index, normalized);
        Ok(())
    }

    fn read_pooled_embedding(&self, seq_id: i32) -> Result<Vec<f32>> {
        if seq_id < 0 {
            return Err(Error::InvalidRequest("embedding slot has no sequence id"));
        }
        let dimensions = self.capabilities.embedding_dimensions;
        if dimensions <= 0 {
            return Err(Error::UnsupportedOperation {
                operation: "embed",
                reason: "model reports zero embedding dimensions; embedding output is unsupported"
                    .to_string(),
            });
        }

        let values = self.native_runtime.embeddings_seq(seq_id)?;
        let len = dimensions as usize;
        if values.len() != len {
            return Err(Error::RuntimeCommand(format!(
                "llama embeddings returned {} values, expected {len}",
                values.len()
            )));
        }
        Ok(values)
    }

    fn complete_slot_with_embedding(&mut self, slot_index: usize, output: SlotEmbeddingOutput) {
        if let Some(slot) = self.slot_scheduler.slots.get_mut(slot_index) {
            slot.embedding_output = Some(output);
            slot.phase = SlotPhase::Completed;
        }
    }
}

fn slot_inputs(runtime: &InferenceRuntime, slot_index: usize) -> Result<(i32, bool)> {
    let slot = runtime
        .slot_scheduler
        .slots
        .get(slot_index)
        .ok_or(Error::RuntimeNotReady)?;
    let request = slot
        .request()
        .ok_or(Error::InvalidRequest("embedding slot has no request"))?;
    let embed_options = request.embed_options.as_ref().ok_or(Error::InvalidRequest(
        "embedding slot reached ReadEmbedding without embed options",
    ))?;
    Ok((slot.seq_id, embed_options.normalize))
}

/// Apply L2 normalization if requested and the pooling type allows it. `Rank`
/// pooling returns raw classifier scores and is exempt by design.
fn apply_normalization(
    mut values: Vec<f32>,
    pooling: PoolingType,
    normalize_requested: bool,
) -> SlotEmbeddingOutput {
    let normalized = normalize_requested && pooling != PoolingType::Rank;
    if normalized {
        l2_normalize(&mut values);
    }
    SlotEmbeddingOutput {
        values,
        pooling,
        normalized,
    }
}

/// In-place L2 normalization. Mirrors llama.cpp's `common_embd_normalize` for
/// the L2 case: divide each element by `sqrt(sum_of_squares)`, zero-norm input
/// stays zero.
fn l2_normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let scale = if norm > 0.0 { (1.0 / norm) as f32 } else { 0.0 };
    for value in values.iter_mut() {
        *value *= scale;
    }
}

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/embedding_read_tests.rs"]
mod embedding_read_tests;
