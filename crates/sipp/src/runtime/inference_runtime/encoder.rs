//! Admission-side encoder prompt ingest.
//!
//! Text-generation slots for encoder-decoder models submit the source prompt
//! through one `llama_encode` call, then rewrite the decoder prompt to the
//! model's decoder-start token and continue through the normal decode loop.

use crate::engine::protocol::{ModelClass, PoolingType};
use crate::error::{Error, Result};
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::{
    PrefillKind, SlotExecutionPlan, SlotPhase, SlotState, TerminalAction,
};

use super::capabilities::RuntimeModelCapabilities;
use super::InferenceRuntime;

#[derive(Debug, PartialEq, Eq)]
enum EncoderAdmissionBatch {
    Ready { end: usize, token_count: i32 },
    Oversized { requested: i32 },
}

impl InferenceRuntime {
    pub(crate) fn text_generation_slot_plan(&self) -> Result<SlotExecutionPlan> {
        text_generation_slot_plan(&self.capabilities)
    }

    pub(crate) fn embedding_slot_plan(&self) -> Result<SlotExecutionPlan> {
        embedding_slot_plan(&self.capabilities)
    }

    fn fail_admitted_slot(&mut self, slot_index: usize, error: &Error) {
        if let Some(slot) = self.slot_scheduler.slots.get_mut(slot_index) {
            slot.fail(format!("admission prefill failed: {error}"));
        }
    }

    pub(super) fn fail_admitted_slots(&mut self, slot_indices: &[usize], error: &Error) {
        for &slot_index in slot_indices {
            self.fail_admitted_slot(slot_index, error);
        }
    }

    pub(super) fn run_encoder_admission_batches(&mut self, slot_indices: &[usize]) -> Result<()> {
        let mut start = 0;
        while start < slot_indices.len() {
            match next_encoder_admission_batch(
                &self.slot_scheduler.slots,
                slot_indices,
                start,
                self.resolved_limits.n_ubatch,
            )? {
                EncoderAdmissionBatch::Ready { end, token_count } => {
                    let batch_slots = &slot_indices[start..end];
                    if let Err(error) = self.run_encoder_admission_batch(batch_slots, token_count) {
                        self.fail_admitted_slots(batch_slots, &error);
                    }
                    start = end;
                }
                EncoderAdmissionBatch::Oversized { requested } => {
                    let error = Error::BatchCapacity {
                        capacity: self.resolved_limits.n_ubatch,
                        requested,
                    };
                    self.fail_admitted_slot(slot_indices[start], &error);
                    start += 1;
                }
            }
        }
        Ok(())
    }

    fn run_encoder_admission_batch(
        &mut self,
        slot_indices: &[usize],
        token_count: i32,
    ) -> Result<()> {
        let max_sequences = i32::try_from(slot_indices.len())
            .map_err(|_| Error::InvalidRequest("encoder batch exceeds i32::MAX sequences"))?;
        self.shared_batch_builder
            .ensure_capacity(token_count, max_sequences)?;
        self.shared_batch_builder.reset();

        for &slot_index in slot_indices {
            let slot = self
                .slot_scheduler
                .slots
                .get(slot_index)
                .ok_or(Error::RuntimeNotReady)?;
            add_encoder_prompt_to_batch(&mut self.shared_batch_builder, slot, token_count)?;
        }

        let status = self
            .native_runtime
            .encode(self.shared_batch_builder.batch())
            .map_err(|error| Error::RuntimeCommand(error.to_string()))?;
        if status != 0 {
            return Err(Error::Decode(status));
        }
        if !self.native_runtime.synchronize() {
            return Err(Error::RuntimeCommand(
                "llama_synchronize() failed after encoder pass".to_string(),
            ));
        }

        for &slot_index in slot_indices {
            let slot = self
                .slot_scheduler
                .slots
                .get(slot_index)
                .ok_or(Error::RuntimeNotReady)?;
            let terminal = slot.plan.terminal;
            let prompt_len = slot
                .request()
                .map(|request| request.prompt_tokens.len())
                .ok_or(Error::InvalidRequest("admitted slot has no request"))?;

            let result = self
                .finalize_encoder_pass(slot_index, prompt_len)
                .and_then(|_| {
                    if terminal == TerminalAction::ReadEmbedding {
                        self.read_slot_embedding(slot_index)
                    } else {
                        Ok(())
                    }
                });
            if let Err(error) = result {
                self.fail_admitted_slot(slot_index, &error);
            }
        }
        Ok(())
    }

    /// Rewrite the prompt for encoder-decoder models (so the existing decode
    /// loop sees a single decoder-start token), or short-circuit straight to
    /// the terminal embedding read for encoder-only models.
    fn finalize_encoder_pass(&mut self, slot_index: usize, prompt_len: usize) -> Result<()> {
        let class = self.capabilities.class;
        let slot = self
            .slot_scheduler
            .slots
            .get_mut(slot_index)
            .ok_or(Error::RuntimeNotReady)?;

        match class {
            ModelClass::EncoderDecoder => {
                let start = self.capabilities.decoder_start_token.ok_or_else(|| {
                    Error::UnsupportedOperation {
                        operation: "query",
                        reason: "encoder-decoder model has no decoder_start_token; \
                                 cannot drive the decoder loop"
                            .to_string(),
                    }
                })?;
                if let Some(request) = slot.request_mut() {
                    request.prompt_tokens.clear();
                    request.prompt_tokens.push(start);
                }
                slot.prefill_cursor = 0;
                slot.phase = SlotPhase::Prefill;
            }
            ModelClass::EncoderOnly => {
                slot.prefill_cursor = prompt_len;
                slot.phase = SlotPhase::Prefill;
            }
            ModelClass::DecoderOnly => {
                debug_assert!(false, "encoder pass invoked on decoder-only model");
            }
        }
        Ok(())
    }
}

fn next_encoder_admission_batch(
    slots: &[SlotState],
    slot_indices: &[usize],
    start: usize,
    max_tokens: i32,
) -> Result<EncoderAdmissionBatch> {
    let mut end = start;
    let mut batch_tokens = 0;

    for (position, &slot_index) in slot_indices.iter().enumerate().skip(start) {
        let prompt_tokens = encoder_prompt_token_count(slots, slot_index)?;
        if prompt_tokens > max_tokens {
            if position == start {
                return Ok(EncoderAdmissionBatch::Oversized {
                    requested: prompt_tokens,
                });
            }
            break;
        }
        if prompt_tokens > max_tokens - batch_tokens {
            break;
        }
        batch_tokens += prompt_tokens;
        end = position + 1;
    }

    Ok(EncoderAdmissionBatch::Ready {
        end,
        token_count: batch_tokens,
    })
}

fn encoder_prompt_token_count(slots: &[SlotState], slot_index: usize) -> Result<i32> {
    let slot = slots.get(slot_index).ok_or(Error::RuntimeNotReady)?;
    let request = slot
        .request()
        .ok_or(Error::InvalidRequest("admitted slot has no request"))?;
    if slot.seq_id < 0 {
        return Err(Error::InvalidRequest(
            "admitted slot has no sequence id for encoder pass",
        ));
    }
    if request.prompt_tokens.is_empty() {
        return Err(Error::InvalidRequest(
            "encoder prompt ingest received an empty token slice",
        ));
    }
    i32::try_from(request.prompt_tokens.len())
        .map_err(|_| Error::InvalidRequest("encoder prompt exceeds i32::MAX tokens"))
}

fn add_encoder_prompt_to_batch(
    batch: &mut LlamaBatchBuilder,
    slot: &SlotState,
    max_tokens: i32,
) -> Result<()> {
    let request = slot
        .request()
        .ok_or(Error::InvalidRequest("admitted slot has no request"))?;
    for (position, token) in request.prompt_tokens.iter().enumerate() {
        let position_i32 = i32::try_from(position)
            .map_err(|_| Error::InvalidRequest("encoder prompt position exceeds i32::MAX"))?;
        if !batch.add_token(*token, position_i32, slot.seq_id, false) {
            return Err(Error::BatchCapacity {
                capacity: max_tokens,
                requested: max_tokens + 1,
            });
        }
    }
    Ok(())
}

pub(super) fn resolve_request_slot_plan_for_capabilities(
    capabilities: &RuntimeModelCapabilities,
    request: &GenerateRequest,
) -> Result<SlotExecutionPlan> {
    if request.embed_options.is_some() {
        embedding_slot_plan(capabilities)
    } else {
        text_generation_slot_plan(capabilities)
    }
}

fn text_generation_slot_plan(capabilities: &RuntimeModelCapabilities) -> Result<SlotExecutionPlan> {
    match (capabilities.class, capabilities.embedding_context) {
        (ModelClass::EncoderOnly, _) => Err(Error::UnsupportedOperation {
            operation: "query",
            reason: "encoder-only models do not generate text; use embed() for vector output"
                .to_string(),
        }),
        (ModelClass::DecoderOnly, true) => Err(Error::UnsupportedOperation {
            operation: "query",
            reason: "this decoder-only model was loaded as an embedding context; load a \
                     text-generation context for query()"
                .to_string(),
        }),
        (ModelClass::DecoderOnly, false) => Ok(SlotExecutionPlan {
            prefill: PrefillKind::Decode,
            terminal: TerminalAction::SampleTokens,
        }),
        (ModelClass::EncoderDecoder, _) => Ok(SlotExecutionPlan {
            prefill: PrefillKind::Encode,
            terminal: TerminalAction::SampleTokens,
        }),
    }
}

fn embedding_slot_plan(capabilities: &RuntimeModelCapabilities) -> Result<SlotExecutionPlan> {
    match (capabilities.class, capabilities.embedding_context) {
        (ModelClass::EncoderOnly, _) => pooled_embedding_plan(capabilities, PrefillKind::Encode),
        (ModelClass::DecoderOnly, true) => pooled_embedding_plan(capabilities, PrefillKind::Decode),
        (ModelClass::DecoderOnly, false) => Err(Error::UnsupportedOperation {
            operation: "embed",
            reason: "decoder-only runtime was not loaded with embeddings=true; reload with \
                     an embedding context or use query() for text output"
                .to_string(),
        }),
        (ModelClass::EncoderDecoder, _) => Err(Error::UnsupportedOperation {
            operation: "embed",
            reason: "encoder-decoder models do not produce embeddings via this runtime".to_string(),
        }),
    }
}

fn pooled_embedding_plan(
    capabilities: &RuntimeModelCapabilities,
    prefill: PrefillKind,
) -> Result<SlotExecutionPlan> {
    if capabilities.pooling_type == PoolingType::None {
        return Err(Error::UnsupportedOperation {
            operation: "embed",
            reason: "pooling=none produces per-token embeddings; embed() requires a pooled \
                     output (mean, cls, last, or rank)"
                .to_string(),
        });
    }
    Ok(SlotExecutionPlan {
        prefill,
        terminal: TerminalAction::ReadEmbedding,
    })
}

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/encoder_tests.rs"]
mod encoder_tests;
