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

impl InferenceRuntime {
    pub(crate) fn text_generation_slot_plan(&self) -> Result<SlotExecutionPlan> {
        text_generation_slot_plan(&self.capabilities)
    }

    pub(crate) fn embedding_slot_plan(&self) -> Result<SlotExecutionPlan> {
        embedding_slot_plan(&self.capabilities)
    }

    pub(super) fn run_admission_prefill(&mut self, slot_index: usize) -> Result<()> {
        let plan = self
            .slot_scheduler
            .slots
            .get(slot_index)
            .ok_or(Error::RuntimeNotReady)?
            .plan;

        if plan.prefill != PrefillKind::Encode {
            return Ok(());
        }
        self.run_encoder_prompt_ingest(slot_index)?;
        // For `EncoderOnly + ReadEmbedding` the prompt-ingest finished the
        // whole inference: pull the pooled embedding straight off the context
        // and mark the slot terminal. `EncoderDecoder + SampleTokens` falls
        // through to the existing decode loop.
        if plan.terminal == TerminalAction::ReadEmbedding {
            self.read_slot_embedding(slot_index)?;
        }
        Ok(())
    }

    pub(super) fn fail_admitted_slot(&mut self, slot_index: usize, error: Error) {
        if let Some(slot) = self.slot_scheduler.slots.get_mut(slot_index) {
            slot.fail(format!("admission prefill failed: {error}"));
        }
    }

    fn run_encoder_prompt_ingest(&mut self, slot_index: usize) -> Result<()> {
        let (seq_id, prompt_tokens) = {
            let slot = self
                .slot_scheduler
                .slots
                .get(slot_index)
                .ok_or(Error::RuntimeNotReady)?;
            let request = slot
                .request()
                .ok_or(Error::InvalidRequest("admitted slot has no request"))?;
            if slot.seq_id < 0 {
                return Err(Error::InvalidRequest(
                    "admitted slot has no sequence id for encoder pass",
                ));
            }
            (slot.seq_id, request.prompt_tokens.clone())
        };

        if prompt_tokens.is_empty() {
            return Err(Error::InvalidRequest(
                "encoder prompt ingest received an empty token slice",
            ));
        }

        let max_tokens = i32::try_from(prompt_tokens.len())
            .map_err(|_| Error::InvalidRequest("encoder prompt exceeds i32::MAX tokens"))?;

        let mut batch = LlamaBatchBuilder::default();
        batch.ensure_capacity(max_tokens, 1)?;
        for (position, token) in prompt_tokens.iter().enumerate() {
            let position_i32 = i32::try_from(position)
                .map_err(|_| Error::InvalidRequest("encoder prompt position exceeds i32::MAX"))?;
            if !batch.add_token(*token, position_i32, seq_id, false) {
                return Err(Error::BatchCapacity {
                    capacity: max_tokens,
                    requested: max_tokens + 1,
                });
            }
        }

        let status = self
            .native_runtime
            .encode(batch.batch())
            .map_err(|error| Error::RuntimeCommand(error.to_string()))?;
        if status != 0 {
            return Err(Error::Decode(status));
        }
        if !self.native_runtime.synchronize() {
            return Err(Error::RuntimeCommand(
                "llama_synchronize() failed after encoder pass".to_string(),
            ));
        }

        self.finalize_encoder_pass(slot_index, prompt_tokens.len())
    }

    pub(super) fn run_encoder_embedding_batch(&mut self, slot_indices: &[usize]) -> Result<()> {
        let max_tokens = encoder_batch_token_count(&self.slot_scheduler.slots, slot_indices)?;
        let max_sequences = i32::try_from(slot_indices.len())
            .map_err(|_| Error::InvalidRequest("encoder batch exceeds i32::MAX sequences"))?;
        self.shared_batch_builder
            .ensure_capacity(max_tokens, max_sequences)?;
        self.shared_batch_builder.reset();

        for &slot_index in slot_indices {
            let slot = self
                .slot_scheduler
                .slots
                .get(slot_index)
                .ok_or(Error::RuntimeNotReady)?;
            add_encoder_prompt_to_batch(&mut self.shared_batch_builder, slot, max_tokens)?;
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
            let prompt_len = self
                .slot_scheduler
                .slots
                .get(slot_index)
                .and_then(SlotState::request)
                .map(|request| request.prompt_tokens.len())
                .ok_or(Error::RuntimeNotReady)?;
            if let Err(error) = self
                .finalize_encoder_pass(slot_index, prompt_len)
                .and_then(|_| self.read_slot_embedding(slot_index))
            {
                self.fail_admitted_slot(slot_index, error);
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

fn encoder_batch_token_count(slots: &[SlotState], slot_indices: &[usize]) -> Result<i32> {
    let mut total_tokens = 0_usize;
    for &slot_index in slot_indices {
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
        total_tokens = total_tokens
            .checked_add(request.prompt_tokens.len())
            .ok_or(Error::InvalidRequest(
                "encoder batch token count overflowed",
            ))?;
    }
    i32::try_from(total_tokens)
        .map_err(|_| Error::InvalidRequest("encoder batch exceeds i32::MAX tokens"))
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
