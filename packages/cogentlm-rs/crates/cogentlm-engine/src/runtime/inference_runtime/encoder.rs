//! Admission-side encoder prompt ingest.
//!
//! Text-generation slots for encoder-decoder models submit the source prompt
//! through one `llama_encode` call, then rewrite the decoder prompt to the
//! model's decoder-start token and continue through the normal decode loop.

use cogentlm_sys as ffi;

use crate::engine::protocol::ModelClass;
use crate::error::{Error, Result};
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::scheduler::{PrefillKind, SlotExecutionPlan, SlotPhase, TerminalAction};

use super::InferenceRuntime;

impl InferenceRuntime {
    pub(crate) fn text_generation_slot_plan(&self) -> Result<SlotExecutionPlan> {
        match (self.capabilities.class, self.capabilities.embedding_context) {
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

    pub(super) fn run_admission_prefill(&mut self, slot_index: usize) -> Result<()> {
        let plan = self.text_generation_slot_plan()?;
        let slot = self
            .slot_scheduler
            .slots
            .get_mut(slot_index)
            .ok_or(Error::RuntimeNotReady)?;
        slot.plan = plan;

        if plan.prefill != PrefillKind::Encode {
            return Ok(());
        }
        self.run_encoder_prompt_ingest(slot_index)
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

        let status = unsafe { ffi::cogent_llama_encode(self.shared_context, &batch.batch) };
        if status != 0 {
            return Err(Error::Decode(status));
        }
        if !unsafe { ffi::cogent_llama_synchronize(self.shared_context) } {
            return Err(Error::RuntimeCommand(
                "llama_synchronize() failed after encoder pass".to_string(),
            ));
        }

        self.finalize_encoder_pass(slot_index, prompt_tokens.len())
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

#[cfg(test)]
mod tests {
    use crate::engine::protocol::{ModelClass, PoolingType};
    use crate::error::Error;
    use crate::runtime::config::NativeRuntimeConfig;
    use crate::runtime::inference_runtime::tests::runtime_tests::test_runtime;
    use crate::runtime::request::GenerateRequest;
    use crate::runtime::scheduler::{PrefillKind, SlotPhase, TerminalAction};
    use crate::runtime::session::SequenceState;

    #[test]
    fn text_generation_plan_uses_encoder_prefill_for_encoder_decoder() {
        let mut runtime = test_runtime(NativeRuntimeConfig::default());
        runtime.capabilities.class = ModelClass::EncoderDecoder;
        runtime.capabilities.decoder_start_token = Some(0);

        let plan = runtime
            .text_generation_slot_plan()
            .expect("encoder-decoder text plan");

        assert_eq!(plan.prefill, PrefillKind::Encode);
        assert_eq!(plan.terminal, TerminalAction::SampleTokens);
    }

    #[test]
    fn text_generation_plan_rejects_encoder_only() {
        let mut runtime = test_runtime(NativeRuntimeConfig::default());
        runtime.capabilities.class = ModelClass::EncoderOnly;
        runtime.capabilities.pooling_type = PoolingType::Mean;
        runtime.capabilities.embedding_context = true;

        let error = runtime
            .text_generation_slot_plan()
            .expect_err("encoder-only query");

        assert!(matches!(
            error,
            Error::UnsupportedOperation {
                operation: "query",
                ..
            }
        ));
    }

    #[test]
    fn text_generation_plan_rejects_decoder_embedding_context() {
        let mut runtime = test_runtime(NativeRuntimeConfig::default());
        runtime.capabilities.embedding_context = true;

        let error = runtime
            .text_generation_slot_plan()
            .expect_err("decoder embedding context query");

        assert!(matches!(
            error,
            Error::UnsupportedOperation {
                operation: "query",
                ..
            }
        ));
    }

    #[test]
    fn encoder_decoder_rewrite_preserves_source_input_token_count() {
        let mut runtime = test_runtime(NativeRuntimeConfig::default());
        runtime.capabilities.class = ModelClass::EncoderDecoder;
        runtime.capabilities.decoder_start_token = Some(42);
        runtime.slot_scheduler.resize(1);

        let mut request = GenerateRequest::new(7, "ctx");
        request.prompt_tokens = vec![11, 12, 13];
        request.input_tokens = 3;
        runtime.slot_scheduler.slots[0].attach_request(request, SequenceState::default());

        runtime
            .finalize_encoder_pass(0, 3)
            .expect("finalize encoder-decoder");

        let slot = &runtime.slot_scheduler.slots[0];
        let request = slot.request().expect("slot request");
        assert_eq!(request.prompt_tokens, vec![42]);
        assert_eq!(request.input_tokens, 3);
        assert_eq!(slot.prefill_cursor, 0);
        assert_eq!(slot.phase, SlotPhase::Prefill);
    }
}
