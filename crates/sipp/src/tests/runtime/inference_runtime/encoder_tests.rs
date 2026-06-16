//! Tests the `runtime::inference_runtime::encoder` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::engine::protocol::{ModelClass, PoolingType};
use crate::error::Error;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::{PrefillKind, SlotPhase, SlotState, TerminalAction};
use crate::runtime::session::KvCacheAdmission;

fn admitted_encoder_slot(
    slot_id: usize,
    request_id: u32,
    seq_id: i32,
    prompt_tokens: Vec<i32>,
) -> SlotState {
    let mut slot = SlotState::new(slot_id);
    let mut request = GenerateRequest::new(request_id, "ctx");
    request.prompt_tokens = prompt_tokens;
    slot.attach_request(
        request,
        KvCacheAdmission {
            seq_id,
            ..KvCacheAdmission::default()
        },
    );
    slot
}

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
fn embedding_plan_uses_encoder_prefill_for_encoder_only() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.capabilities.class = ModelClass::EncoderOnly;
    runtime.capabilities.pooling_type = PoolingType::Mean;
    runtime.capabilities.embedding_context = true;

    let plan = runtime
        .embedding_slot_plan()
        .expect("encoder embedding plan");

    assert_eq!(plan.prefill, PrefillKind::Encode);
    assert_eq!(plan.terminal, TerminalAction::ReadEmbedding);
}

#[test]
fn embedding_plan_uses_decode_prefill_for_decoder_embedding_context() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.capabilities.embedding_context = true;
    runtime.capabilities.pooling_type = PoolingType::Mean;

    let plan = runtime
        .embedding_slot_plan()
        .expect("decoder embedding plan");

    assert_eq!(plan.prefill, PrefillKind::Decode);
    assert_eq!(plan.terminal, TerminalAction::ReadEmbedding);
}

#[test]
fn embedding_plan_rejects_pooling_none() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.capabilities.class = ModelClass::EncoderOnly;
    runtime.capabilities.embedding_context = true;
    runtime.capabilities.pooling_type = PoolingType::None;

    let error = runtime
        .embedding_slot_plan()
        .expect_err("pooling none embedding");

    assert!(matches!(
        error,
        Error::UnsupportedOperation {
            operation: "embed",
            ..
        }
    ));
}

#[test]
fn encoder_decoder_rewrite_preserves_source_input_token_count() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.capabilities.class = ModelClass::EncoderDecoder;
    runtime.capabilities.decoder_start_token = Some(42);
    runtime.slot_scheduler.resize(1, &mut runtime.kv_cache);

    let mut request = GenerateRequest::new(7, "ctx");
    request.prompt_tokens = vec![11, 12, 13];
    request.input_tokens = 3;
    runtime.slot_scheduler.slots[0].attach_request(request, KvCacheAdmission::default());

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

#[test]
fn encoder_batch_token_count_sums_admitted_prompt_tokens() {
    let slots = vec![
        admitted_encoder_slot(0, 7, 0, vec![11, 12, 13]),
        admitted_encoder_slot(1, 8, 1, vec![21, 22]),
    ];

    let token_count =
        super::encoder_batch_token_count(&slots, &[0, 1]).expect("encoder batch token count");

    assert_eq!(token_count, 5);
}

#[test]
fn encoder_batch_token_count_rejects_empty_prompt() {
    let slots = vec![admitted_encoder_slot(0, 7, 0, Vec::new())];

    let error = super::encoder_batch_token_count(&slots, &[0]).expect_err("empty encoder prompt");

    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("empty token slice")
    ));
}

#[test]
fn encoder_batch_token_count_rejects_missing_sequence_id() {
    let slots = vec![admitted_encoder_slot(0, 7, -1, vec![11])];

    let error = super::encoder_batch_token_count(&slots, &[0]).expect_err("missing sequence id");

    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("no sequence id")
    ));
}

#[test]
fn add_encoder_prompt_to_batch_appends_all_prompt_tokens() {
    let slot = admitted_encoder_slot(0, 7, 0, vec![11, 12, 13]);
    let mut batch = LlamaBatchBuilder::default();
    batch.ensure_capacity(3, 1).expect("batch capacity");

    super::add_encoder_prompt_to_batch(&mut batch, &slot, 3).expect("add encoder prompt");

    assert_eq!(batch.n_tokens(), 3);
}
