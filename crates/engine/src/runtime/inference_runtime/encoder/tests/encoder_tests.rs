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
