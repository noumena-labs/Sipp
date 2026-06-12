//! Tests the `runtime::inference_runtime::lifecycle` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::engine::protocol::ModelClass;
use crate::error::Error;
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{NativeRuntimeConfig, ResolvedRuntimeLimits};

#[test]
fn encoder_only_enables_embedding_context_before_common_params() {
    let mut config = NativeRuntimeConfig::default();

    apply_model_class_defaults(&mut config, ModelClass::EncoderOnly).expect("defaults");

    assert_eq!(config.context.embeddings, Some(true));
}

#[test]
fn decoder_only_and_encoder_decoder_defaults_preserve_supported_configs() {
    let mut decoder_config = NativeRuntimeConfig::default();
    apply_model_class_defaults(&mut decoder_config, ModelClass::DecoderOnly).expect("decoder");
    assert_eq!(decoder_config.context.embeddings, None);

    let mut encoder_decoder_config = NativeRuntimeConfig::default();
    apply_model_class_defaults(&mut encoder_decoder_config, ModelClass::EncoderDecoder)
        .expect("encoder-decoder defaults");
    assert_eq!(encoder_decoder_config.context.embeddings, None);
    assert_eq!(encoder_decoder_config.context.n_parallel, Some(1));
}

#[test]
fn encoder_decoder_rejects_embedding_context_before_common_params() {
    let mut config = NativeRuntimeConfig::default();
    config.context.embeddings = Some(true);

    let error = apply_model_class_defaults(&mut config, ModelClass::EncoderDecoder)
        .expect_err("encoder-decoder embeddings");

    assert!(
        matches!(error, Error::UnsupportedOperation { operation: "load", reason }
            if reason.contains("embedding output"))
    );
}

#[test]
fn runtime_parts_new_allocates_minimum_scheduler_and_batch_state() {
    let config = NativeRuntimeConfig::default();
    let parts =
        RuntimeParts::new(&config, ResolvedRuntimeLimits::default()).expect("runtime parts");

    assert_eq!(parts.max_sequences, 1);
    assert_eq!(parts.slot_scheduler.slots.len(), 1);
    assert!(parts.scratch_token_capacity >= 1);
}

#[test]
fn empty_native_runtime_reports_missing_model_class() {
    let runtime = NativeRuntimeHandle::empty_for_tests();

    let error = model_class_from_init(&runtime).expect_err("model class");

    assert!(matches!(
        error,
        Error::UnsupportedOperation { operation: "load", reason }
            if reason.contains("neither encoder nor decoder")
    ));
}

#[test]
fn init_multimodal_context_without_projector_is_noop_and_projector_needs_native_context() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();

    init_multimodal_context(&config, &mut runtime).expect("no projector");

    let mut config = NativeRuntimeConfig::default();
    config.multimodal.projector_path = Some("projector.gguf".into());
    let error = init_multimodal_context(&config, &mut runtime).expect_err("projector init");

    assert!(matches!(
        error,
        Error::NullPointer("sipp_mtmd_init_from_file")
    ));
}

#[test]
fn encoder_decoder_rejects_parallel_contexts_before_common_params() {
    let mut config = NativeRuntimeConfig::default();
    config.context.n_parallel = Some(2);

    let error = apply_model_class_defaults(&mut config, ModelClass::EncoderDecoder)
        .expect_err("encoder-decoder parallelism");

    assert!(
        matches!(error, Error::UnsupportedOperation { operation: "load", reason }
            if reason.contains("n_parallel=1"))
    );
}
