use super::*;
use crate::engine::protocol::ModelClass;
use crate::error::Error;
use crate::runtime::config::NativeRuntimeConfig;

#[test]
fn encoder_only_enables_embedding_context_before_common_params() {
    let mut config = NativeRuntimeConfig::default();

    apply_model_class_defaults(&mut config, ModelClass::EncoderOnly).expect("defaults");

    assert_eq!(config.context.embeddings, Some(true));
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
