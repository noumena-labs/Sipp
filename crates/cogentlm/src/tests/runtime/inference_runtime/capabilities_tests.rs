//! Tests the `runtime::inference_runtime::capabilities` module in `cogentlm`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::engine::protocol::{EmbeddingCapabilities, ModelClass, PoolingType};

#[test]
fn public_capabilities_hide_decoder_start_token() {
    let capabilities = RuntimeModelCapabilities {
        class: ModelClass::EncoderDecoder,
        embedding_dimensions: 768,
        pooling_type: PoolingType::Mean,
        decoder_start_token: Some(0),
        has_chat_template: false,
        embedding_context: false,
    }
    .to_public();

    assert_eq!(capabilities.model_class, ModelClass::EncoderDecoder);
    assert!(capabilities.supports_text_generation);
    assert!(!capabilities.supports_embeddings);
    assert!(capabilities.embedding.is_none());
}

#[test]
fn public_capabilities_include_embedding_metadata_only_when_supported() {
    let capabilities = RuntimeModelCapabilities {
        class: ModelClass::EncoderOnly,
        embedding_dimensions: 1024,
        pooling_type: PoolingType::Cls,
        decoder_start_token: None,
        has_chat_template: false,
        embedding_context: true,
    }
    .to_public();

    assert!(!capabilities.supports_text_generation);
    assert!(capabilities.supports_embeddings);
    assert_eq!(
        capabilities.embedding,
        Some(EmbeddingCapabilities {
            dimensions: 1024,
            pooling: PoolingType::Cls,
        })
    );
}

#[test]
fn public_capabilities_hide_unpooled_embedding_context() {
    let capabilities = RuntimeModelCapabilities {
        class: ModelClass::EncoderOnly,
        embedding_dimensions: 1024,
        pooling_type: PoolingType::None,
        decoder_start_token: None,
        has_chat_template: false,
        embedding_context: true,
    }
    .to_public();

    assert!(!capabilities.supports_embeddings);
    assert!(capabilities.embedding.is_none());
}
