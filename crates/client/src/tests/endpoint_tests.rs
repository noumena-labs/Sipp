//! Tests the `endpoint` module in `cogentlm-client`.
//!
//! Covers endpoint reference classification and capability mapping with
//! deterministic model-capability fixtures instead of loaded native models.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::{EmbeddingCapabilities, ModelCapabilities, ModelClass, PoolingType};

use super::*;

fn model_capabilities(text: bool, chat_template: bool, embeddings: bool) -> ModelCapabilities {
    ModelCapabilities {
        model_class: ModelClass::DecoderOnly,
        supports_text_generation: text,
        supports_embeddings: embeddings,
        has_chat_template: chat_template,
        embedding: embeddings.then_some(EmbeddingCapabilities {
            dimensions: 3,
            pooling: PoolingType::Mean,
        }),
    }
}

#[test]
fn endpoint_ref_classifies_local_only() {
    let local = EndpointRef::Local {
        id: "local".to_string(),
    };
    let remote = EndpointRef::Remote {
        id: "remote".to_string(),
    };

    assert!(local.is_local());
    assert!(!remote.is_local());
    assert_eq!(local.clone(), local);
    assert_ne!(local, remote);
    assert!(format!("{remote:?}").contains("remote"));

    let mut hasher = DefaultHasher::new();
    local.hash(&mut hasher);
    let first_hash = hasher.finish();
    let mut hasher = DefaultHasher::new();
    local.clone().hash(&mut hasher);
    assert_eq!(hasher.finish(), first_hash);
}

#[test]
fn local_capabilities_map_model_support_flags() {
    let text_chat_embed = EndpointCapabilities::from_local(&model_capabilities(true, true, true));
    assert_eq!(text_chat_embed.query, CapabilitySupport::Supported);
    assert_eq!(text_chat_embed.chat, CapabilitySupport::Supported);
    assert_eq!(text_chat_embed.embed, CapabilitySupport::Supported);

    let raw_text_only = EndpointCapabilities::from_local(&model_capabilities(true, false, false));
    assert_eq!(raw_text_only.query, CapabilitySupport::Supported);
    assert_eq!(raw_text_only.chat, CapabilitySupport::Unsupported);
    assert_eq!(raw_text_only.embed, CapabilitySupport::Unsupported);

    let embed_only = EndpointCapabilities::from_local(&model_capabilities(false, true, true));
    assert_eq!(embed_only.query, CapabilitySupport::Unsupported);
    assert_eq!(embed_only.chat, CapabilitySupport::Unsupported);
    assert_eq!(embed_only.embed, CapabilitySupport::Supported);
}

#[test]
fn operation_lookup_maps_known_verbs_and_rejects_unknown() {
    let capabilities = EndpointCapabilities {
        query: CapabilitySupport::Supported,
        chat: CapabilitySupport::Unknown,
        embed: CapabilitySupport::Unsupported,
    };

    assert_eq!(capabilities.clone(), capabilities);
    assert!(format!("{capabilities:?}").contains("query"));
    assert_eq!(
        capabilities.for_operation("query"),
        CapabilitySupport::Supported
    );
    assert_eq!(
        capabilities.for_operation("chat"),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        capabilities.for_operation("embed"),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        capabilities.for_operation("rerank"),
        CapabilitySupport::Unsupported
    );
}

#[cfg(feature = "remote")]
#[test]
fn unknown_capabilities_are_unknown_for_remote_endpoints() {
    let capabilities = EndpointCapabilities::unknown();

    assert_eq!(capabilities.query, CapabilitySupport::Unknown);
    assert_eq!(capabilities.chat, CapabilitySupport::Unknown);
    assert_eq!(capabilities.embed, CapabilitySupport::Unknown);
}
