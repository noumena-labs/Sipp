//! Tests the `model` module in `cogentlm-providers`.
//!
//! Covers provider capability defaults and model value construction with
//! deterministic model-free fixtures.

use cogentlm_core::CapabilitySupport;

use super::*;

#[test]
fn unknown_capabilities_mark_every_surface_unknown() {
    let capabilities = ProviderCapabilities::unknown();

    assert_eq!(capabilities.chat, CapabilitySupport::Unknown);
    assert_eq!(capabilities.generate, CapabilitySupport::Unknown);
    assert_eq!(capabilities.embeddings, CapabilitySupport::Unknown);
    assert_eq!(capabilities.token_emission, CapabilitySupport::Unknown);
}

#[test]
fn provider_model_preserves_public_fields_and_raw_payload() {
    let model = ProviderModel {
        id: "model-a".to_string(),
        provider: ProviderKind::Proxy,
        display_name: Some("Model A".to_string()),
        capabilities: ProviderCapabilities::unknown(),
        context_window: Some(8192),
        max_output_tokens: Some(1024),
        raw: serde_json::json!({ "id": "model-a", "owned_by": "test" }),
    };

    assert_eq!(model.id, "model-a");
    assert_eq!(model.provider, ProviderKind::Proxy);
    assert_eq!(model.display_name.as_deref(), Some("Model A"));
    assert_eq!(model.context_window, Some(8192));
    assert_eq!(model.max_output_tokens, Some(1024));
    assert_eq!(model.raw["owned_by"], "test");
}
