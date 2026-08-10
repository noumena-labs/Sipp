//! Tests endpoint and request conversion to core types.
//!
//! Covers structurally distinct endpoint variants and typed request conversion
//! with pure values and no native runtime execution.

use super::*;
use serde_json::json;

#[test]
fn endpoint_ref_maps_id() {
    let endpoint = CoreEndpointRef::try_from(&EndpointRef {
        id: "endpoint-a".to_string(),
    })
    .expect("endpoint");

    assert_eq!(endpoint.id(), "endpoint-a");
}

#[test]
fn query_request_maps_extra() {
    let request_dto = SippQueryRequest {
        request_id: Some("request-1".to_string()),
        endpoint: Some(EndpointRef {
            id: "custom".to_string(),
        }),
        prompt: "hello".to_string(),
        options: Some(SippTextOptions {
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: None,
            stop: None,
        }),
        local: None,
        extra: Some(json!({ "seed": 7 })),
        emit_tokens: Some(true),
    };
    let request = CoreQueryRequest::try_from(&request_dto).expect("query request");

    assert_eq!(
        request.endpoint.as_ref().map(CoreEndpointRef::id),
        Some("custom")
    );
    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options.max_tokens, Some(8));
    assert_eq!(request.extra.get("seed"), Some(&json!(7)));
    assert!(request.emit_tokens);
}

#[test]
fn gateway_input_maps_through_add_shape() {
    let input = GatewayEndpointInput {
        base_url: "https://gateway.example.test".to_string(),
        target: "developer-model".to_string(),
        authentication: GatewayAuthentication::Bearer("secret".to_string()),
        static_headers: Vec::new(),
        timeout_ms: Some(5_000),
        query_route: Some("/generate".to_string()),
        chat_route: Some("/conversation".to_string()),
        embed_route: Some("/vectorize".to_string()),
        protocol_options: Some(json!({ "profile": "custom" })),
    };

    let _: CoreEndpoint = input.try_into().expect("gateway endpoint");
}

#[test]
fn local_input_maps_managed_model() {
    let input = LocalEndpointInput {
        model: managed_model("model-a"),
        runtime: NativeRuntimeConfig::default(),
    };

    let _: CoreEndpoint = input.try_into().expect("local endpoint");
}

#[test]
fn gateway_authentication_is_typed() {
    let authentication = CoreGatewayAuthentication::from(GatewayAuthentication::Header {
        name: "x-api-key".to_string(),
        value: "secret".to_string(),
    });

    assert!(matches!(
        authentication,
        CoreGatewayAuthentication::Header { .. }
    ));
}

#[test]
fn provider_input_maps_selected_variant() {
    let input = ProviderEndpointInput::OpenAi(OpenAiProviderInput {
        model: "model-a".to_string(),
        api_key: "secret".to_string(),
        base_url: None,
        timeout_ms: None,
    });

    let _: CoreEndpoint = input.try_into().expect("provider endpoint");
}

#[test]
fn provider_selection_rejects_fields_from_another_adapter() {
    let input = ProviderSelectionInput {
        provider: "openai".to_string(),
        model: "model-a".to_string(),
        api_key: Some("secret".to_string()),
        base_url: None,
        timeout_ms: None,
        version: Some("2023-06-01".to_string()),
        auth_header_name: None,
        auth_header_value: None,
        static_headers: None,
        correlation_header: None,
    };

    let error = ProviderEndpointInput::try_from(input).expect_err("invalid provider fields");
    assert_eq!(
        error.to_string(),
        "version is not valid for the OpenAI provider"
    );
}

fn managed_model(id: &str) -> CoreManagedModel {
    CoreManagedModel {
        id: id.to_string(),
        name: id.to_string(),
        bytes: 1,
        modality: sipp::lifecycle::ModelModality::Text,
        status: sipp::lifecycle::ModelStatus::Ready,
    }
}

#[test]
fn extra_must_be_a_json_object() {
    let request = SippEmbedRequest {
        input: "hello".to_string(),
        extra: Some(json!(["bad"])),
        ..SippEmbedRequest::default()
    };

    assert!(CoreEmbedRequest::try_from(&request).is_err());
}

#[test]
fn chat_message_requires_valid_roles() {
    let invalid = SippChatRequest {
        messages: vec![ChatMessage {
            role: "tool".to_string(),
            content: "bad".to_string(),
        }],
        ..SippChatRequest::default()
    };
    assert!(CoreChatRequest::try_from(&invalid).is_err());

    let request_dto = SippChatRequest {
        messages: vec![ChatMessage {
            role: "assistant".to_string(),
            content: "ok".to_string(),
        }],
        ..SippChatRequest::default()
    };
    let request = CoreChatRequest::try_from(&request_dto).expect("chat request");
    assert_eq!(request.messages[0].role, CoreChatRole::Assistant);
    assert_eq!(request.messages[0].content, "ok");
}

#[test]
fn required_fields_do_not_deserialize_from_defaults() {
    let missing_prompt = json!({ "endpoint": { "id": "local" } });
    assert!(serde_json::from_value::<SippQueryRequest>(missing_prompt).is_err());

    let missing_endpoint_id = json!({});
    assert!(serde_json::from_value::<EndpointRef>(missing_endpoint_id).is_err());

    let missing_gpu_count = json!({ "placement": { "gpu_layers": {} } });
    assert!(serde_json::from_value::<NativeRuntimeConfig>(missing_gpu_count).is_err());
}

#[test]
fn local_media_is_never_json_encoded() {
    let request = SippQueryRequest {
        prompt: "describe image".to_string(),
        local: Some(LocalTextOptions {
            media: vec![vec![1, 2, 3, 4]],
            ..LocalTextOptions::default()
        }),
        ..SippQueryRequest::default()
    };

    let value = serde_json::to_value(request).expect("request json");
    assert_eq!(value.pointer("/local/media"), None);
}

#[test]
fn finite_f32_fields_reject_non_finite_values() {
    let text = SippTextOptions {
        temperature: Some(f64::INFINITY),
        ..SippTextOptions::default()
    };
    assert!(CoreTextOptions::try_from(&text).is_err());

    let sampling = SamplingRuntimeConfig {
        logit_bias: Some(vec![LogitBiasConfig {
            token: 1,
            bias: f64::NAN,
        }]),
        ..SamplingRuntimeConfig::default()
    };
    assert!(CoreSamplingRuntimeConfig::try_from(&sampling).is_err());

    let placement = ModelPlacementConfig {
        tensor_split: Some(vec![f64::NEG_INFINITY]),
        ..ModelPlacementConfig::default()
    };
    assert!(CoreModelPlacementConfig::try_from(&placement).is_err());

    let context = ContextRuntimeConfig {
        rope_freq_base: Some(f64::INFINITY),
        ..ContextRuntimeConfig::default()
    };
    assert!(sipp::engine::ContextRuntimeConfig::try_from(&context).is_err());
}

#[test]
fn partial_sampling_runtime_config_preserves_core_defaults() {
    let sampling = SamplingRuntimeConfig {
        repeat_penalty: Some(1.2),
        backend_sampling: Some(false),
        ..SamplingRuntimeConfig::default()
    };

    let config = CoreSamplingRuntimeConfig::try_from(&sampling).expect("sampling config");

    assert_eq!(config.repeat_penalty, Some(1.2));
    assert_eq!(config.top_k, Some(40));
    assert_eq!(
        config.samplers,
        vec![
            SamplerStage::TopK,
            SamplerStage::Penalties,
            SamplerStage::TopP,
            SamplerStage::Temperature
        ]
    );
    assert!(!config.backend_sampling);
}

#[test]
fn local_text_sampling_converts_to_sparse_runtime_override() {
    let local = LocalTextOptions {
        sampling: Some(SamplingRuntimeConfig {
            repeat_penalty: Some(1.2),
            ..SamplingRuntimeConfig::default()
        }),
        ..LocalTextOptions::default()
    };

    let local = CoreLocalTextOptions::try_from(local).expect("local text options");
    let sampling = local.sampling.expect("sampling override");

    assert_eq!(sampling.repeat_penalty, Some(1.2));
    assert_eq!(sampling.top_k, None);
    assert_eq!(sampling.backend_sampling, None);
}

#[test]
fn camel_case_request_fields_deserialize() {
    let request: SippQueryRequest = serde_json::from_value(json!({
        "requestId": "r-1",
        "prompt": "hi",
        "options": { "maxTokens": 4, "topP": 0.9 },
        "emitTokens": true,
    }))
    .expect("camelCase request");

    assert_eq!(request.request_id.as_deref(), Some("r-1"));
    assert_eq!(request.emit_tokens, Some(true));
    let options = request.options.expect("options");
    assert_eq!(options.max_tokens, Some(4));
    assert_eq!(options.top_p, Some(0.9));
}

#[test]
fn snake_case_request_fields_deserialize() {
    let request: SippQueryRequest = serde_json::from_value(json!({
        "request_id": "r-2",
        "prompt": "hi",
        "options": { "max_tokens": 5, "top_p": 0.8 },
        "emit_tokens": false,
    }))
    .expect("snake_case request");

    assert_eq!(request.request_id.as_deref(), Some("r-2"));
    let options = request.options.expect("options");
    assert_eq!(options.max_tokens, Some(5));
    assert_eq!(options.top_p, Some(0.8));
}
