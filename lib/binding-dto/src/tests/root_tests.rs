//! Tests endpoint, request, and descriptor conversion to core types.
//!
//! Covers strict variant fields and typed request conversion with pure values
//! and no native runtime execution.

use super::*;
use serde_json::json;
use sipp::DEFAULT_STORAGE_ROOT;

#[test]
fn endpoint_ref_maps_closed_builtin_kinds() {
    let local_dto = EndpointRef {
        kind: "local".to_string(),
        id: "local-a".to_string(),
    };
    let local = CoreEndpointRef::try_from(&local_dto).expect("local endpoint");
    assert!(matches!(local, CoreEndpointRef::Local { id } if id == "local-a"));

    let gateway_dto = EndpointRef {
        kind: "gateway".to_string(),
        id: "gateway-a".to_string(),
    };
    let gateway = CoreEndpointRef::try_from(&gateway_dto).expect("gateway endpoint");
    assert!(matches!(gateway, CoreEndpointRef::Gateway { id } if id == "gateway-a"));

    let provider_dto = EndpointRef {
        kind: "provider".to_string(),
        id: "provider-a".to_string(),
    };
    let provider = CoreEndpointRef::try_from(&provider_dto).expect("provider endpoint");
    assert!(matches!(provider, CoreEndpointRef::Provider { id } if id == "provider-a"));

    let invalid = EndpointRef {
        kind: "custom_http".to_string(),
        id: "bad".to_string(),
    };
    assert!(CoreEndpointRef::try_from(&invalid).is_err());
}

#[test]
fn query_request_maps_gateway_endpoint_options() {
    let request_dto = SippQueryRequest {
        request_id: Some("request-1".to_string()),
        endpoint: Some(EndpointRef {
            kind: "gateway".to_string(),
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
        endpoint_options: Some(json!({ "seed": 7 })),
        provider_options: None,
        emit_tokens: Some(true),
    };
    let request = CoreQueryRequest::try_from(&request_dto).expect("query request");

    assert!(matches!(
        request.endpoint,
        Some(CoreEndpointRef::Gateway { id }) if id == "custom"
    ));
    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options.max_tokens, Some(8));
    assert_eq!(request.endpoint_options.get("seed"), Some(&json!(7)));
    assert!(request.emit_tokens);
}

#[test]
fn gateway_descriptor_maps_through_add_shape() {
    let descriptor_dto = EndpointDescriptor {
        kind: "gateway".to_string(),
        base_url: Some("https://gateway.example.test".to_string()),
        target: Some("developer-model".to_string()),
        authentication: Some(GatewayAuthentication {
            kind: "bearer".to_string(),
            value: Some("secret".to_string()),
            header_name: None,
        }),
        timeout_ms: Some(5_000),
        query_route: Some("/generate".to_string()),
        chat_route: Some("/conversation".to_string()),
        embed_route: Some("/vectorize".to_string()),
        protocol_options: Some(json!({ "profile": "custom" })),
        ..EndpointDescriptor::default()
    };
    let descriptor = CoreEndpointDescriptor::try_from(&descriptor_dto).expect("gateway descriptor");

    match descriptor {
        CoreEndpointDescriptor::Gateway(descriptor) => {
            assert_eq!(descriptor.target, "developer-model");
            assert_eq!(descriptor.routes.query, "/generate");
            assert_eq!(
                descriptor.protocol_options.get("profile"),
                Some(&json!("custom"))
            );
            assert!(matches!(
                descriptor.authentication,
                CoreGatewayAuthentication::Bearer(_)
            ));
        }
        _ => panic!("expected gateway descriptor"),
    }
}

#[test]
fn local_descriptor_maps_explicit_remote_source_and_storage_root() {
    let descriptor = CoreEndpointDescriptor::try_from(&EndpointDescriptor {
        kind: "local".to_string(),
        source: Some(ModelSource {
            kind: "remote".to_string(),
            model_urls: Some(vec!["https://example.test/model.gguf".to_string()]),
            projector_url: Some("https://example.test/mmproj.gguf".to_string()),
            ..ModelSource::default()
        }),
        storage_root: Some(".sipp-models".to_string()),
        ..EndpointDescriptor::default()
    })
    .expect("local descriptor");

    let CoreEndpointDescriptor::Local(local) = descriptor else {
        panic!("expected local descriptor");
    };
    assert_eq!(local.storage_root, PathBuf::from(".sipp-models"));
}

#[test]
fn local_descriptor_defaults_storage_root() {
    let descriptor = CoreEndpointDescriptor::try_from(&EndpointDescriptor {
        kind: "local".to_string(),
        source: Some(ModelSource {
            kind: "remote".to_string(),
            model_urls: Some(vec!["https://example.test/model.gguf".to_string()]),
            ..ModelSource::default()
        }),
        ..EndpointDescriptor::default()
    })
    .expect("local descriptor");

    let CoreEndpointDescriptor::Local(local) = descriptor else {
        panic!("expected local descriptor");
    };
    assert_eq!(local.storage_root, PathBuf::from(DEFAULT_STORAGE_ROOT));
}

#[test]
fn provider_descriptor_rejects_unsupported_provider_name() {
    let error = CoreEndpointDescriptor::try_from(&EndpointDescriptor {
        kind: "provider".to_string(),
        provider: Some("openai-compatible".to_string()),
        model: Some("model-a".to_string()),
        ..EndpointDescriptor::default()
    })
    .expect_err("unsupported provider name");

    assert!(error
        .to_string()
        .contains("provider must be one of: openai, anthropic, openai_compatible"));
}

#[test]
fn local_endpoint_rejects_unknown_source_kind() {
    let error = CoreEndpointDescriptor::try_from(&EndpointDescriptor {
        kind: "local".to_string(),
        source: Some(ModelSource {
            kind: "installed".to_string(),
            ..ModelSource::default()
        }),
        ..EndpointDescriptor::default()
    })
    .expect_err("unknown source kind");
    assert!(error
        .to_string()
        .contains("model source kind must be local or remote"));
}

#[test]
fn endpoint_options_must_be_json_objects() {
    let request = SippEmbedRequest {
        input: "hello".to_string(),
        endpoint_options: Some(json!(["bad"])),
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
    let missing_prompt = json!({ "endpoint": { "kind": "local", "id": "local" } });
    assert!(serde_json::from_value::<SippQueryRequest>(missing_prompt).is_err());

    let missing_endpoint_id = json!({ "kind": "local" });
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
