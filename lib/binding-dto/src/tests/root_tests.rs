//! Tests endpoint, request, and descriptor conversion to core types.

use super::*;
use serde_json::json;

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
fn gateway_endpoint_descriptor_maps_through_add_shape() {
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
        CoreEndpointDescriptor::Gateway(config) => {
            assert_eq!(config.target, "developer-model");
            assert_eq!(config.routes.query, "/generate");
            assert_eq!(
                config.protocol_options.get("profile"),
                Some(&json!("custom"))
            );
            assert!(matches!(
                config.authentication,
                CoreGatewayAuthentication::Bearer(_)
            ));
        }
        _ => panic!("expected gateway descriptor"),
    }
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
