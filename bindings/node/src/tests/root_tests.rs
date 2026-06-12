//! Unit tests for the Node endpoint and request mapping boundary.

use super::*;
use serde_json::json;

#[test]
fn endpoint_ref_maps_closed_builtin_kinds() {
    let local = EndpointRef {
        kind: "local".to_string(),
        id: "local-a".to_string(),
    }
    .to_core()
    .expect("local endpoint");
    assert!(matches!(
        local,
        CoreEndpointRef::Local { id } if id == "local-a"
    ));

    let gateway = EndpointRef {
        kind: "gateway".to_string(),
        id: "gateway-a".to_string(),
    }
    .to_core()
    .expect("gateway endpoint");
    assert!(matches!(
        gateway,
        CoreEndpointRef::Gateway { id } if id == "gateway-a"
    ));

    let provider = EndpointRef {
        kind: "provider".to_string(),
        id: "provider-a".to_string(),
    }
    .to_core()
    .expect("provider endpoint");
    assert!(matches!(
        provider,
        CoreEndpointRef::Provider { id } if id == "provider-a"
    ));

    let invalid = EndpointRef {
        kind: "custom_http".to_string(),
        id: "bad".to_string(),
    };
    assert!(invalid.to_core().is_err());
}

#[test]
fn query_request_maps_gateway_endpoint_options() {
    let request = SippQueryRequest {
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
    }
    .to_core()
    .expect("query request");

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
    let descriptor = EndpointDescriptor {
        kind: "gateway".to_string(),
        model_path: None,
        config: None,
        base_url: Some("https://gateway.example.test".to_string()),
        target: Some("developer-model".to_string()),
        authentication: Some(GatewayAuthentication {
            kind: "bearer".to_string(),
            value: Some("secret".to_string()),
            header_name: None,
        }),
        provider: None,
        model: None,
        api_key: None,
        timeout_ms: Some(5_000),
        version: None,
        auth_header_name: None,
        auth_header_value: None,
        static_headers: None,
        correlation_header: None,
        query_route: Some("/generate".to_string()),
        chat_route: Some("/conversation".to_string()),
        embed_route: Some("/vectorize".to_string()),
        protocol_options: Some(json!({ "profile": "custom" })),
    }
    .to_core()
    .expect("gateway descriptor");

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
        request_id: None,
        endpoint: None,
        input: "hello".to_string(),
        local: None,
        endpoint_options: Some(json!(["bad"])),
        provider_options: None,
    };

    assert!(request.to_core().is_err());
}
