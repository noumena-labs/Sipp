//! Remote binding tests for the parent module.

use super::*;
use serde_json::json;

#[test]
fn remote_gateway_config_maps_core_fields() {
    let config = RemoteGatewayConfig {
        alias: "pro-chat".to_string(),
        base_url: "http://localhost:8787".to_string(),
        token: "token".to_string(),
        timeout_ms: Some(1500),
    };

    let core = config.to_core();

    assert_eq!(core.alias, "pro-chat");
    assert_eq!(core.base_url, "http://localhost:8787");
    assert_eq!(format!("{:?}", core.token), "RemoteSecret([redacted])");
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
}

#[test]
fn remote_gateway_config_from_value_rejects_direct_provider_fields() {
    for field in [
        "apiKey",
        "providerApiKey",
        "providerBaseUrl",
        "headers",
        "authorization",
    ] {
        let error = match remote_gateway_config_from_value(json!({
            "alias": "pro-chat",
            "baseUrl": "https://gateway.example.test",
            "token": "provider-secret",
            field: "provider-secret"
        })) {
            Ok(_) => panic!("direct-provider field must fail"),
            Err(error) => error,
        };
        let message = error.reason.as_str();

        assert_eq!(
            message,
            format!("unsupported remote gateway config field: {field}")
        );
        assert!(!message.contains("provider-secret"));
    }
}

#[test]
fn remote_gateway_config_from_value_requires_gateway_shape() {
    assert!(remote_gateway_config_from_value(json!("not-object")).is_err());
    assert!(remote_gateway_config_from_value(json!({
        "alias": "pro-chat",
        "baseUrl": "https://gateway.example.test",
        "token": "token",
        "timeoutMs": 0
    }))
    .is_err());

    let config = remote_gateway_config_from_value(json!({
        "alias": "pro-chat",
        "baseUrl": "https://gateway.example.test",
        "token": "token",
        "timeoutMs": 1500
    }))
    .expect("gateway config");

    assert_eq!(config.alias, "pro-chat");
    assert_eq!(config.base_url, "https://gateway.example.test");
    assert_eq!(config.token, "token");
    assert_eq!(config.timeout_ms, Some(1500));
}

#[test]
fn add_remote_rejects_gateway_base_url_userinfo() {
    let client = CogentClient::new().expect("client");
    let error = match client.add_remote(
        "pro".to_string(),
        json!({
            "alias": "pro-chat",
            "baseUrl": "https://user:gateway-secret@gateway.example.test",
            "token": "gateway-token"
        }),
    ) {
        Ok(_) => panic!("gateway URL userinfo must fail"),
        Err(error) => error,
    };
    let message = error.reason.as_str();

    assert_eq!(error.status, Status::InvalidArg);
    assert_eq!(
        message,
        "remote gateway error (invalid_request): gateway base_url must not include userinfo"
    );
    assert!(!message.contains("gateway-secret"));
    assert!(!message.contains("gateway-token"));
}

#[test]
fn gateway_options_must_be_json_object() {
    assert!(gateway_options_or_empty(Some(json!("not-object"))).is_err());

    let options = gateway_options_or_empty(Some(json!({ "seed": 7 }))).expect("options");

    assert_eq!(options.get("seed"), Some(&json!(7)));
}

#[test]
fn remote_request_rejects_local_only_gateway_options() {
    let client = CogentClient::new().expect("client");
    let endpoint = client
        .add_remote(
            "pro".to_string(),
            json!({
                "alias": "pro-chat",
                "baseUrl": "https://gateway.example.test",
                "token": "gateway-token"
            }),
        )
        .expect("add remote");

    let run = client
        .query(CogentQueryRequest {
            endpoint: Some(endpoint),
            prompt: "hello".to_string(),
            options: None,
            local: None,
            gateway_options: Some(json!({ "grammar": "root ::= \"ok\"" })),
            emit_tokens: None,
        })
        .expect("query run");
    let response = run
        .response
        .lock()
        .expect("response mutex")
        .take()
        .expect("response future");
    let error = block_on(response).expect_err("local-only gateway option must fail");

    assert!(matches!(
        error,
        CoreClientError::InvalidRequest(message)
            if message == "gateway_options cannot contain local-only field: grammar"
    ));
}

#[test]
fn remote_error_maps_to_node_status_and_message() {
    let error = CoreRemoteError::new(CoreRemoteErrorKind::InvalidRequest, "bad request");

    assert_eq!(
        remote_error_message(&error),
        "remote gateway error (invalid_request): bad request"
    );
    assert_eq!(
        remote_error_status(CoreRemoteErrorKind::InvalidRequest),
        Status::InvalidArg
    );
    assert_eq!(
        remote_error_status(CoreRemoteErrorKind::RateLimited),
        Status::GenericFailure
    );
}
