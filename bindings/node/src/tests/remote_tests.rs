//! Remote binding tests for the parent module.

use super::*;
use serde_json::json;

fn gateway_descriptor(alias: &str, base_url: &str, token: &str) -> EndpointDescriptor {
    EndpointDescriptor {
        kind: "gateway".to_string(),
        model_path: None,
        config: None,
        alias: Some(alias.to_string()),
        base_url: Some(base_url.to_string()),
        token: Some(token.to_string()),
        provider: None,
        model: None,
        api_key: None,
        timeout_ms: None,
        version: None,
        auth_header_name: None,
        auth_header_value: None,
        static_headers: None,
    }
}

fn add_gateway(
    client: &CogentClient,
    id: &str,
    alias: &str,
    base_url: &str,
    token: &str,
) -> Result<EndpointRef> {
    let descriptor = gateway_descriptor(alias, base_url, token).to_core()?;
    let endpoint = client
        .inner
        .lock()
        .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))
        .and_then(|mut client| {
            block_on(client.add(id, descriptor)).map_err(client_error_without_env)
        })?;
    Ok(endpoint_ref_to_node(endpoint))
}

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
fn add_rejects_empty_alias_before_gateway_transport_config() {
    let client = CogentClient::new().expect("client");
    let error = match add_gateway(
        &client,
        "pro",
        "   ",
        "https://user:gateway-secret@gateway.example.test",
        "gateway-token",
    ) {
        Ok(_) => panic!("blank alias must reject before transport config"),
        Err(error) => error,
    };
    let message = error.reason.as_str();

    assert_eq!(error.status, Status::InvalidArg);
    assert_eq!(message, "remote alias must not be empty");
    assert!(!message.contains("gateway-secret"));
    assert!(!message.contains("gateway-token"));
}

#[test]
fn add_rejects_id_whitespace_before_gateway_transport_config() {
    let client = CogentClient::new().expect("client");
    let error = match add_gateway(
        &client,
        " pro ",
        "pro-chat",
        "https://user:gateway-secret@gateway.example.test",
        "gateway-token",
    ) {
        Ok(_) => panic!("whitespace id must reject before transport config"),
        Err(error) => error,
    };
    let message = error.reason.as_str();

    assert_eq!(error.status, Status::InvalidArg);
    assert_eq!(message, "remote id must not contain surrounding whitespace");
    assert!(!message.contains("gateway-secret"));
    assert!(!message.contains("gateway-token"));
}

#[test]
fn add_rejects_alias_surrounding_whitespace() {
    let client = CogentClient::new().expect("client");
    let error = match add_gateway(
        &client,
        "pro",
        " pro-chat ",
        "https://gateway.example.test",
        "gateway-token",
    ) {
        Ok(_) => panic!("whitespace alias must reject"),
        Err(error) => error,
    };

    assert_eq!(error.status, Status::InvalidArg);
    assert_eq!(
        error.reason,
        "remote alias must not contain surrounding whitespace"
    );
}

#[test]
fn add_rejects_gateway_base_url_userinfo() {
    let client = CogentClient::new().expect("client");
    let error = match add_gateway(
        &client,
        "pro",
        "pro-chat",
        "https://user:gateway-secret@gateway.example.test",
        "gateway-token",
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
    let endpoint = add_gateway(
        &client,
        "pro",
        "pro-chat",
        "https://gateway.example.test",
        "gateway-token",
    )
    .expect("add remote");

    let run = client
        .query(CogentQueryRequest {
            endpoint: Some(endpoint),
            prompt: "hello".to_string(),
            options: None,
            local: None,
            gateway_options: Some(json!({ "grammar": "root ::= \"ok\"" })),
            provider_options: None,
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
