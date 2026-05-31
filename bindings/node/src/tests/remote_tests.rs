//! Remote binding tests for the parent module.

use super::*;
use cogentlm_client::RemoteKind as CoreRemoteKind;
use serde_json::json;

#[test]
fn remote_proxy_config_maps_static_headers() {
    let config = RemoteConfig {
        kind: "proxy".to_string(),
        model: "proxy-model".to_string(),
        api_key: None,
        base_url: Some("http://localhost".to_string()),
        version: None,
        auth: Some(RemoteAuthConfig {
            bearer: Some("token".to_string()),
            header: None,
        }),
        protocol: None,
        static_headers: Some(vec![RemoteStaticHeaderConfig {
            name: "x-cogent-test".to_string(),
            value: "yes".to_string(),
        }]),
        timeout_ms: Some(1500),
    };

    let core = config.to_core().expect("remote proxy config");
    let CoreRemoteConfig::Proxy(core) = core else {
        panic!("expected proxy config");
    };

    assert_eq!(
        core.static_headers,
        vec![("x-cogent-test".to_string(), "yes".to_string())]
    );
    assert_eq!(core.protocol, CoreRemoteProtocol::OpenAiCompatible);
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
    match core.auth {
        CoreRemoteAuth::Bearer(secret) => {
            assert_eq!(format!("{secret:?}"), "RemoteSecret([redacted])")
        }
        CoreRemoteAuth::Header { .. } => panic!("expected bearer auth"),
    }
}

#[test]
fn remote_anthropic_config_maps_core_fields() {
    let config = RemoteConfig {
        kind: "anthropic".to_string(),
        model: "claude-test".to_string(),
        api_key: Some("token".to_string()),
        base_url: Some("http://localhost".to_string()),
        version: Some("2023-06-01".to_string()),
        auth: None,
        protocol: None,
        static_headers: None,
        timeout_ms: Some(1500),
    };

    let core = config.to_core().expect("remote anthropic config");
    let CoreRemoteConfig::Anthropic(core) = core else {
        panic!("expected anthropic config");
    };

    assert_eq!(core.model, "claude-test");
    assert_eq!(format!("{:?}", core.api_key), "RemoteSecret([redacted])");
    assert_eq!(core.base_url.as_deref(), Some("http://localhost"));
    assert_eq!(core.version.as_deref(), Some("2023-06-01"));
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
}

#[test]
fn remote_options_must_be_json_object() {
    assert!(remote_options_or_empty(Some(json!("not-object"))).is_err());

    let options = remote_options_or_empty(Some(json!({ "seed": 7 }))).expect("options");

    assert_eq!(options.get("seed"), Some(&json!(7)));
}

#[test]
fn remote_error_maps_to_node_status_and_message() {
    let error = CoreRemoteError::new(
        CoreRemoteErrorKind::InvalidRequest,
        CoreRemoteKind::Proxy,
        "bad request",
    );

    assert_eq!(
        remote_error_message(&error),
        "proxy remote error (invalid_request): bad request"
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
