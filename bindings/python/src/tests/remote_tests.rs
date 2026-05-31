//! Remote binding tests for the parent module.

use super::*;
use cogentlm_client::{
    RemoteError as CoreRemoteError, RemoteErrorKind as CoreRemoteErrorKind,
    RemoteKind as CoreRemoteKind,
};
use std::time::Duration;

#[test]
fn py_remote_proxy_config_maps_static_headers() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let auth = Py::new(
            py,
            PyRemoteAuth {
                core: CoreRemoteAuth::Bearer(CoreRemoteSecret::new("token")),
            },
        )
        .expect("auth");
        let config = PyRemoteConfig::proxy(
            py,
            "proxy-model".to_string(),
            "http://localhost".to_string(),
            auth,
            "openai_compatible".to_string(),
            Some(vec![("x-cogent-test".to_string(), "yes".to_string())]),
            Some(1500),
        )
        .expect("proxy config");
        let core = config.to_core();
        let CoreRemoteConfig::Proxy(core) = core else {
            panic!("expected proxy config");
        };

        assert_eq!(
            core.static_headers,
            vec![("x-cogent-test".to_string(), "yes".to_string())]
        );
        assert_eq!(core.protocol, CoreRemoteProtocol::OpenAiCompatible);
        assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
    });
}

#[test]
fn py_remote_anthropic_config_maps_core_fields() {
    let config = PyRemoteConfig::anthropic(
        "claude-test".to_string(),
        "token".to_string(),
        Some("http://localhost".to_string()),
        Some("2023-06-01".to_string()),
        Some(1500),
    );
    let core = config.to_core();
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
fn py_remote_options_must_be_json_object() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let list = PyList::empty_bound(py).into_py(py);
        assert!(py_remote_options_or_empty(py, Some(list)).is_err());

        let dict = PyDict::new_bound(py);
        dict.set_item("seed", 7).expect("set seed");
        let options = py_remote_options_or_empty(py, Some(dict.into_py(py))).expect("options");

        assert_eq!(options.get("seed"), Some(&serde_json::json!(7)));
    });
}

#[test]
fn py_remote_error_has_structured_fields() {
    pyo3::prepare_freethreaded_python();
    let mut error = CoreRemoteError::new(
        CoreRemoteErrorKind::RateLimited,
        CoreRemoteKind::Proxy,
        "too many requests",
    );
    error.status = Some(429);
    error.code = Some("rate_limit".to_string());
    error.retry_after = Some(Duration::from_millis(1500));
    error.request_id = Some("req-123".to_string());
    error.raw = Some(Box::new(serde_json::json!({ "error": "limit" })));

    let py_error = to_py_remote_error(error);

    Python::with_gil(|py| {
        assert!(py_error.matches(py, py.get_type_bound::<RemoteError>()));
        let value = py_error.value_bound(py);

        assert_eq!(
            value
                .getattr("kind")
                .expect("kind")
                .extract::<String>()
                .expect("kind string"),
            "rate_limited"
        );
        assert_eq!(
            value
                .getattr("remote_kind")
                .expect("remote_kind")
                .extract::<String>()
                .expect("remote_kind string"),
            "proxy"
        );
        assert_eq!(
            value
                .getattr("status")
                .expect("status")
                .extract::<u16>()
                .expect("status value"),
            429
        );
        assert_eq!(
            value
                .getattr("code")
                .expect("code")
                .extract::<String>()
                .expect("code string"),
            "rate_limit"
        );

        let raw_body = value
            .getattr("raw_body")
            .expect("raw body")
            .downcast_into::<PyDict>()
            .expect("raw body dict");
        assert_eq!(
            raw_body
                .get_item("error")
                .expect("raw error")
                .expect("raw error item")
                .extract::<String>()
                .expect("raw error string"),
            "limit"
        );
    });
}
