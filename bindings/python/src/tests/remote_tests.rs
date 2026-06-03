//! Remote binding tests for the parent module.

use super::*;
use cogentlm_client::RemoteErrorKind as CoreRemoteErrorKind;

#[test]
fn py_remote_gateway_config_maps_core_fields() {
    let config = PyRemoteGatewayConfig::new(
        "pro-chat".to_string(),
        "http://localhost:8787".to_string(),
        "token".to_string(),
        Some(1500),
    );

    let core = config.to_core();

    assert_eq!(core.alias, "pro-chat");
    assert_eq!(core.base_url, "http://localhost:8787");
    assert_eq!(format!("{:?}", core.token), "RemoteSecret([redacted])");
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
}

#[test]
fn py_add_remote_rejects_gateway_base_url_userinfo() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let config = Py::new(
            py,
            PyRemoteGatewayConfig::new(
                "pro-chat".to_string(),
                "https://user:gateway-secret@gateway.example.test".to_string(),
                "gateway-token".to_string(),
                None,
            ),
        )
        .expect("config");
        let error = match client.add_remote(py, "pro".to_string(), config) {
            Ok(_) => panic!("gateway URL userinfo must fail"),
            Err(error) => error,
        };
        let message = error.value_bound(py).to_string();

        assert_eq!(
            message,
            "remote gateway error (invalid_request): gateway base_url must not include userinfo"
        );
        assert!(!message.contains("gateway-secret"));
        assert!(!message.contains("gateway-token"));
    });
}

#[test]
fn py_gateway_options_must_be_json_object() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let list = PyList::empty_bound(py).into_py(py);
        assert!(py_gateway_options_or_empty(py, Some(list)).is_err());

        let dict = PyDict::new_bound(py);
        dict.set_item("seed", 7).expect("set seed");
        let options = py_gateway_options_or_empty(py, Some(dict.into_py(py))).expect("options");

        assert_eq!(options.get("seed"), Some(&serde_json::json!(7)));
    });
}

#[test]
fn py_remote_request_rejects_local_only_gateway_options() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let config = Py::new(
            py,
            PyRemoteGatewayConfig::new(
                "pro-chat".to_string(),
                "https://gateway.example.test".to_string(),
                "gateway-token".to_string(),
                None,
            ),
        )
        .expect("config");
        let endpoint = client
            .add_remote(py, "pro".to_string(), config)
            .expect("add remote");
        let endpoint = Py::new(py, endpoint).expect("endpoint");
        let gateway_options = PyDict::new_bound(py);
        gateway_options
            .set_item("grammar", "root ::= \"ok\"")
            .expect("set grammar");

        let run = client
            .query(
                py,
                "hello".to_string(),
                Some(endpoint),
                None,
                None,
                Some(gateway_options.into_py(py)),
                false,
            )
            .expect("query run");
        let error = run
            .result(py)
            .expect_err("local-only gateway option must fail");
        let message = error.value_bound(py).to_string();

        assert_eq!(
            message,
            "gateway_options cannot contain local-only field: grammar"
        );
    });
}

#[test]
fn py_remote_error_has_structured_fields() {
    pyo3::prepare_freethreaded_python();
    let mut error = CoreRemoteError::new(CoreRemoteErrorKind::RateLimited, "slow down");
    error.status = Some(429);
    error.code = Some("rate_limited".to_string());
    error.request_id = Some("req-1".to_string());
    error.retry_after = Some(Duration::from_millis(2500));

    let py_error = to_py_remote_error(error);
    Python::with_gil(|py| {
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
                .getattr("retry_after_ms")
                .expect("retry")
                .extract::<f64>()
                .expect("retry number"),
            2500.0
        );
    });
}
