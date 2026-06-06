//! Remote binding tests for the parent module.

use super::*;
use cogentlm_client::RemoteErrorKind as CoreRemoteErrorKind;

fn add_gateway(
    py: Python<'_>,
    client: &PyCogentClient,
    id: &str,
    alias: &str,
    base_url: &str,
    token: &str,
) -> PyResult<PyEndpointRef> {
    let descriptor = Py::new(
        py,
        PyGatewayDescriptor::new(
            alias.to_string(),
            base_url.to_string(),
            token.to_string(),
            None,
        )?,
    )?;
    client.add(py, id.to_string(), descriptor.into_py(py))
}

#[test]
fn py_remote_gateway_config_maps_core_fields() {
    let config = PyRemoteGatewayConfig::new(
        "pro-chat".to_string(),
        "http://localhost:8787".to_string(),
        "token".to_string(),
        Some(1500),
    )
    .expect("config");

    let core = config.to_core();

    assert_eq!(core.alias, "pro-chat");
    assert_eq!(core.base_url, "http://localhost:8787");
    assert_eq!(format!("{:?}", core.token), "RemoteSecret([redacted])");
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
}

#[test]
fn py_remote_gateway_config_rejects_zero_timeout() {
    pyo3::prepare_freethreaded_python();

    let error = match PyRemoteGatewayConfig::new(
        "pro-chat".to_string(),
        "https://gateway.example.test".to_string(),
        "gateway-token".to_string(),
        Some(0),
    ) {
        Ok(_) => panic!("zero timeout should fail"),
        Err(error) => error,
    };

    Python::with_gil(|py| {
        assert_eq!(
            error.value_bound(py).to_string(),
            "RemoteGatewayConfig.timeout_ms must be a positive integer"
        );
    });
}

#[test]
fn py_add_rejects_empty_alias_before_gateway_transport_config() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let error = match add_gateway(
            py,
            &client,
            "pro",
            "   ",
            "https://user:gateway-secret@gateway.example.test",
            "gateway-token",
        ) {
            Ok(_) => panic!("blank alias must reject before transport config"),
            Err(error) => error,
        };
        let message = error.value_bound(py).to_string();

        assert_eq!(message, "remote alias must not be empty");
        assert!(!message.contains("gateway-secret"));
        assert!(!message.contains("gateway-token"));
    });
}

#[test]
fn py_add_rejects_id_whitespace_before_gateway_transport_config() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let error = match add_gateway(
            py,
            &client,
            " pro ",
            "pro-chat",
            "https://user:gateway-secret@gateway.example.test",
            "gateway-token",
        ) {
            Ok(_) => panic!("whitespace id must reject before transport config"),
            Err(error) => error,
        };
        let message = error.value_bound(py).to_string();

        assert_eq!(message, "remote id must not contain surrounding whitespace");
        assert!(!message.contains("gateway-secret"));
        assert!(!message.contains("gateway-token"));
    });
}

#[test]
fn py_add_rejects_alias_surrounding_whitespace() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let error = match add_gateway(
            py,
            &client,
            "pro",
            " pro-chat ",
            "https://gateway.example.test",
            "gateway-token",
        ) {
            Ok(_) => panic!("whitespace alias must reject"),
            Err(error) => error,
        };

        assert_eq!(
            error.value_bound(py).to_string(),
            "remote alias must not contain surrounding whitespace"
        );
    });
}

#[test]
fn py_add_rejects_gateway_base_url_userinfo() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let error = match add_gateway(
            py,
            &client,
            "pro",
            "pro-chat",
            "https://user:gateway-secret@gateway.example.test",
            "gateway-token",
        ) {
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
fn py_gateway_options_reject_non_string_keys_with_gateway_error() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item(7, "provider-secret")
            .expect("set non-string key");

        let error = py_gateway_options_or_empty(py, Some(dict.into_py(py)))
            .expect_err("non-string keys must fail");
        let message = error.value_bound(py).to_string();

        assert_eq!(message, "JSON option object keys must be strings");
        assert!(!message.contains("provider-secret"));
    });
}

#[test]
fn py_gateway_options_reject_recursive_containers() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("self", dict.clone())
            .expect("set recursive dict");

        let error = py_gateway_options_or_empty(py, Some(dict.into_py(py)))
            .expect_err("recursive containers must fail");
        let message = error.value_bound(py).to_string();

        assert_eq!(message, "JSON options must contain JSON-compatible values");
    });
}

#[test]
fn py_remote_request_rejects_local_only_gateway_options() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let client = PyCogentClient::new().expect("client");
        let endpoint = add_gateway(
            py,
            &client,
            "pro",
            "pro-chat",
            "https://gateway.example.test",
            "gateway-token",
        )
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
                None,
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
