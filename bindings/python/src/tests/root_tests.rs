//! Unit tests for the Python endpoint and request mapping boundary.

use super::*;
use pyo3::types::{PyDict, PyList};

#[test]
fn py_endpoint_ref_supports_gateway_kind() {
    let endpoint = PyEndpointRef::gateway("gateway".to_string());

    assert_eq!(endpoint.kind(), "gateway");
    assert!(matches!(
        endpoint.to_core(),
        CoreEndpointRef::Gateway { id } if id == "gateway"
    ));
}

#[test]
fn py_chat_message_requires_valid_roles() {
    assert!(PyChatMessage::new("tool".to_string(), "bad".to_string()).is_err());

    let message = PyChatMessage::new("assistant".to_string(), "ok".to_string())
        .expect("chat message")
        .to_core()
        .expect("core message");

    assert_eq!(message.role, ChatRole::Assistant);
    assert_eq!(message.content, "ok");
}

#[test]
fn py_endpoint_options_require_json_objects() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let invalid = PyList::new_bound(py, [1, 2, 3]);
        assert!(py_endpoint_options_or_empty(py, Some(invalid.into_py(py))).is_err());

        let valid = PyDict::new_bound(py);
        valid.set_item("seed", 7).expect("seed");
        let options = py_endpoint_options_or_empty(py, Some(valid.into_py(py))).expect("options");
        assert_eq!(options["seed"], serde_json::json!(7));
    });
}

#[test]
fn py_gateway_descriptor_maps_routes_authentication_and_options() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let options = PyDict::new_bound(py);
        options.set_item("profile", "custom").expect("profile");
        let config = PyGatewayDescriptor::new(
            py,
            "developer-model".to_string(),
            "https://gateway.example.test".to_string(),
            "header",
            Some("secret".to_string()),
            Some("x-api-key".to_string()),
            Some(BTreeMap::from([(
                "x-tenant".to_string(),
                "developer".to_string(),
            )])),
            Some(5_000),
            Some("/generate".to_string()),
            Some("/conversation".to_string()),
            Some("/vectorize".to_string()),
            Some(options.into_py(py)),
        )
        .expect("config");

        assert_eq!(config.core.target, "developer-model");
        assert_eq!(config.core.routes.query, "/generate");
        assert_eq!(
            config.core.protocol_options["profile"],
            serde_json::json!("custom")
        );
        assert!(matches!(
            config.core.authentication,
            CoreGatewayAuthentication::Header { ref name, .. }
                if name == "x-api-key"
        ));
    });
}

#[test]
fn endpoint_response_dict_preserves_gateway_kind() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let value = endpoint_ref_to_dict(
            py,
            CoreEndpointRef::Gateway {
                id: "gateway".to_string(),
            },
        )
        .expect("endpoint");
        let endpoint = value.bind(py).downcast::<PyDict>().expect("dict");

        assert_eq!(
            endpoint
                .get_item("kind")
                .expect("kind")
                .expect("kind value")
                .extract::<String>()
                .expect("kind string"),
            "gateway"
        );
    });
}
