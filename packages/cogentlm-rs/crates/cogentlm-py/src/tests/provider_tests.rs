//! Provider binding tests for the parent module.

use super::*;
use std::time::Duration;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn py_provider_proxy_config_maps_static_headers() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let auth = Py::new(
            py,
            PyProviderAuth {
                core: ProviderAuth::Bearer(SecretString::new("token")),
            },
        )
        .expect("auth");
        let config = PyProviderProxyConfig::new(
            py,
            "http://localhost".to_string(),
            auth,
            "openai_compatible".to_string(),
            Some(vec![("x-cogent-test".to_string(), "yes".to_string())]),
            Some(1500),
        )
        .expect("proxy config");
        let core = config.to_core();

        assert_eq!(
            core.static_headers,
            vec![("x-cogent-test".to_string(), "yes".to_string())]
        );
        assert_eq!(core.protocol, ProxyProtocol::OpenAiCompatible);
        assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
    });
}

#[test]
fn py_provider_anthropic_client_uses_native_kind() {
    pyo3::prepare_freethreaded_python();
    let client = PyProviderClient::anthropic(
        "token".to_string(),
        Some("http://localhost".to_string()),
        Some("2023-06-01".to_string()),
        Some(1500),
    )
    .expect("anthropic client");

    assert_eq!(client.kind(), "anthropic");
}

#[test]
fn py_provider_options_must_be_json_object() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let list = PyList::empty_bound(py).into_py(py);
        assert!(py_provider_options_or_empty(py, Some(list)).is_err());

        let dict = PyDict::new_bound(py);
        dict.set_item("seed", 7).expect("set seed");
        let options = py_provider_options_or_empty(py, Some(dict.into_py(py))).expect("options");

        assert_eq!(options.get("seed"), Some(&serde_json::json!(7)));
    });
}

#[test]
fn py_provider_generation_options_reject_invalid_numbers() {
    assert!(PyProviderGenerationOptions::new(Some(0), None, None, None).is_err());
    assert!(PyProviderGenerationOptions::new(None, Some(f32::NAN), None, None).is_err());
}

#[test]
fn py_provider_chat_response_maps_usage_and_metadata() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let response = ProviderChatResponse {
            result: ProviderTextOutput {
                text: "hi".to_string(),
                finish_reason: cogentlm_engine::engine::FinishReason::Stop,
            },
            usage: Some(TokenUsage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                total_tokens: Some(3),
            }),
            metadata: ProviderResponseMetadata {
                provider: cogentlm_providers::ProviderKind::Proxy,
                model: "proxy-model".to_string(),
                request_id: Some("req-1".to_string()),
                response_id: Some("resp-1".to_string()),
                finish_reason_raw: Some("stop".to_string()),
                raw: serde_json::json!({ "id": "resp-1" }),
            },
        };

        let value = provider_chat_response_to_dict(py, response).expect("response dict");
        let dict = value.bind(py).downcast::<PyDict>().expect("dict");
        let result = dict
            .get_item("result")
            .expect("result")
            .expect("result item")
            .downcast_into::<PyDict>()
            .expect("result dict");
        let usage = dict
            .get_item("usage")
            .expect("usage")
            .expect("usage item")
            .downcast_into::<PyDict>()
            .expect("usage dict");
        let metadata = dict
            .get_item("metadata")
            .expect("metadata")
            .expect("metadata item")
            .downcast_into::<PyDict>()
            .expect("metadata dict");

        assert_eq!(
            result
                .get_item("finish_reason")
                .expect("finish reason")
                .expect("finish reason item")
                .extract::<String>()
                .expect("finish reason string"),
            "stop"
        );
        assert_eq!(
            usage
                .get_item("total_tokens")
                .expect("total tokens")
                .expect("total tokens item")
                .extract::<u32>()
                .expect("total tokens value"),
            3
        );
        assert_eq!(
            metadata
                .get_item("request_id")
                .expect("request id")
                .expect("request id item")
                .extract::<String>()
                .expect("request id string"),
            "req-1"
        );
    });
}

#[test]
fn py_provider_embedding_response_maps_usage_and_metadata() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let response = ProviderEmbeddingResponse {
            result: ProviderEmbeddingOutput {
                values: vec![0.25, -0.5],
            },
            usage: Some(TokenUsage {
                input_tokens: Some(2),
                output_tokens: None,
                total_tokens: Some(2),
            }),
            metadata: ProviderResponseMetadata {
                provider: cogentlm_providers::ProviderKind::Proxy,
                model: "proxy-model".to_string(),
                request_id: Some("req-embed".to_string()),
                response_id: None,
                finish_reason_raw: None,
                raw: serde_json::json!({ "object": "list" }),
            },
        };

        let value = provider_embedding_response_to_dict(py, response).expect("response dict");
        let dict = value.bind(py).downcast::<PyDict>().expect("dict");
        let result = dict
            .get_item("result")
            .expect("result")
            .expect("result item")
            .downcast_into::<PyDict>()
            .expect("result dict");
        let usage = dict
            .get_item("usage")
            .expect("usage")
            .expect("usage item")
            .downcast_into::<PyDict>()
            .expect("usage dict");
        let metadata = dict
            .get_item("metadata")
            .expect("metadata")
            .expect("metadata item")
            .downcast_into::<PyDict>()
            .expect("metadata dict");

        assert_eq!(
            result
                .get_item("values")
                .expect("values")
                .expect("values item")
                .extract::<Vec<f32>>()
                .expect("values list"),
            vec![0.25_f32, -0.5_f32]
        );
        assert_eq!(
            usage
                .get_item("total_tokens")
                .expect("total tokens")
                .expect("total tokens item")
                .extract::<u32>()
                .expect("total tokens value"),
            2
        );
        assert_eq!(
            metadata
                .get_item("request_id")
                .expect("request id")
                .expect("request id item")
                .extract::<String>()
                .expect("request id string"),
            "req-embed"
        );
    });
}

#[test]
fn py_provider_error_has_structured_fields() {
    pyo3::prepare_freethreaded_python();
    let mut error = CoreProviderError::new(
        cogentlm_providers::ProviderErrorKind::RateLimited,
        cogentlm_providers::ProviderKind::Proxy,
        "too many requests",
    );
    error.status = Some(429);
    error.code = Some("rate_limit".to_string());
    error.retry_after = Some(Duration::from_millis(1500));
    error.request_id = Some("req-123".to_string());
    error.raw = Some(Box::new(serde_json::json!({ "error": "limit" })));

    let py_error = to_py_provider_error(error);

    Python::with_gil(|py| {
        assert!(py_error.matches(py, py.get_type_bound::<ProviderError>()));
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
                .getattr("provider")
                .expect("provider")
                .extract::<String>()
                .expect("provider string"),
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
        assert_eq!(
            value
                .getattr("request_id")
                .expect("request id")
                .extract::<String>()
                .expect("request id string"),
            "req-123"
        );
        assert_eq!(
            value
                .getattr("retry_after_ms")
                .expect("retry after")
                .extract::<f64>()
                .expect("retry after value"),
            1500.0
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

#[test]
fn py_provider_stream_callback_receives_proxy_tokens() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = PyModule::from_code_bound(
            py,
            r#"
seen = []
def on_token(batch):
    seen.append(batch["text"])
"#,
            "provider_callback_test.py",
            "provider_callback_test",
        )
        .expect("callback module");
        let callback = module.getattr("on_token").expect("callback").into_py(py);
        let runtime = provider_runtime().expect("runtime");

        let summary = py
            .allow_threads(|| {
                runtime.block_on(async {
                    let server = MockServer::start().await;
                    Mock::given(method("POST"))
                        .and(path("/chat/completions"))
                        .and(header("authorization", "Bearer token"))
                        .respond_with(
                            ResponseTemplate::new(200)
                                .insert_header("content-type", "text/event-stream")
                                .set_body_string(concat!(
                                    "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}\n\n",
                                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
                                    "data: [DONE]\n\n"
                                )),
                        )
                        .mount(&server)
                        .await;

                    let client = ProviderClient::proxy(ProxyConfig {
                        base_url: server.uri(),
                        auth: ProviderAuth::Bearer(SecretString::new("token")),
                        protocol: ProxyProtocol::OpenAiCompatible,
                        static_headers: Vec::new(),
                        timeout: None,
                    })
                    .map_err(to_py_provider_error)?;
                    let request = ProviderChatRequest {
                        model: "proxy-model".to_string(),
                        messages: vec![ChatMessage::new(ChatRole::User, "hi")],
                        options: ProviderGenerationOptions::default(),
                        provider_options: ProviderOptions::new(),
                    };

                    provider_stream_chat_to_py(client, request, Some(callback)).await
                })
            })
            .expect("stream summary");

        let seen = module
            .getattr("seen")
            .expect("seen")
            .extract::<Vec<String>>()
            .expect("seen values");
        assert_eq!(seen, vec!["hello".to_string()]);
        assert_eq!(summary.finish_reason.as_deref(), Some("stop"));
        assert_eq!(summary.usage.expect("usage").total_tokens, Some(2));
    });
}
