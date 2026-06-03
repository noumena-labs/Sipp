//! Tests the `client` module in `cogentlm-client`.
//!
//! Covers endpoint registration, resolution, and facade dispatch through fake
//! endpoints and provider configs rather than loaded models or remote calls.

use super::*;
use crate::dispatch::InferenceEndpoint;
#[cfg(feature = "remote")]
use crate::{LocalEmbedOptions, LocalTextOptions, RemoteGatewayConfig, RemoteSecret};

#[cfg(feature = "remote")]
use futures::StreamExt;
#[cfg(feature = "remote")]
use serde_json::json;
#[cfg(feature = "remote")]
use wiremock::matchers::{body_json, header, method, path};
#[cfg(feature = "remote")]
use wiremock::{Mock, MockServer, ResponseTemplate};

struct FakeEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
}

impl InferenceEndpoint for FakeEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn query(&self, _request: CogentQueryRequest) -> CogentTextRun {
        CogentTextRun::ready_err(CogentError::Internal("fake query".to_string()))
    }

    fn chat(&self, _request: CogentChatRequest) -> CogentTextRun {
        CogentTextRun::ready_err(CogentError::Internal("fake chat".to_string()))
    }

    fn embed(&self, _request: CogentEmbedRequest) -> CogentEmbeddingRun {
        CogentEmbeddingRun::ready_err(CogentError::Internal("fake embed".to_string()))
    }
}

fn capabilities(
    query: CapabilitySupport,
    chat: CapabilitySupport,
    embed: CapabilitySupport,
) -> EndpointCapabilities {
    EndpointCapabilities { query, chat, embed }
}

fn insert_fake(
    client: &mut CogentClient,
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
) {
    client.endpoints.insert(
        endpoint.clone(),
        Arc::new(FakeEndpoint {
            endpoint,
            capabilities,
        }),
    );
}

fn supported_capabilities() -> EndpointCapabilities {
    capabilities(
        CapabilitySupport::Supported,
        CapabilitySupport::Supported,
        CapabilitySupport::Supported,
    )
}

fn expect_client_error<T>(result: CogentResult<T>, context: &str) -> CogentError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn default_client_starts_empty() {
    let client = CogentClient::default();
    let error = expect_client_error(
        client.resolve(None, "query"),
        "default client should not resolve endpoints",
    );

    assert!(matches!(
        error,
        CogentError::NoSupportedEndpoint { operation: "query" }
    ));
}

#[test]
fn automatic_resolution_is_local_only_and_support_based() {
    let mut client = CogentClient::new();
    let selected = EndpointRef::Local {
        id: "local-a".to_string(),
    };
    insert_fake(
        &mut client,
        selected.clone(),
        capabilities(
            CapabilitySupport::Supported,
            CapabilitySupport::Unsupported,
            CapabilitySupport::Unsupported,
        ),
    );
    insert_fake(
        &mut client,
        EndpointRef::Local {
            id: "local-b".to_string(),
        },
        capabilities(
            CapabilitySupport::Unsupported,
            CapabilitySupport::Supported,
            CapabilitySupport::Unsupported,
        ),
    );

    let endpoint = client.resolve(None, "query").expect("resolved endpoint");

    assert_eq!(endpoint.endpoint(), &selected);
}

#[test]
fn automatic_resolution_ignores_remote_endpoints() {
    let mut client = CogentClient::new();
    let remote = EndpointRef::Remote {
        id: "remote-a".to_string(),
    };
    client.endpoints.insert(
        remote.clone(),
        Arc::new(FakeEndpoint {
            endpoint: remote,
            capabilities: EndpointCapabilities {
                query: CapabilitySupport::Supported,
                chat: CapabilitySupport::Supported,
                embed: CapabilitySupport::Supported,
            },
        }),
    );

    let error = match client.resolve(None, "query") {
        Ok(_) => panic!("omitted endpoint must not select remote"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        CogentError::NoSupportedEndpoint { operation: "query" }
    ));
}

#[cfg(feature = "remote")]
#[test]
fn explicit_remote_allows_unknown_capabilities() {
    let mut client = CogentClient::new();
    let remote = EndpointRef::Remote {
        id: "remote-a".to_string(),
    };
    client.endpoints.insert(
        remote.clone(),
        Arc::new(FakeEndpoint {
            endpoint: remote.clone(),
            capabilities: EndpointCapabilities::unknown(),
        }),
    );

    let endpoint = client
        .resolve(Some(&remote), "query")
        .expect("unknown remote capability is gateway-authoritative");

    assert_eq!(endpoint.endpoint(), &remote);
}

#[test]
fn duplicate_endpoint_registration_is_invalid() {
    let mut client = CogentClient::new();
    let endpoint = EndpointRef::Local {
        id: "local".to_string(),
    };
    insert_fake(
        &mut client,
        endpoint.clone(),
        capabilities(
            CapabilitySupport::Supported,
            CapabilitySupport::Unsupported,
            CapabilitySupport::Unsupported,
        ),
    );

    let error = client
        .reject_duplicate(&endpoint)
        .expect_err("duplicate must reject");

    assert!(matches!(error, CogentError::InvalidRequest(_)));
}

#[test]
fn endpoint_ids_must_not_contain_surrounding_whitespace() {
    let error = normalize_id(" local ", "local id").expect_err("whitespace id must reject");

    assert!(matches!(
        error,
        CogentError::InvalidRequest(message)
            if message == "local id must not contain surrounding whitespace"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn remote_gateway_examples_preserve_env_values_for_core_validation() {
    let source = include_str!("../../examples/remote_common/mod.rs");
    let env_string = source
        .split("fn env_string")
        .nth(1)
        .and_then(|section| section.split("\n}").next())
        .expect("remote example env_string helper");

    assert!(env_string.contains("env::var(name)"));
    assert!(!env_string.contains("trim"));
}

#[cfg(feature = "remote")]
#[test]
fn add_remote_validates_id_before_gateway_transport_config() {
    let mut client = CogentClient::new();
    let error = client
        .add_remote(
            " pro ",
            RemoteGatewayConfig {
                alias: "pro-chat".to_string(),
                base_url: "https://user:gateway-secret@gateway.example.test".to_string(),
                token: RemoteSecret::new("gateway-token"),
                timeout: None,
            },
        )
        .expect_err("remote id must reject before transport config");

    assert!(matches!(
        &error,
        CogentError::InvalidRequest(message)
            if message == "remote id must not contain surrounding whitespace"
    ));
    let message = error.to_string();
    assert!(!message.contains("gateway-secret"));
    assert!(!message.contains("gateway-token"));
}

#[cfg(feature = "remote")]
#[test]
fn add_remote_validates_alias_before_gateway_transport_config() {
    let mut client = CogentClient::new();
    let error = client
        .add_remote(
            "pro",
            RemoteGatewayConfig {
                alias: "   ".to_string(),
                base_url: "https://user:gateway-secret@gateway.example.test".to_string(),
                token: RemoteSecret::new("gateway-token"),
                timeout: None,
            },
        )
        .expect_err("blank alias must reject before transport config");

    assert!(matches!(
        &error,
        CogentError::InvalidRequest(message) if message == "remote alias must not be empty"
    ));
    let message = error.to_string();
    assert!(!message.contains("gateway-secret"));
    assert!(!message.contains("gateway-token"));
}

#[cfg(feature = "remote")]
#[test]
fn add_remote_rejects_alias_with_surrounding_whitespace() {
    let mut client = CogentClient::new();
    let error = client
        .add_remote(
            "pro",
            RemoteGatewayConfig {
                alias: " pro-chat ".to_string(),
                base_url: "https://gateway.example.test".to_string(),
                token: RemoteSecret::new("gateway-token"),
                timeout: None,
            },
        )
        .expect_err("whitespace alias must reject");

    assert!(matches!(
        error,
        CogentError::InvalidRequest(message)
            if message == "remote alias must not contain surrounding whitespace"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn explicit_remote_rejects_local_only_request_fields() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let server = MockServer::start().await;
        let mut client = CogentClient::new();
        let endpoint = client
            .add_remote("pro", remote_config(&server, "alias", "token"))
            .expect("add remote");

        let query_error = client
            .query(CogentQueryRequest {
                endpoint: Some(endpoint.clone()),
                prompt: "hello".to_string(),
                local: LocalTextOptions {
                    context_key: Some("ctx".to_string()),
                    ..LocalTextOptions::default()
                },
                ..CogentQueryRequest::default()
            })
            .await
            .expect_err("remote query must reject local text options");
        let embed_error = client
            .embed(CogentEmbedRequest {
                endpoint: Some(endpoint),
                input: "hello".to_string(),
                local: LocalEmbedOptions {
                    normalize: Some(true),
                    ..LocalEmbedOptions::default()
                },
                ..CogentEmbedRequest::default()
            })
            .await
            .expect_err("remote embed must reject local embed options");

        assert!(matches!(
            query_error,
            CogentError::InvalidRequest(message)
                if message == "local text options are not valid for remote endpoints"
        ));
        assert!(matches!(
            embed_error,
            CogentError::InvalidRequest(message)
                if message == "local embed options are not valid for remote endpoints"
        ));
    });
}

#[cfg(feature = "remote")]
#[test]
fn update_remote_rotates_gateway_config_without_changing_endpoint_id() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/query"))
            .and(header("authorization", "Bearer token-one"))
            .and(body_json(json!({
                "model": "alias-one",
                "prompt": "hello",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp-one",
                "model": "alias-one",
                "text": "first",
                "finish_reason": "stop"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/query"))
            .and(header("authorization", "Bearer token-two"))
            .and(body_json(json!({
                "model": "alias-two",
                "prompt": "hello",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp-two",
                "model": "alias-two",
                "text": "second",
                "finish_reason": "stop"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut client = CogentClient::new();
        let endpoint = client
            .add_remote("pro", remote_config(&server, "alias-one", "token-one"))
            .expect("add remote");
        let first = client
            .query(CogentQueryRequest {
                endpoint: Some(endpoint.clone()),
                prompt: "hello".to_string(),
                ..CogentQueryRequest::default()
            })
            .await
            .expect("first remote query");

        let updated = client
            .update_remote("pro", remote_config(&server, "alias-two", "token-two"))
            .expect("update remote");
        let second = client
            .query(CogentQueryRequest {
                endpoint: Some(updated.clone()),
                prompt: "hello".to_string(),
                ..CogentQueryRequest::default()
            })
            .await
            .expect("second remote query");

        assert_eq!(endpoint, EndpointRef::Remote { id: "pro".into() });
        assert_eq!(updated, endpoint);
        assert_eq!(first.text, "first");
        assert_eq!(second.text, "second");
    });
}

#[cfg(feature = "remote")]
#[test]
fn remote_stream_requires_terminal_done_event() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/query"))
            .and(header("authorization", "Bearer token"))
            .and(body_json(json!({
                "model": "alias",
                "prompt": "hello",
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("x-request-id", "req-truncated")
                    .set_body_string("event: token\ndata: {\"text\":\"partial\"}\n\n"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let mut client = CogentClient::new();
        let endpoint = client
            .add_remote("pro", remote_config(&server, "alias", "token"))
            .expect("add remote");
        let run = client.query(CogentQueryRequest {
            endpoint: Some(endpoint),
            prompt: "hello".to_string(),
            emit_tokens: true,
            ..CogentQueryRequest::default()
        });
        let (mut tokens, response) = run.into_parts();

        let batch = tokens.next().await.expect("partial token batch");
        assert_eq!(batch.text, "partial");
        assert!(tokens.next().await.is_none());

        let error = response
            .await
            .expect_err("truncated stream must not produce a final response");
        assert!(matches!(
            error,
            CogentError::Remote(remote)
                if remote.message == "gateway stream ended before done event"
        ));
    });
}

#[cfg(feature = "remote")]
fn remote_config(server: &MockServer, alias: &str, token: &str) -> RemoteGatewayConfig {
    RemoteGatewayConfig {
        alias: alias.to_string(),
        base_url: server.uri(),
        token: RemoteSecret::new(token),
        timeout: None,
    }
}
