//! Tests the `client` module in `cogentlm-client`.
//!
//! Covers endpoint registration, resolution, and facade dispatch through fake
//! endpoints and provider configs rather than loaded models or remote calls.

use super::*;
use crate::dispatch::InferenceEndpoint;
#[cfg(feature = "providers")]
use crate::{CogentTextOptions, ProviderAuthConfig, ProviderEndpointConfig, ProviderSecret};
#[cfg(feature = "remote")]
use crate::{LocalEmbedOptions, LocalTextOptions, RemoteGatewayConfig, RemoteSecret};

#[cfg(any(feature = "remote", feature = "providers"))]
use futures::StreamExt;
#[cfg(any(feature = "remote", feature = "providers"))]
use serde_json::json;
#[cfg(any(feature = "remote", feature = "providers"))]
use wiremock::matchers::{body_json, header, method, path};
#[cfg(any(feature = "remote", feature = "providers"))]
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

#[cfg(feature = "providers")]
#[test]
fn automatic_resolution_ignores_provider_endpoints() {
    let mut client = CogentClient::new();
    let provider = EndpointRef::Provider {
        id: "provider-a".to_string(),
    };
    insert_fake(&mut client, provider, supported_capabilities());

    let error = match client.resolve(None, "query") {
        Ok(_) => panic!("omitted endpoint must not select provider"),
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
fn replace_endpoint_preserves_same_kind_ref() {
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

    client.replace_endpoint(
        endpoint.clone(),
        Arc::new(FakeEndpoint {
            endpoint: endpoint.clone(),
            capabilities: supported_capabilities(),
        }),
    );
    let resolved = client
        .resolve(Some(&endpoint), "embed")
        .expect("replacement endpoint");

    assert_eq!(resolved.endpoint(), &endpoint);
    assert_eq!(client.endpoints.len(), 1);
}

#[test]
fn replace_endpoint_invalidates_cross_kind_ref() {
    let mut client = CogentClient::new();
    let local = EndpointRef::Local {
        id: "shared".to_string(),
    };
    let remote = EndpointRef::Remote {
        id: "shared".to_string(),
    };
    insert_fake(&mut client, local.clone(), supported_capabilities());

    client.replace_endpoint(
        remote.clone(),
        Arc::new(FakeEndpoint {
            endpoint: remote.clone(),
            capabilities: supported_capabilities(),
        }),
    );
    let old_error = match client.resolve(Some(&local), "query") {
        Ok(_) => panic!("old cross-kind ref must not resolve"),
        Err(error) => error,
    };
    let resolved = client
        .resolve(Some(&remote), "query")
        .expect("replacement remote");

    assert!(matches!(old_error, CogentError::EndpointNotFound(found) if found == local));
    assert_eq!(resolved.endpoint(), &remote);
    assert_eq!(client.endpoints.len(), 1);
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
fn rust_example_env_helpers_preserve_values_for_core_validation() {
    let source = include_str!("../../../../examples/rust/src/support.rs");
    let required_env = source
        .split("pub fn required_env")
        .nth(1)
        .and_then(|section| section.split("\n}").next())
        .expect("rust example required_env helper");

    assert!(required_env.contains("env::var(name)"));
    assert!(!required_env.contains("trim"));
}

#[cfg(feature = "remote")]
#[test]
fn add_validates_remote_id_before_gateway_transport_config() {
    let mut client = CogentClient::new();
    let error = futures::executor::block_on(client.add(
        " pro ",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: "pro-chat".to_string(),
            base_url: "https://user:gateway-secret@gateway.example.test".to_string(),
            token: RemoteSecret::new("gateway-token"),
            timeout: None,
        }),
    ))
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
fn add_validates_remote_alias_before_gateway_transport_config() {
    let mut client = CogentClient::new();
    let error = futures::executor::block_on(client.add(
        "pro",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: "   ".to_string(),
            base_url: "https://user:gateway-secret@gateway.example.test".to_string(),
            token: RemoteSecret::new("gateway-token"),
            timeout: None,
        }),
    ))
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
fn add_rejects_remote_alias_with_surrounding_whitespace() {
    let mut client = CogentClient::new();
    let error = futures::executor::block_on(client.add(
        "pro",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: " pro-chat ".to_string(),
            base_url: "https://gateway.example.test".to_string(),
            token: RemoteSecret::new("gateway-token"),
            timeout: None,
        }),
    ))
    .expect_err("whitespace alias must reject");

    assert!(matches!(
        error,
        CogentError::InvalidRequest(message)
            if message == "remote alias must not contain surrounding whitespace"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn unified_add_registers_remote_gateway_descriptors() {
    let mut client = CogentClient::new();
    let endpoint = futures::executor::block_on(client.add(
        "pro",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: "pro-chat".to_string(),
            base_url: "https://gateway.example.test".to_string(),
            token: RemoteSecret::new("gateway-token"),
            timeout: None,
        }),
    ))
    .expect("add gateway descriptor");

    assert_eq!(endpoint, EndpointRef::Remote { id: "pro".into() });
}

#[cfg(all(feature = "remote", feature = "providers"))]
#[test]
fn unified_add_replaces_endpoint_across_kinds() {
    let mut client = CogentClient::new();
    let remote = futures::executor::block_on(client.add(
        "shared",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: "pro-chat".to_string(),
            base_url: "https://gateway.example.test".to_string(),
            token: RemoteSecret::new("gateway-token"),
            timeout: None,
        }),
    ))
    .expect("add gateway");
    let provider = futures::executor::block_on(client.add(
        "shared",
        EndpointDescriptor::provider(ProviderEndpointConfig::openai_compatible(
            "direct-model",
            "https://provider.example.test",
            ProviderAuthConfig::Bearer(ProviderSecret::new("provider-token")),
        )),
    ))
    .expect("replace with provider");

    let old_error = match client.resolve(Some(&remote), "query") {
        Ok(_) => panic!("old gateway ref must not resolve"),
        Err(error) => error,
    };
    let resolved = client
        .resolve(Some(&provider), "query")
        .expect("provider replacement");

    assert_eq!(
        provider,
        EndpointRef::Provider {
            id: "shared".into()
        }
    );
    assert!(matches!(old_error, CogentError::EndpointNotFound(found) if found == remote));
    assert_eq!(resolved.endpoint(), &provider);
}

#[cfg(feature = "providers")]
#[test]
fn unified_add_registers_provider_and_routes_query() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .and(header("authorization", "Bearer provider-token"))
            .and(body_json(json!({
                "model": "direct-model",
                "prompt": "hello",
                "max_tokens": 3
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "cmpl-1",
                "model": "direct-model",
                "choices": [{ "text": "hi", "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut client = CogentClient::new();
        let endpoint = client
            .add(
                "openai-compatible",
                EndpointDescriptor::provider(ProviderEndpointConfig::openai_compatible(
                    "direct-model",
                    server.uri(),
                    ProviderAuthConfig::Bearer(ProviderSecret::new("provider-token")),
                )),
            )
            .await
            .expect("add provider descriptor");
        let response = client
            .query(CogentQueryRequest {
                endpoint: Some(endpoint.clone()),
                prompt: "hello".to_string(),
                options: CogentTextOptions {
                    max_tokens: Some(3),
                    ..CogentTextOptions::default()
                },
                ..CogentQueryRequest::default()
            })
            .await
            .expect("provider query response");

        assert_eq!(
            endpoint,
            EndpointRef::Provider {
                id: "openai-compatible".into()
            }
        );
        assert_eq!(response.endpoint, endpoint);
        assert_eq!(response.text, "hi");
        assert_eq!(response.usage.expect("usage").total_tokens, Some(2));
        assert!(response.local_stats.is_none());
    });
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
            .add(
                "pro",
                EndpointDescriptor::gateway(remote_config(&server, "alias", "token")),
            )
            .await
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
fn repeated_add_rotates_gateway_config_without_changing_endpoint_ref() {
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
            .add(
                "pro",
                EndpointDescriptor::gateway(remote_config(&server, "alias-one", "token-one")),
            )
            .await
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
            .add(
                "pro",
                EndpointDescriptor::gateway(remote_config(&server, "alias-two", "token-two")),
            )
            .await
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
fn failed_replacement_preserves_registered_gateway() {
    let mut client = CogentClient::new();
    let endpoint = futures::executor::block_on(client.add(
        "pro",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: "pro-chat".to_string(),
            base_url: "https://gateway.example.test".to_string(),
            token: RemoteSecret::new("gateway-token"),
            timeout: None,
        }),
    ))
    .expect("add gateway");

    let error = futures::executor::block_on(client.add(
        "pro",
        EndpointDescriptor::gateway(RemoteGatewayConfig {
            alias: "replacement".to_string(),
            base_url: "https://user:secret@gateway.example.test".to_string(),
            token: RemoteSecret::new("replacement-token"),
            timeout: None,
        }),
    ))
    .expect_err("invalid replacement");
    let resolved = client
        .resolve(Some(&endpoint), "query")
        .expect("previous gateway remains registered");

    assert!(matches!(error, CogentError::Remote(_)));
    assert_eq!(resolved.endpoint(), &endpoint);
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
            .add(
                "pro",
                EndpointDescriptor::gateway(remote_config(&server, "alias", "token")),
            )
            .await
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
