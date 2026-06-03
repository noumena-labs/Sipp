//! Tests the `client` module in `cogentlm-client`.
//!
//! Covers endpoint registration, resolution, and facade dispatch through fake
//! endpoints and provider configs rather than loaded models or remote calls.

use super::*;
use crate::dispatch::InferenceEndpoint;
#[cfg(feature = "providers")]
use crate::{RemoteAuth, RemoteSecret};

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
    insert_fake(
        &mut client,
        EndpointRef::Remote {
            id: "remote".to_string(),
        },
        capabilities(
            CapabilitySupport::Supported,
            CapabilitySupport::Supported,
            CapabilitySupport::Supported,
        ),
    );

    let error = expect_client_error(
        client.resolve(None, "query"),
        "remote endpoint should not auto-select",
    );

    assert!(matches!(
        error,
        CogentError::NoSupportedEndpoint { operation: "query" }
    ));
}

#[test]
fn automatic_resolution_rejects_ambiguous_local_matches() {
    let mut client = CogentClient::new();
    for id in ["local-a", "local-b"] {
        insert_fake(
            &mut client,
            EndpointRef::Local { id: id.to_string() },
            capabilities(
                CapabilitySupport::Supported,
                CapabilitySupport::Unsupported,
                CapabilitySupport::Unsupported,
            ),
        );
    }

    let error = expect_client_error(
        client.resolve(None, "query"),
        "two local query endpoints are ambiguous",
    );

    assert!(matches!(
        error,
        CogentError::AmbiguousEndpoint { operation: "query" }
    ));
}

#[test]
fn facade_dispatches_query_chat_and_embed_to_selected_endpoint() {
    let mut client = CogentClient::new();
    let endpoint = EndpointRef::Local {
        id: "local".to_string(),
    };
    insert_fake(&mut client, endpoint.clone(), supported_capabilities());

    let query_error = futures::executor::block_on(client.query(CogentQueryRequest {
        endpoint: Some(endpoint.clone()),
        ..CogentQueryRequest::default()
    }))
    .expect_err("fake query error");
    assert!(matches!(
        query_error,
        CogentError::Internal(message) if message == "fake query"
    ));

    let chat_error = futures::executor::block_on(client.chat(CogentChatRequest {
        endpoint: Some(endpoint.clone()),
        ..CogentChatRequest::default()
    }))
    .expect_err("fake chat error");
    assert!(matches!(
        chat_error,
        CogentError::Internal(message) if message == "fake chat"
    ));

    let embed_error = futures::executor::block_on(client.embed(CogentEmbedRequest {
        endpoint: Some(endpoint),
        ..CogentEmbedRequest::default()
    }))
    .expect_err("fake embed error");
    assert!(matches!(
        embed_error,
        CogentError::Internal(message) if message == "fake embed"
    ));
}

#[test]
fn facade_returns_resolution_errors_as_ready_runs() {
    let client = CogentClient::new();
    let endpoint = EndpointRef::Local {
        id: "missing".to_string(),
    };

    let error = futures::executor::block_on(client.query(CogentQueryRequest {
        endpoint: Some(endpoint.clone()),
        ..CogentQueryRequest::default()
    }))
    .expect_err("missing endpoint");

    assert!(matches!(
        error,
        CogentError::EndpointNotFound(found) if found == endpoint
    ));

    let chat_error = futures::executor::block_on(client.chat(CogentChatRequest {
        endpoint: Some(endpoint.clone()),
        ..CogentChatRequest::default()
    }))
    .expect_err("missing endpoint");
    assert!(matches!(
        chat_error,
        CogentError::EndpointNotFound(found) if found == endpoint
    ));

    let embed_error = futures::executor::block_on(client.embed(CogentEmbedRequest {
        endpoint: Some(endpoint.clone()),
        ..CogentEmbedRequest::default()
    }))
    .expect_err("missing endpoint");
    assert!(matches!(
        embed_error,
        CogentError::EndpointNotFound(found) if found == endpoint
    ));
}

#[test]
fn explicit_resolution_allows_unknown_capability_support() {
    let mut client = CogentClient::new();
    let endpoint = EndpointRef::Remote {
        id: "remote".to_string(),
    };
    insert_fake(
        &mut client,
        endpoint.clone(),
        capabilities(
            CapabilitySupport::Unknown,
            CapabilitySupport::Unknown,
            CapabilitySupport::Unknown,
        ),
    );

    let resolved = client
        .resolve(Some(&endpoint), "query")
        .expect("unknown support is attempted explicitly");

    assert_eq!(resolved.endpoint(), &endpoint);
}

#[test]
fn explicit_resolution_rejects_missing_and_unsupported_endpoints() {
    let mut client = CogentClient::new();
    let unsupported = EndpointRef::Local {
        id: "local".to_string(),
    };
    insert_fake(
        &mut client,
        unsupported.clone(),
        capabilities(
            CapabilitySupport::Unsupported,
            CapabilitySupport::Supported,
            CapabilitySupport::Unsupported,
        ),
    );

    let missing = EndpointRef::Local {
        id: "missing".to_string(),
    };
    let missing_error =
        expect_client_error(client.resolve(Some(&missing), "query"), "missing endpoint");
    assert!(matches!(
        missing_error,
        CogentError::EndpointNotFound(endpoint) if endpoint == missing
    ));

    let unsupported_error = expect_client_error(
        client.resolve(Some(&unsupported), "query"),
        "unsupported endpoint",
    );
    assert!(matches!(
        unsupported_error,
        CogentError::UnsupportedOperation {
            endpoint,
            operation: "query"
        } if endpoint == unsupported
    ));
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
fn normalize_id_trims_and_rejects_blank_values() {
    assert_eq!(
        normalize_id("  endpoint  ", "endpoint").expect("normalized id"),
        "endpoint"
    );

    let error = normalize_id(" \t ", "endpoint").expect_err("blank id");
    assert!(matches!(
        error,
        CogentError::InvalidRequest(message) if message.contains("endpoint")
    ));
}

#[cfg(feature = "providers")]
#[test]
fn add_remote_registers_endpoint_and_reuses_executor() {
    let mut client = CogentClient::new();
    let first = client
        .add_remote(
            " remote-a ",
            RemoteConfig::proxy(
                "model-a",
                "http://localhost:11434",
                RemoteAuth::Bearer(RemoteSecret::new("secret-a")),
            ),
        )
        .expect("first remote");
    let second = client
        .add_remote(
            "remote-b",
            RemoteConfig::proxy(
                "model-b",
                "http://localhost:11435",
                RemoteAuth::Bearer(RemoteSecret::new("secret-b")),
            ),
        )
        .expect("second remote");

    assert_eq!(
        first,
        EndpointRef::Remote {
            id: "remote-a".to_string()
        }
    );
    assert_eq!(
        second,
        EndpointRef::Remote {
            id: "remote-b".to_string()
        }
    );
    assert!(client.remote_executor.is_some());
    assert_eq!(client.endpoints.len(), 2);
}

#[cfg(feature = "providers")]
#[test]
fn add_remote_rejects_blank_remote_id_before_transport_build() {
    let mut client = CogentClient::new();
    let error = expect_client_error(
        client.add_remote(
            " ",
            RemoteConfig::proxy(
                "model",
                "http://localhost:11434",
                RemoteAuth::Bearer(RemoteSecret::new("secret")),
            ),
        ),
        "blank remote id",
    );

    assert!(matches!(
        error,
        CogentError::InvalidRequest(message) if message.contains("remote id")
    ));
    assert!(client.remote_executor.is_none());
}

#[cfg(feature = "providers")]
#[test]
fn add_remote_rejects_blank_remote_model_id_after_building_transport() {
    let mut client = CogentClient::new();
    let error = expect_client_error(
        client.add_remote(
            "remote",
            RemoteConfig::proxy(
                " ",
                "http://localhost:11434",
                RemoteAuth::Bearer(RemoteSecret::new("secret")),
            ),
        ),
        "blank remote model id",
    );

    assert!(matches!(
        error,
        CogentError::InvalidRequest(message) if message.contains("remote model id")
    ));
}
