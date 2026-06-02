//! Tests the `client` module in `cogentlm-client`.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

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

fn expect_client_error<T>(result: CogentResult<T>, context: &str) -> CogentError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
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
