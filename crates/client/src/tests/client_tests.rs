//! Unit tests for the parent module.

use super::*;
use crate::dispatch::InferenceEndpoint;

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

#[test]
fn automatic_resolution_is_local_only_and_support_based() {
    let mut client = CogentClient::new();
    let selected = EndpointRef::Local {
        id: "local-a".to_string(),
    };
    client.endpoints.insert(
        selected.clone(),
        Arc::new(FakeEndpoint {
            endpoint: selected.clone(),
            capabilities: EndpointCapabilities {
                query: CapabilitySupport::Supported,
                chat: CapabilitySupport::Unsupported,
                embed: CapabilitySupport::Unsupported,
            },
        }),
    );
    client.endpoints.insert(
        EndpointRef::Local {
            id: "local-b".to_string(),
        },
        Arc::new(FakeEndpoint {
            endpoint: EndpointRef::Local {
                id: "local-b".to_string(),
            },
            capabilities: EndpointCapabilities {
                query: CapabilitySupport::Unsupported,
                chat: CapabilitySupport::Supported,
                embed: CapabilitySupport::Unsupported,
            },
        }),
    );

    let endpoint = client.resolve(None, "query").expect("resolved endpoint");

    assert_eq!(endpoint.endpoint(), &selected);
}

#[test]
fn duplicate_endpoint_registration_is_invalid() {
    let mut client = CogentClient::new();
    let endpoint = EndpointRef::Local {
        id: "local".to_string(),
    };
    client.endpoints.insert(
        endpoint.clone(),
        Arc::new(FakeEndpoint {
            endpoint: endpoint.clone(),
            capabilities: EndpointCapabilities {
                query: CapabilitySupport::Supported,
                chat: CapabilitySupport::Unsupported,
                embed: CapabilitySupport::Unsupported,
            },
        }),
    );

    let error = client
        .reject_duplicate(&endpoint)
        .expect_err("duplicate must reject");

    assert!(matches!(error, CogentError::InvalidRequest(_)));
}
