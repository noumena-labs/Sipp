use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::{CogentEngine, NativeRuntimeConfig};

use crate::dispatch::InferenceEndpoint;
use crate::engine_endpoint::EngineEndpoint;
#[cfg(feature = "providers")]
use crate::provider_endpoint::ProviderEndpoint;
use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError, CogentQueryRequest,
    CogentResult, CogentTextRun, EndpointCapabilities, EndpointRef,
};

/// Public inference facade over registered local and provider endpoints.
pub struct CogentClient {
    endpoints: HashMap<EndpointRef, Arc<dyn InferenceEndpoint>>,
    default_endpoint: Option<EndpointRef>,
}

impl CogentClient {
    /// Create an empty client with no registered endpoints.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            default_endpoint: None,
        }
    }

    /// Register an already-loaded local engine endpoint.
    pub async fn add_engine(
        &mut self,
        id: impl Into<String>,
        engine: CogentEngine,
    ) -> CogentResult<()> {
        let engine_id = normalize_id(id, "engine id")?;
        let endpoint = EndpointRef::LocalEngine { engine: engine_id };
        self.reject_duplicate(&endpoint)?;

        let state = engine.state().await?;
        let model = state
            .model
            .ok_or_else(|| CogentError::Internal("loaded engine has no model state".to_string()))?;
        let capabilities = EndpointCapabilities::from_local(&model.capabilities);
        self.endpoints.insert(
            endpoint.clone(),
            Arc::new(EngineEndpoint::new(endpoint, capabilities, engine)),
        );
        Ok(())
    }

    /// Load a local engine and register it under the provided endpoint id.
    pub async fn load_engine(
        &mut self,
        id: impl Into<String>,
        model_path: impl Into<PathBuf>,
        config: NativeRuntimeConfig,
    ) -> CogentResult<()> {
        let engine = CogentEngine::load(model_path.into(), config).await?;
        self.add_engine(id, engine).await
    }

    #[cfg(feature = "providers")]
    /// Register a provider-backed model endpoint.
    pub fn add_provider_model(
        &mut self,
        provider: impl Into<String>,
        model: impl Into<String>,
        client: cogentlm_providers::ProviderClient,
        executor: crate::ProviderExecutor,
    ) -> CogentResult<()> {
        let provider = normalize_id(provider, "provider id")?;
        let model = normalize_id(model, "provider model id")?;
        let endpoint = EndpointRef::ProviderModel { provider, model };
        self.reject_duplicate(&endpoint)?;
        self.endpoints.insert(
            endpoint.clone(),
            Arc::new(ProviderEndpoint::new(
                endpoint,
                EndpointCapabilities::unknown(),
                client,
                executor,
            )),
        );
        Ok(())
    }

    /// Set the endpoint used when a request does not name one explicitly.
    pub fn set_default_endpoint(&mut self, endpoint: EndpointRef) -> CogentResult<()> {
        if !self.endpoints.contains_key(&endpoint) {
            return Err(CogentError::EndpointNotFound(endpoint));
        }
        self.default_endpoint = Some(endpoint);
        Ok(())
    }

    /// Submit a raw-prompt text generation request.
    pub fn query(&self, request: CogentQueryRequest) -> CogentTextRun {
        match self.resolve(request.endpoint.as_ref(), "query") {
            Ok(endpoint) => endpoint.query(request),
            Err(error) => CogentTextRun::ready_err(error),
        }
    }

    /// Submit a chat generation request.
    pub fn chat(&self, request: CogentChatRequest) -> CogentTextRun {
        match self.resolve(request.endpoint.as_ref(), "chat") {
            Ok(endpoint) => endpoint.chat(request),
            Err(error) => CogentTextRun::ready_err(error),
        }
    }

    /// Submit a single-input embedding request.
    pub fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun {
        match self.resolve(request.endpoint.as_ref(), "embed") {
            Ok(endpoint) => endpoint.embed(request),
            Err(error) => CogentEmbeddingRun::ready_err(error),
        }
    }

    fn resolve(
        &self,
        requested: Option<&EndpointRef>,
        operation: &'static str,
    ) -> CogentResult<Arc<dyn InferenceEndpoint>> {
        let selected = if let Some(endpoint) = requested {
            endpoint
        } else if let Some(endpoint) = &self.default_endpoint {
            endpoint
        } else {
            return self.resolve_single_local(operation);
        };
        let endpoint = self
            .endpoints
            .get(selected)
            .cloned()
            .ok_or_else(|| CogentError::EndpointNotFound(selected.clone()))?;
        ensure_supported(endpoint.as_ref(), operation)?;
        Ok(endpoint)
    }

    fn resolve_single_local(
        &self,
        operation: &'static str,
    ) -> CogentResult<Arc<dyn InferenceEndpoint>> {
        let mut matches = self
            .endpoints
            .values()
            .filter(|endpoint| endpoint.endpoint().is_local_engine())
            .filter(|endpoint| {
                endpoint.capabilities().for_operation(operation) == CapabilitySupport::Supported
            });

        let Some(endpoint) = matches.next().cloned() else {
            return Err(CogentError::NoSupportedEndpoint { operation });
        };
        if matches.next().is_some() {
            return Err(CogentError::AmbiguousEndpoint { operation });
        }
        Ok(endpoint)
    }

    fn reject_duplicate(&self, endpoint: &EndpointRef) -> CogentResult<()> {
        if self.endpoints.contains_key(endpoint) {
            Err(CogentError::InvalidRequest(
                "endpoint already registered".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

impl Default for CogentClient {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_supported(endpoint: &dyn InferenceEndpoint, operation: &'static str) -> CogentResult<()> {
    if endpoint.capabilities().for_operation(operation) == CapabilitySupport::Unsupported {
        Err(CogentError::UnsupportedOperation {
            endpoint: endpoint.endpoint().clone(),
            operation,
        })
    } else {
        Ok(())
    }
}

fn normalize_id(id: impl Into<String>, name: &'static str) -> CogentResult<String> {
    let id = id.into();
    let id = id.trim().to_string();
    if id.is_empty() {
        Err(CogentError::InvalidRequest(format!(
            "{name} must not be empty"
        )))
    } else {
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
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
        let selected = EndpointRef::LocalEngine {
            engine: "local-a".to_string(),
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
            EndpointRef::LocalEngine {
                engine: "local-b".to_string(),
            },
            Arc::new(FakeEndpoint {
                endpoint: EndpointRef::LocalEngine {
                    engine: "local-b".to_string(),
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
        let endpoint = EndpointRef::LocalEngine {
            engine: "local".to_string(),
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
}
