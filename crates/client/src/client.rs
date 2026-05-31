use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::{CogentEngine, NativeRuntimeConfig};

use crate::dispatch::InferenceEndpoint;
use crate::local_endpoint::LocalEndpoint;
#[cfg(feature = "providers")]
use crate::remote_endpoint::RemoteEndpoint;
#[cfg(feature = "providers")]
use crate::remote_executor::RemoteExecutor;
#[cfg(feature = "providers")]
use crate::RemoteConfig;
use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError, CogentQueryRequest,
    CogentResult, CogentTextRun, EndpointCapabilities, EndpointRef,
};

/// Public inference facade over registered local and remote endpoints.
pub struct CogentClient {
    endpoints: HashMap<EndpointRef, Arc<dyn InferenceEndpoint>>,
    #[cfg(feature = "providers")]
    remote_executor: Option<RemoteExecutor>,
}

impl CogentClient {
    /// Create an empty client with no registered endpoints.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            #[cfg(feature = "providers")]
            remote_executor: None,
        }
    }

    async fn register_local(
        &mut self,
        id: impl Into<String>,
        engine: CogentEngine,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "local id")?;
        let endpoint = EndpointRef::Local { id };
        self.reject_duplicate(&endpoint)?;

        let state = engine.state().await?;
        let model = state
            .model
            .ok_or_else(|| CogentError::Internal("loaded engine has no model state".to_string()))?;
        let capabilities = EndpointCapabilities::from_local(&model.capabilities);
        self.endpoints.insert(
            endpoint.clone(),
            Arc::new(LocalEndpoint::new(endpoint.clone(), capabilities, engine)),
        );
        Ok(endpoint)
    }

    /// Load a local model and register it under the provided endpoint id.
    pub async fn add_local(
        &mut self,
        id: impl Into<String>,
        model_path: impl Into<PathBuf>,
        config: NativeRuntimeConfig,
    ) -> CogentResult<EndpointRef> {
        let engine = CogentEngine::load(model_path.into(), config).await?;
        self.register_local(id, engine).await
    }

    /// Register a remote model endpoint.
    #[cfg(feature = "providers")]
    pub fn add_remote(
        &mut self,
        id: impl Into<String>,
        config: RemoteConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "remote id")?;
        let endpoint = EndpointRef::Remote { id };
        self.reject_duplicate(&endpoint)?;
        let (model, client) = config.build()?;
        let model = normalize_id(model, "remote model id")?;
        let executor = self.remote_executor()?;
        self.endpoints.insert(
            endpoint.clone(),
            Arc::new(RemoteEndpoint::new(
                endpoint.clone(),
                model,
                EndpointCapabilities::unknown(),
                client,
                executor,
            )),
        );
        Ok(endpoint)
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
            .filter(|endpoint| endpoint.endpoint().is_local())
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

    #[cfg(feature = "providers")]
    fn remote_executor(&mut self) -> CogentResult<RemoteExecutor> {
        if let Some(executor) = &self.remote_executor {
            return Ok(executor.clone());
        }

        let executor = RemoteExecutor::new()?;
        self.remote_executor = Some(executor.clone());
        Ok(executor)
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
}
