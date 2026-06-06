use std::collections::HashMap;
use std::sync::Arc;

use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::CogentEngine;

use crate::dispatch::InferenceEndpoint;
use crate::local_endpoint::LocalEndpoint;
#[cfg(feature = "providers")]
use crate::provider_endpoint::ProviderEndpoint;
#[cfg(feature = "remote")]
use crate::remote_endpoint::RemoteEndpoint;
#[cfg(any(feature = "remote", feature = "providers"))]
use crate::remote_executor::RemoteExecutor;
#[cfg(feature = "providers")]
use crate::ProviderEndpointConfig;
#[cfg(feature = "remote")]
use crate::RemoteGatewayConfig;
use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError, CogentQueryRequest,
    CogentResult, CogentTextRun, EndpointCapabilities, EndpointDescriptor, EndpointRef,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/client_tests.rs"]
mod client_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Public inference facade over registered local, gateway, and provider endpoints.
pub struct CogentClient {
    endpoints: HashMap<EndpointRef, Arc<dyn InferenceEndpoint>>,
    #[cfg(any(feature = "remote", feature = "providers"))]
    remote_executor: Option<RemoteExecutor>,
}

impl CogentClient {
    /// Create an empty client with no registered endpoints.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            #[cfg(any(feature = "remote", feature = "providers"))]
            remote_executor: None,
        }
    }

    /// Register or replace a local, gateway, or direct provider endpoint.
    ///
    /// Reusing an id replaces the existing endpoint after the new descriptor
    /// has been validated and constructed. Changing endpoint kind invalidates
    /// previously returned references for that id.
    ///
    /// # Errors
    ///
    /// Returns an error when the id or descriptor is invalid, endpoint
    /// construction fails, or the requested endpoint feature is unavailable.
    pub async fn add(
        &mut self,
        id: impl Into<String>,
        descriptor: EndpointDescriptor,
    ) -> CogentResult<EndpointRef> {
        match descriptor {
            EndpointDescriptor::LocalModel(descriptor) => {
                let engine = CogentEngine::load(descriptor.model_path, descriptor.config).await?;
                self.register_local(id, engine).await
            }
            #[cfg(feature = "remote")]
            EndpointDescriptor::Gateway(config) => self.register_remote(id, config),
            #[cfg(feature = "providers")]
            EndpointDescriptor::Provider(config) => self.register_provider(id, config),
        }
    }

    async fn register_local(
        &mut self,
        id: impl Into<String>,
        engine: CogentEngine,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "local id")?;
        let endpoint = EndpointRef::Local { id };

        let state = engine.state().await?;
        let model = state
            .model
            .ok_or_else(|| CogentError::Internal("loaded engine has no model state".to_string()))?;
        let capabilities = EndpointCapabilities::from_local(&model.capabilities);
        self.replace_endpoint(
            endpoint.clone(),
            Arc::new(LocalEndpoint::new(endpoint.clone(), capabilities, engine)),
        );
        Ok(endpoint)
    }

    #[cfg(feature = "remote")]
    fn register_remote(
        &mut self,
        id: impl Into<String>,
        config: RemoteGatewayConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "remote id")?;
        let endpoint = EndpointRef::Remote { id };
        let (model, client) = config.build()?;
        let executor = self.remote_executor()?;
        self.replace_endpoint(
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

    #[cfg(feature = "providers")]
    fn register_provider(
        &mut self,
        id: impl Into<String>,
        config: ProviderEndpointConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "provider id")?;
        let endpoint = EndpointRef::Provider { id };
        let (model, transport, secrets) = config.build()?;
        let executor = self.remote_executor()?;
        self.replace_endpoint(
            endpoint.clone(),
            Arc::new(ProviderEndpoint::new(
                endpoint.clone(),
                model,
                EndpointCapabilities::unknown(),
                transport,
                executor,
                secrets,
            )),
        );
        Ok(endpoint)
    }

    fn replace_endpoint(
        &mut self,
        endpoint: EndpointRef,
        implementation: Arc<dyn InferenceEndpoint>,
    ) {
        let id = endpoint.id();
        self.endpoints.retain(|registered, _| registered.id() != id);
        self.endpoints.insert(endpoint, implementation);
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

    #[cfg(any(feature = "remote", feature = "providers"))]
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
    let trimmed = id.trim();
    if trimmed.is_empty() {
        Err(CogentError::InvalidRequest(format!(
            "{name} must not be empty"
        )))
    } else if trimmed != id.as_str() {
        Err(CogentError::InvalidRequest(format!(
            "{name} must not contain surrounding whitespace"
        )))
    } else {
        Ok(id)
    }
}
