use std::collections::HashMap;
use std::sync::Arc;

use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::CogentEngine;

use crate::dispatch::InferenceEndpoint;
#[cfg(not(target_family = "wasm"))]
use crate::gateway_endpoint::GatewayEndpoint;
#[cfg(not(target_family = "wasm"))]
use crate::io_executor::IoExecutor;
use crate::local_endpoint::LocalEndpoint;
#[cfg(all(feature = "providers", not(target_family = "wasm")))]
use crate::provider_endpoint::ProviderEndpoint;
#[cfg(feature = "providers")]
use crate::ProviderEndpointConfig;
use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError, CogentQueryRequest,
    CogentRequestContext, CogentResult, CogentTextRun, EndpointCapabilities, EndpointDescriptor,
    EndpointRef,
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
    #[cfg(not(target_family = "wasm"))]
    io_executor: Option<IoExecutor>,
}

impl CogentClient {
    /// Create an empty client with no registered endpoints.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            #[cfg(not(target_family = "wasm"))]
            io_executor: None,
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
            EndpointDescriptor::Gateway(config) => self.register_gateway(id, config),
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

    #[cfg(not(target_family = "wasm"))]
    fn register_gateway(
        &mut self,
        id: impl Into<String>,
        config: crate::GatewayEndpointConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "gateway id")?;
        let endpoint = EndpointRef::Gateway { id };
        let executor = self.io_executor()?;
        self.replace_endpoint(
            endpoint.clone(),
            Arc::new(GatewayEndpoint::new(endpoint.clone(), config, executor)?),
        );
        Ok(endpoint)
    }

    #[cfg(target_family = "wasm")]
    fn register_gateway(
        &mut self,
        id: impl Into<String>,
        _config: crate::GatewayEndpointConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "gateway id")?;
        Err(CogentError::UnsupportedOperation {
            endpoint: EndpointRef::Gateway { id },
            operation: "gateway endpoint registration",
        })
    }

    #[cfg(all(feature = "providers", not(target_family = "wasm")))]
    fn register_provider(
        &mut self,
        id: impl Into<String>,
        config: ProviderEndpointConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "provider id")?;
        let endpoint = EndpointRef::Provider { id };
        let (model, transport, secrets) = config.build()?;
        let executor = self.io_executor()?;
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

    #[cfg(all(feature = "providers", target_family = "wasm"))]
    fn register_provider(
        &mut self,
        id: impl Into<String>,
        _config: ProviderEndpointConfig,
    ) -> CogentResult<EndpointRef> {
        let id = normalize_id(id, "provider id")?;
        Err(CogentError::UnsupportedOperation {
            endpoint: EndpointRef::Provider { id },
            operation: "provider endpoint registration",
        })
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
        self.query_with_context(CogentRequestContext::default(), request)
    }

    /// Submit raw-prompt generation with request-scoped correlation metadata.
    pub fn query_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentQueryRequest,
    ) -> CogentTextRun {
        match self.resolve(request.endpoint.as_ref(), "query") {
            Ok(endpoint) => endpoint.query_with_context(context, request),
            Err(error) => CogentTextRun::ready_err(error),
        }
    }

    /// Submit a chat generation request.
    pub fn chat(&self, request: CogentChatRequest) -> CogentTextRun {
        self.chat_with_context(CogentRequestContext::default(), request)
    }

    /// Submit chat generation with request-scoped correlation metadata.
    pub fn chat_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentChatRequest,
    ) -> CogentTextRun {
        match self.resolve(request.endpoint.as_ref(), "chat") {
            Ok(endpoint) => endpoint.chat_with_context(context, request),
            Err(error) => CogentTextRun::ready_err(error),
        }
    }

    /// Submit a single-input embedding request.
    pub fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun {
        self.embed_with_context(CogentRequestContext::default(), request)
    }

    /// Submit an embedding request with request-scoped correlation metadata.
    pub fn embed_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        match self.resolve(request.endpoint.as_ref(), "embed") {
            Ok(endpoint) => endpoint.embed_with_context(context, request),
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

    #[cfg(not(target_family = "wasm"))]
    fn io_executor(&mut self) -> CogentResult<IoExecutor> {
        if let Some(executor) = &self.io_executor {
            return Ok(executor.clone());
        }

        let executor = IoExecutor::new()?;
        self.io_executor = Some(executor.clone());
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
