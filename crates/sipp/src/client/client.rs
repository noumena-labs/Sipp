use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::core::CapabilitySupport;
use crate::engine::SippEngine;
use crate::lifecycle::{ModelLoadOptions, ModelStore};

use crate::client::dispatch::InferenceEndpoint;
#[cfg(not(target_family = "wasm"))]
use crate::client::gateway_endpoint::GatewayEndpoint;
#[cfg(not(target_family = "wasm"))]
use crate::client::io_executor::IoExecutor;
use crate::client::local_endpoint::LocalEndpoint;
#[cfg(all(feature = "providers", not(target_family = "wasm")))]
use crate::client::provider_endpoint::ProviderEndpoint;
#[cfg(feature = "providers")]
use crate::client::ProviderDescriptor;
use crate::client::{
    EndpointCapabilities, EndpointDescriptor, EndpointRef, SippChatRequest, SippEmbedRequest,
    SippEmbeddingRun, SippError, SippQueryRequest, SippRequestContext, SippResult, SippTextRun,
    DEFAULT_STORAGE_ROOT,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/client_tests.rs"]
mod client_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Public inference facade over registered local, gateway, and provider endpoints.
pub struct SippClient {
    models: ModelStore,
    endpoints: HashMap<EndpointRef, Arc<dyn InferenceEndpoint>>,
    local_models: HashMap<String, String>,
    #[cfg(not(target_family = "wasm"))]
    io_executor: Option<IoExecutor>,
}

impl SippClient {
    /// Create an empty client with no registered endpoints.
    ///
    /// # Errors
    ///
    /// Returns an error when the default model store cannot be opened.
    pub fn new() -> SippResult<Self> {
        Self::with_storage_root(DEFAULT_STORAGE_ROOT)
    }

    /// Create an empty client with a model store rooted at `storage_root`.
    ///
    /// # Errors
    ///
    /// Returns an error when the model store cannot be opened.
    pub fn with_storage_root(storage_root: impl Into<PathBuf>) -> SippResult<Self> {
        Ok(Self {
            models: ModelStore::local(storage_root)?,
            endpoints: HashMap::new(),
            local_models: HashMap::new(),
            #[cfg(not(target_family = "wasm"))]
            io_executor: None,
        })
    }

    /// Create a client whose sandbox-owned local files are stored relative to
    /// `local_source_root` in the model registry.
    ///
    /// This constructor is reserved for platform bindings whose application
    /// container can move between launches.
    ///
    /// # Errors
    ///
    /// Returns an error when the model store cannot be opened.
    #[doc(hidden)]
    pub fn with_storage_root_and_local_source_root(
        storage_root: impl Into<PathBuf>,
        local_source_root: impl Into<PathBuf>,
    ) -> SippResult<Self> {
        Ok(Self {
            models: ModelStore::local_with_source_root(storage_root, local_source_root)?,
            endpoints: HashMap::new(),
            local_models: HashMap::new(),
            #[cfg(not(target_family = "wasm"))]
            io_executor: None,
        })
    }

    /// Access the client model store.
    pub fn models(&self) -> &ModelStore {
        &self.models
    }

    /// Register or replace a local, gateway, or direct provider endpoint.
    ///
    /// Reusing an id replaces the existing endpoint after the new descriptor
    /// has been validated and constructed.
    ///
    /// # Errors
    ///
    /// Returns an error when the id or descriptor is invalid, endpoint
    /// construction fails, or the requested endpoint feature is unavailable.
    pub async fn add(
        &mut self,
        id: impl Into<String>,
        descriptor: impl Into<EndpointDescriptor>,
    ) -> SippResult<EndpointRef> {
        match descriptor.into() {
            EndpointDescriptor::Local(descriptor) => {
                let model_id = descriptor.model_id.clone();
                let engine = self.acquire_local_engine(descriptor).await?;
                self.register_local(id, model_id, engine).await
            }
            EndpointDescriptor::Gateway(descriptor) => self.register_gateway(id, descriptor).await,
            #[cfg(feature = "providers")]
            EndpointDescriptor::Provider(descriptor) => {
                self.register_provider(id, descriptor).await
            }
        }
    }

    async fn acquire_local_engine(
        &mut self,
        descriptor: crate::client::LocalDescriptor,
    ) -> SippResult<SippEngine> {
        self.models
            .load_engine(
                &descriptor.model_id,
                ModelLoadOptions {
                    runtime: descriptor.runtime,
                    ..ModelLoadOptions::default()
                },
            )
            .await
            .map_err(SippError::from)
    }

    async fn register_local(
        &mut self,
        id: impl Into<String>,
        model_id: String,
        engine: SippEngine,
    ) -> SippResult<EndpointRef> {
        let id = normalize_id(id, "local id")?;
        let endpoint = EndpointRef::from_id(id);

        let state = engine.state().await?;
        let model = state
            .model
            .ok_or_else(|| SippError::Internal("loaded engine has no model state".to_string()))?;
        let capabilities = EndpointCapabilities::from_local(&model.capabilities);
        self.replace_endpoint(
            endpoint.clone(),
            Arc::new(LocalEndpoint::new(endpoint.clone(), capabilities, engine)),
            Some(model_id),
        )
        .await;
        Ok(endpoint)
    }

    #[cfg(not(target_family = "wasm"))]
    async fn register_gateway(
        &mut self,
        id: impl Into<String>,
        descriptor: crate::client::GatewayDescriptor,
    ) -> SippResult<EndpointRef> {
        let id = normalize_id(id, "gateway id")?;
        let endpoint = EndpointRef::from_id(id);
        let executor = self.io_executor()?;
        self.replace_endpoint(
            endpoint.clone(),
            Arc::new(GatewayEndpoint::new(
                endpoint.clone(),
                descriptor,
                executor,
            )?),
            None,
        )
        .await;
        Ok(endpoint)
    }

    #[cfg(target_family = "wasm")]
    async fn register_gateway(
        &mut self,
        id: impl Into<String>,
        _descriptor: crate::client::GatewayDescriptor,
    ) -> SippResult<EndpointRef> {
        let id = normalize_id(id, "gateway id")?;
        Err(SippError::UnsupportedOperation {
            endpoint: EndpointRef::from_id(id),
            operation: "gateway endpoint registration",
        })
    }

    #[cfg(all(feature = "providers", not(target_family = "wasm")))]
    async fn register_provider(
        &mut self,
        id: impl Into<String>,
        descriptor: ProviderDescriptor,
    ) -> SippResult<EndpointRef> {
        let id = normalize_id(id, "provider id")?;
        let endpoint = EndpointRef::from_id(id);
        let (model, transport, secrets) = descriptor.build()?;
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
            None,
        )
        .await;
        Ok(endpoint)
    }

    #[cfg(all(feature = "providers", target_family = "wasm"))]
    async fn register_provider(
        &mut self,
        id: impl Into<String>,
        _descriptor: ProviderDescriptor,
    ) -> SippResult<EndpointRef> {
        let id = normalize_id(id, "provider id")?;
        Err(SippError::UnsupportedOperation {
            endpoint: EndpointRef::from_id(id),
            operation: "provider endpoint registration",
        })
    }

    async fn replace_endpoint(
        &mut self,
        endpoint: EndpointRef,
        implementation: Arc<dyn InferenceEndpoint>,
        model_id: Option<String>,
    ) {
        let id = endpoint.id().to_string();
        let previous = self.local_models.remove(&id);
        self.models
            .replace_usage(previous.as_deref(), model_id.as_deref())
            .await;
        if let Some(model_id) = model_id {
            self.local_models.insert(id, model_id);
        }
        self.endpoints.insert(endpoint, implementation);
    }

    /// Remove a registered endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is invalid or no endpoint uses it.
    pub async fn remove(&mut self, id: &str) -> SippResult<()> {
        let id = normalize_id(id, "endpoint id")?;
        let endpoint = EndpointRef::from_id(id.clone());
        let implementation = self
            .endpoints
            .remove(&endpoint)
            .ok_or_else(|| SippError::InvalidRequest(format!("endpoint not found: {id}")))?;
        let close_result = implementation.close().await;
        let model_id = self.local_models.remove(&id);
        self.models.replace_usage(model_id.as_deref(), None).await;
        close_result
    }

    /// Submit a raw-prompt text generation request.
    pub fn query(&self, request: SippQueryRequest) -> SippTextRun {
        self.query_with_context(SippRequestContext::default(), request)
    }

    /// Submit raw-prompt generation with request-scoped correlation metadata.
    pub fn query_with_context(
        &self,
        context: SippRequestContext,
        request: SippQueryRequest,
    ) -> SippTextRun {
        match self.resolve(request.endpoint.as_ref(), "query") {
            Ok(endpoint) => endpoint.query_with_context(context, request),
            Err(error) => SippTextRun::ready_err(error),
        }
    }

    /// Submit a chat generation request.
    pub fn chat(&self, request: SippChatRequest) -> SippTextRun {
        self.chat_with_context(SippRequestContext::default(), request)
    }

    /// Submit chat generation with request-scoped correlation metadata.
    pub fn chat_with_context(
        &self,
        context: SippRequestContext,
        request: SippChatRequest,
    ) -> SippTextRun {
        match self.resolve(request.endpoint.as_ref(), "chat") {
            Ok(endpoint) => endpoint.chat_with_context(context, request),
            Err(error) => SippTextRun::ready_err(error),
        }
    }

    /// Submit a single-input embedding request.
    pub fn embed(&self, request: SippEmbedRequest) -> SippEmbeddingRun {
        self.embed_with_context(SippRequestContext::default(), request)
    }

    /// Submit an embedding request with request-scoped correlation metadata.
    pub fn embed_with_context(
        &self,
        context: SippRequestContext,
        request: SippEmbedRequest,
    ) -> SippEmbeddingRun {
        match self.resolve(request.endpoint.as_ref(), "embed") {
            Ok(endpoint) => endpoint.embed_with_context(context, request),
            Err(error) => SippEmbeddingRun::ready_err(error),
        }
    }

    fn resolve(
        &self,
        requested: Option<&EndpointRef>,
        operation: &'static str,
    ) -> SippResult<Arc<dyn InferenceEndpoint>> {
        let selected = if let Some(endpoint) = requested {
            endpoint
        } else {
            return self.resolve_single_local(operation);
        };
        let endpoint = self
            .endpoints
            .get(selected)
            .cloned()
            .ok_or_else(|| SippError::EndpointNotFound(selected.clone()))?;
        ensure_supported(endpoint.as_ref(), operation)?;
        Ok(endpoint)
    }

    fn resolve_single_local(
        &self,
        operation: &'static str,
    ) -> SippResult<Arc<dyn InferenceEndpoint>> {
        let mut matches = self
            .endpoints
            .values()
            .filter(|endpoint| self.local_models.contains_key(endpoint.endpoint().id()))
            .filter(|endpoint| {
                endpoint.capabilities().for_operation(operation) == CapabilitySupport::Supported
            });

        let Some(endpoint) = matches.next().cloned() else {
            return Err(SippError::NoSupportedEndpoint { operation });
        };
        if matches.next().is_some() {
            return Err(SippError::AmbiguousEndpoint { operation });
        }
        Ok(endpoint)
    }

    #[cfg(not(target_family = "wasm"))]
    fn io_executor(&mut self) -> SippResult<IoExecutor> {
        if let Some(executor) = &self.io_executor {
            return Ok(executor.clone());
        }

        let executor = IoExecutor::new()?;
        self.io_executor = Some(executor.clone());
        Ok(executor)
    }
}

fn ensure_supported(endpoint: &dyn InferenceEndpoint, operation: &'static str) -> SippResult<()> {
    if endpoint.capabilities().for_operation(operation) == CapabilitySupport::Unsupported {
        Err(SippError::UnsupportedOperation {
            endpoint: endpoint.endpoint().clone(),
            operation,
        })
    } else {
        Ok(())
    }
}

fn normalize_id(id: impl Into<String>, name: &'static str) -> SippResult<String> {
    let id = id.into();
    let trimmed = id.trim();
    if trimmed.is_empty() {
        Err(SippError::InvalidRequest(format!(
            "{name} must not be empty"
        )))
    } else if trimmed != id.as_str() {
        Err(SippError::InvalidRequest(format!(
            "{name} must not contain surrounding whitespace"
        )))
    } else {
        Ok(id)
    }
}
