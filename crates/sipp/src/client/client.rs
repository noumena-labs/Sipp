use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::core::{CapabilitySupport, Operation};
use crate::engine::SippEngine;
use crate::lifecycle::{ModelActivationPlan, ModelLoadOptions, ModelStore};

use crate::client::dispatch::InferenceEndpoint;
use crate::client::endpoint::{EndpointKind, Local as LocalEndpointInput};
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
    Endpoint, EndpointCapabilities, EndpointRef, SippAudioRun, SippChatRequest, SippEmbedRequest,
    SippEmbeddingRun, SippError, SippListenRequest, SippQueryRequest, SippRequestContext,
    SippResult, SippSpeakRequest, SippTextRun, DEFAULT_STORAGE_ROOT,
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

enum PreparedEndpoint {
    Local {
        activation: Box<ModelActivationPlan>,
    },
    #[cfg(not(target_family = "wasm"))]
    Constructed {
        implementation: Arc<dyn InferenceEndpoint>,
    },
}

struct ActivatedEndpoint {
    implementation: Arc<dyn InferenceEndpoint>,
    model_id: Option<String>,
}

impl PreparedEndpoint {
    async fn activate(
        self,
        models: &ModelStore,
        endpoint: &EndpointRef,
    ) -> SippResult<ActivatedEndpoint> {
        match self {
            Self::Local { activation } => {
                let activation = *activation;
                let engine = SippEngine::load(&activation.model_path, activation.runtime.clone())
                    .await
                    .map_err(SippError::from)?;
                let capabilities = match local_capabilities(&engine).await {
                    Ok(capabilities) => capabilities,
                    Err(error) => return Err(close_failed_activation(&engine, error).await),
                };
                if let Err(error) = models.commit_activation(&activation).await {
                    return Err(close_failed_activation(&engine, SippError::from(error)).await);
                }
                Ok(ActivatedEndpoint {
                    implementation: Arc::new(LocalEndpoint::new(
                        endpoint.clone(),
                        capabilities,
                        engine,
                    )),
                    model_id: Some(activation.model_id),
                })
            }
            #[cfg(not(target_family = "wasm"))]
            Self::Constructed { implementation } => Ok(ActivatedEndpoint {
                implementation,
                model_id: None,
            }),
        }
    }
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
    /// Reusing an id validates the new input, destroys the existing endpoint,
    /// and then activates and publishes the replacement.
    ///
    /// # Errors
    ///
    /// Returns an error when the id or endpoint input is invalid, endpoint
    /// construction fails, or the requested endpoint feature is unavailable.
    pub async fn add(
        &mut self,
        id: impl Into<String>,
        endpoint: impl Into<Endpoint>,
    ) -> SippResult<EndpointRef> {
        let id = normalize_id(id, "endpoint id")?;
        let endpoint_ref = EndpointRef::from_id(id);
        let prepared = self
            .prepare_endpoint(endpoint_ref.clone(), endpoint.into())
            .await?;
        self.replace_endpoint(endpoint_ref.clone(), prepared)
            .await?;
        Ok(endpoint_ref)
    }

    async fn prepare_endpoint(
        &mut self,
        endpoint: EndpointRef,
        input: Endpoint,
    ) -> SippResult<PreparedEndpoint> {
        match input.kind {
            EndpointKind::Local(local) => {
                let LocalEndpointInput { model_id, runtime } = *local;
                let activation = self
                    .models
                    .prepare_activation(
                        &model_id,
                        ModelLoadOptions {
                            runtime,
                            ..ModelLoadOptions::default()
                        },
                    )
                    .await
                    .map_err(SippError::from)?;
                Ok(PreparedEndpoint::Local {
                    activation: Box::new(activation),
                })
            }
            EndpointKind::Gateway(gateway) => self.prepare_gateway(endpoint, *gateway),
            #[cfg(feature = "providers")]
            EndpointKind::Provider(provider) => self.prepare_provider(endpoint, *provider),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn prepare_gateway(
        &mut self,
        endpoint: EndpointRef,
        descriptor: crate::client::GatewayDescriptor,
    ) -> SippResult<PreparedEndpoint> {
        let executor = self.io_executor()?;
        Ok(PreparedEndpoint::Constructed {
            implementation: Arc::new(GatewayEndpoint::new(
                endpoint.clone(),
                descriptor,
                executor,
            )?),
        })
    }

    #[cfg(target_family = "wasm")]
    fn prepare_gateway(
        &mut self,
        endpoint: EndpointRef,
        _descriptor: crate::client::GatewayDescriptor,
    ) -> SippResult<PreparedEndpoint> {
        Err(SippError::UnsupportedOperation {
            endpoint,
            operation: "gateway endpoint registration",
        })
    }

    #[cfg(all(feature = "providers", not(target_family = "wasm")))]
    fn prepare_provider(
        &mut self,
        endpoint: EndpointRef,
        descriptor: ProviderDescriptor,
    ) -> SippResult<PreparedEndpoint> {
        let (model, transport, secrets) = descriptor.build()?;
        let executor = self.io_executor()?;
        Ok(PreparedEndpoint::Constructed {
            implementation: Arc::new(ProviderEndpoint::new(
                endpoint.clone(),
                model,
                EndpointCapabilities::remote_text(),
                transport,
                executor,
                secrets,
            )),
        })
    }

    #[cfg(all(feature = "providers", target_family = "wasm"))]
    fn prepare_provider(
        &mut self,
        endpoint: EndpointRef,
        _descriptor: ProviderDescriptor,
    ) -> SippResult<PreparedEndpoint> {
        Err(SippError::UnsupportedOperation {
            endpoint,
            operation: "provider endpoint registration",
        })
    }

    async fn replace_endpoint(
        &mut self,
        endpoint: EndpointRef,
        prepared: PreparedEndpoint,
    ) -> SippResult<()> {
        self.retire_endpoint(&endpoint).await?;
        let activated = prepared.activate(&self.models, &endpoint).await?;
        let id = endpoint.id().to_string();
        if let Some(model_id) = activated.model_id {
            self.models.mark_used(&model_id).await;
            self.local_models.insert(id, model_id);
        }
        self.endpoints.insert(endpoint, activated.implementation);
        Ok(())
    }

    async fn retire_endpoint(&mut self, endpoint: &EndpointRef) -> SippResult<()> {
        let Some(implementation) = self.endpoints.remove(endpoint) else {
            return Ok(());
        };
        if let Some(model_id) = self.local_models.remove(endpoint.id()) {
            self.models.mark_unused(&model_id).await;
        }
        implementation.close().await
    }

    /// Remove a registered endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is invalid or no endpoint uses it.
    pub async fn remove(&mut self, id: &str) -> SippResult<()> {
        let id = normalize_id(id, "endpoint id")?;
        let endpoint = EndpointRef::from_id(id.clone());
        if !self.endpoints.contains_key(&endpoint) {
            return Err(SippError::InvalidRequest(format!(
                "endpoint not found: {id}"
            )));
        }
        self.retire_endpoint(&endpoint).await
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
        match self.resolve(request.endpoint.as_ref(), Operation::Query) {
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
        match self.resolve(request.endpoint.as_ref(), Operation::Chat) {
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
        match self.resolve(request.endpoint.as_ref(), Operation::Embed) {
            Ok(endpoint) => endpoint.embed_with_context(context, request),
            Err(error) => SippEmbeddingRun::ready_err(error),
        }
    }

    /// Submit an encoded-audio transcription request.
    pub fn listen(&self, request: impl Into<SippListenRequest>) -> SippTextRun {
        self.listen_with_context(SippRequestContext::default(), request.into())
    }

    /// Submit encoded-audio transcription with request-scoped correlation metadata.
    pub fn listen_with_context(
        &self,
        context: SippRequestContext,
        request: SippListenRequest,
    ) -> SippTextRun {
        if let Err(error) = crate::client::validate::listen(&request) {
            return SippTextRun::ready_err(error);
        }
        match self.resolve(request.endpoint.as_ref(), Operation::Listen) {
            Ok(endpoint) => endpoint.listen_with_context(context, request),
            Err(error) => SippTextRun::ready_err(error),
        }
    }

    /// Submit a text-to-WAV synthesis request.
    pub fn speak(&self, request: impl Into<SippSpeakRequest>) -> SippAudioRun {
        self.speak_with_context(SippRequestContext::default(), request.into())
    }

    /// Submit text-to-WAV synthesis with request-scoped correlation metadata.
    pub fn speak_with_context(
        &self,
        context: SippRequestContext,
        request: SippSpeakRequest,
    ) -> SippAudioRun {
        if let Err(error) = crate::client::validate::speak(&request) {
            return SippAudioRun::ready_err(error);
        }
        match self.resolve(request.endpoint.as_ref(), Operation::Speak) {
            Ok(endpoint) => endpoint.speak_with_context(context, request),
            Err(error) => SippAudioRun::ready_err(error),
        }
    }

    fn resolve(
        &self,
        requested: Option<&EndpointRef>,
        operation: Operation,
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

    fn resolve_single_local(&self, operation: Operation) -> SippResult<Arc<dyn InferenceEndpoint>> {
        let mut matches = self
            .endpoints
            .values()
            .filter(|endpoint| self.local_models.contains_key(endpoint.endpoint().id()))
            .filter(|endpoint| {
                endpoint.capabilities().for_operation(operation) == CapabilitySupport::Supported
            });

        let Some(endpoint) = matches.next().cloned() else {
            return Err(SippError::NoSupportedEndpoint {
                operation: operation.as_str(),
            });
        };
        if matches.next().is_some() {
            return Err(SippError::AmbiguousEndpoint {
                operation: operation.as_str(),
            });
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

async fn local_capabilities(engine: &SippEngine) -> SippResult<EndpointCapabilities> {
    let state = engine.state().await?;
    let model = state
        .model
        .ok_or_else(|| SippError::Internal("loaded engine has no model state".to_string()))?;
    Ok(EndpointCapabilities::from_local(&model.capabilities))
}

/// Closes an engine whose activation failed and returns the activation error.
///
/// The activation error is returned with its original variant so callers keep
/// matching on its classification. A failure to close the unusable engine
/// is best effort and must not hide that primary error inside `Internal`.
async fn close_failed_activation(engine: &SippEngine, error: SippError) -> SippError {
    let _ = engine.close().await;
    error
}

fn ensure_supported(endpoint: &dyn InferenceEndpoint, operation: Operation) -> SippResult<()> {
    if endpoint.capabilities().for_operation(operation) == CapabilitySupport::Unsupported {
        Err(SippError::UnsupportedOperation {
            endpoint: endpoint.endpoint().clone(),
            operation: operation.as_str(),
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
