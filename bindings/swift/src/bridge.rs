//! UniFFI projection for model and endpoint lifecycle operations.

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use sipp::lifecycle::{
    ManagedModel as CoreManagedModel, ModelError as CoreModelError,
    ModelModality as CoreModelModality, ModelStatus as CoreModelStatus,
};
use sipp::{
    endpoint::Local as CoreLocalEndpoint, Endpoint as CoreEndpoint,
    SippCancellationReason as CoreCancellationReason, SippClient as CoreSippClient,
    SippError as CoreSippError,
};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::inference::{
    FfiAudioRun, FfiChatRequest, FfiEmbedRequest, FfiEmbeddingRun, FfiListenRequest,
    FfiQueryRequest, FfiSpeakRequest, FfiTextRun,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/bridge_tests.rs"]
mod bridge_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Model modality carried across the internal Swift FFI boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiModelModality {
    /// Text-only inference model.
    Text,
    /// Vision-language model.
    Vision,
    /// Audio understanding or speech recognition model.
    Audio,
    /// Model supporting multiple media modalities.
    Multimodal,
}

impl From<CoreModelModality> for FfiModelModality {
    fn from(value: CoreModelModality) -> Self {
        match value {
            CoreModelModality::Text => Self::Text,
            CoreModelModality::Vision => Self::Vision,
            CoreModelModality::Audio => Self::Audio,
            CoreModelModality::Multimodal => Self::Multimodal,
        }
    }
}

/// Model installation state carried across the internal Swift FFI boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiModelStatus {
    /// Model is ready to load.
    Ready,
    /// Vision model requires a compatible projector.
    NeedsProjector,
    /// Model registration is incomplete or corrupt.
    Broken,
}

impl From<CoreModelStatus> for FfiModelStatus {
    fn from(value: CoreModelStatus) -> Self {
        match value {
            CoreModelStatus::Ready => Self::Ready,
            CoreModelStatus::NeedsProjector => Self::NeedsProjector,
            CoreModelStatus::Broken => Self::Broken,
        }
    }
}

/// Managed model value returned by the internal Swift FFI boundary.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiManagedModel {
    /// Stable model identifier.
    pub id: String,
    /// Display name derived from model metadata or the source filename.
    pub name: String,
    /// Combined model and projector byte count.
    pub bytes: u64,
    /// Detected inference modality.
    pub modality: FfiModelModality,
    /// Current installation state.
    pub status: FfiModelStatus,
}

impl From<CoreManagedModel> for FfiManagedModel {
    fn from(value: CoreManagedModel) -> Self {
        Self {
            id: value.id,
            name: value.name,
            bytes: value.bytes,
            modality: value.modality.into(),
            status: value.status.into(),
        }
    }
}

impl From<FfiManagedModel> for CoreManagedModel {
    fn from(value: FfiManagedModel) -> Self {
        Self {
            id: value.id,
            name: value.name,
            bytes: value.bytes,
            modality: match value.modality {
                FfiModelModality::Text => CoreModelModality::Text,
                FfiModelModality::Vision => CoreModelModality::Vision,
                FfiModelModality::Audio => CoreModelModality::Audio,
                FfiModelModality::Multimodal => CoreModelModality::Multimodal,
            },
            status: match value.status {
                FfiModelStatus::Ready => CoreModelStatus::Ready,
                FfiModelStatus::NeedsProjector => CoreModelStatus::NeedsProjector,
                FfiModelStatus::Broken => CoreModelStatus::Broken,
            },
        }
    }
}

/// Unregistered local endpoint input carried across the Swift FFI boundary.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiEndpoint {
    /// Managed model selected for local inference.
    pub model: FfiManagedModel,
}

impl From<FfiEndpoint> for CoreEndpoint {
    fn from(value: FfiEndpoint) -> Self {
        let model = CoreManagedModel::from(value.model);
        CoreLocalEndpoint::new(&model).into()
    }
}

/// Registered endpoint identity carried across the Swift FFI boundary.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiEndpointRef {
    /// Stable client-scoped endpoint identifier.
    pub id: String,
}

/// Result of a model registration used by Swift's bookmark transaction.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiModelRegistration {
    /// Registered model value.
    pub model: FfiManagedModel,
    /// Whether this call created a model rather than replacing the same id.
    pub created: bool,
}

/// Stable cancellation reason carried by typed Swift bridge errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiCancellationReason {
    /// The application explicitly cancelled the request.
    CallerCancelled,
    /// The downstream client disconnected.
    ClientDisconnected,
    /// The hosting process is shutting down.
    ServerShutdown,
    /// The request exceeded an application deadline.
    DeadlineExceeded,
}

impl From<CoreCancellationReason> for FfiCancellationReason {
    fn from(value: CoreCancellationReason) -> Self {
        match value {
            CoreCancellationReason::CallerCancelled => Self::CallerCancelled,
            CoreCancellationReason::ClientDisconnected => Self::ClientDisconnected,
            CoreCancellationReason::ServerShutdown => Self::ServerShutdown,
            CoreCancellationReason::DeadlineExceeded => Self::DeadlineExceeded,
        }
    }
}

/// Typed failure returned by the internal Swift FFI boundary.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    /// Model registration, storage, or lifecycle failure.
    #[error("{message}")]
    ModelLifecycle {
        /// Stable lifecycle error code.
        code: String,
        /// Human-readable failure description.
        message: String,
        /// HTTP status for remote acquisition failures.
        status: Option<u16>,
        /// Server-requested retry delay in milliseconds.
        retry_after_ms: Option<u64>,
    },
    /// Invalid caller input.
    #[error("{message}")]
    InvalidArgument {
        /// Human-readable validation failure.
        message: String,
    },
    /// Operation unavailable for the selected local endpoint or runtime.
    #[error("{message}")]
    UnsupportedOperation {
        /// Human-readable operation failure.
        message: String,
    },
    /// Structured failure returned by an inference endpoint.
    #[error("{message}")]
    Endpoint {
        /// Endpoint-defined stable classification.
        kind: String,
        /// Transport status code when supplied by the endpoint.
        status: Option<u16>,
        /// Endpoint-specific error code when supplied.
        code: Option<String>,
        /// Human-readable endpoint failure.
        message: String,
        /// Upstream request identifier when supplied.
        request_id: Option<String>,
    },
    /// Endpoint selection failed before inference began.
    #[error("{message}")]
    EndpointSelection {
        /// Human-readable endpoint selection failure.
        message: String,
    },
    /// An in-flight run was cancelled explicitly.
    #[error("request cancelled")]
    Cancelled {
        /// Stable cancellation classification.
        reason: FfiCancellationReason,
    },
    /// A one-shot internal run value was requested more than once.
    #[error("{message}")]
    InvalidState {
        /// Human-readable run-state failure.
        message: String,
    },
    /// Native client or bridge runtime failure.
    #[error("{message}")]
    Runtime {
        /// Human-readable runtime failure.
        message: String,
    },
}

impl From<CoreModelError> for FfiError {
    fn from(error: CoreModelError) -> Self {
        Self::ModelLifecycle {
            code: error.code().to_owned(),
            message: error.to_string(),
            status: error.status(),
            retry_after_ms: error.retry_after_ms(),
        }
    }
}

impl From<CoreSippError> for FfiError {
    fn from(error: CoreSippError) -> Self {
        match error {
            CoreSippError::ModelLifecycle(error) => error.into(),
            CoreSippError::InvalidRequest(message) => Self::InvalidArgument { message },
            CoreSippError::Local(error) => match error {
                sipp::error::Error::InvalidRequest(message)
                | sipp::error::Error::InvalidConfig(message) => Self::InvalidArgument {
                    message: message.to_owned(),
                },
                error @ sipp::error::Error::UnsupportedOperation { .. } => {
                    Self::UnsupportedOperation {
                        message: error.to_string(),
                    }
                }
                error @ (sipp::error::Error::InteriorNul(_)
                | sipp::error::Error::ModelLoad { .. }
                | sipp::error::Error::ContextInit
                | sipp::error::Error::NullPointer(_)
                | sipp::error::Error::Tokenize
                | sipp::error::Error::TokenToPiece { .. }
                | sipp::error::Error::Decode(_)
                | sipp::error::Error::BatchCapacity { .. }
                | sipp::error::Error::PromptTooLong { .. }
                | sipp::error::Error::SamplerInit
                | sipp::error::Error::RuntimeNotReady
                | sipp::error::Error::RuntimeCommand(_)) => Self::Runtime {
                    message: error.to_string(),
                },
            },
            CoreSippError::Endpoint(error) => {
                let message = error.to_string();
                Self::Endpoint {
                    kind: error.kind,
                    status: error.status,
                    code: error.code,
                    message,
                    request_id: error.request_id,
                }
            }
            CoreSippError::Provider(error) => {
                let message = error.to_string();
                Self::Endpoint {
                    kind: error.kind.as_str().to_owned(),
                    status: error.status,
                    code: error.code,
                    message,
                    request_id: error.request_id,
                }
            }
            CoreSippError::Cancelled { reason } => Self::Cancelled {
                reason: reason.into(),
            },
            error @ CoreSippError::UnsupportedOperation { .. } => Self::UnsupportedOperation {
                message: error.to_string(),
            },
            error @ (CoreSippError::EndpointNotFound(_)
            | CoreSippError::AmbiguousEndpoint { .. }
            | CoreSippError::NoSupportedEndpoint { .. }) => Self::EndpointSelection {
                message: error.to_string(),
            },
            error @ CoreSippError::Internal(_) => Self::Runtime {
                message: error.to_string(),
            },
        }
    }
}

pub(crate) struct BridgeExecutor {
    runtime: Runtime,
}

struct BridgeTask<T> {
    handle: JoinHandle<T>,
}

impl<T> Drop for BridgeTask<T> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl BridgeExecutor {
    pub(crate) fn new() -> Result<Self, FfiError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("sipp-swift-runtime")
            .enable_all()
            .build()
            .map_err(|error| FfiError::Runtime {
                message: format!("failed to start the Swift bridge runtime: {error}"),
            })?;
        Ok(Self { runtime })
    }

    pub(crate) async fn execute<T, F>(&self, future: F) -> Result<T, FfiError>
    where
        T: Send + 'static,
        F: Future<Output = Result<T, FfiError>> + Send + 'static,
    {
        let mut task = BridgeTask {
            handle: self.runtime.spawn(future),
        };
        (&mut task.handle)
            .await
            .map_err(|error| FfiError::Runtime {
                message: format!("Swift bridge task failed: {error}"),
            })?
    }
}

type SharedClient = Arc<RwLock<CoreSippClient>>;

/// Internal model-store object exposed to generated Swift bindings.
#[derive(uniffi::Object)]
pub struct FfiModelStore {
    client: SharedClient,
    executor: Arc<BridgeExecutor>,
}

#[uniffi::export]
impl FfiModelStore {
    /// Register local paths or HTTP(S) model sources.
    ///
    /// # Errors
    ///
    /// Returns a typed lifecycle or runtime error when registration fails.
    pub async fn add(&self, sources: Vec<String>) -> Result<FfiModelRegistration, FfiError> {
        let client = Arc::clone(&self.client);
        self.executor
            .execute(async move {
                let client = client.read().await;
                let registration = client
                    .models()
                    .add_with_outcome(sources)
                    .await
                    .map_err(FfiError::from)?;
                Ok(FfiModelRegistration {
                    model: FfiManagedModel::from(registration.model),
                    created: registration.created,
                })
            })
            .await
    }

    /// List registered models.
    ///
    /// # Errors
    ///
    /// Returns a typed lifecycle or runtime error when the registry cannot be read.
    pub async fn list(&self) -> Result<Vec<FfiManagedModel>, FfiError> {
        self.client
            .read()
            .await
            .models()
            .list()
            .await
            .map(|models| models.into_iter().map(FfiManagedModel::from).collect())
            .map_err(FfiError::from)
    }

    /// Remove a registered model that no endpoint currently uses.
    ///
    /// # Errors
    ///
    /// Returns a typed lifecycle or runtime error when removal fails.
    pub async fn remove(&self, model_id: String) -> Result<(), FfiError> {
        self.client
            .read()
            .await
            .models()
            .remove(&model_id)
            .await
            .map_err(FfiError::from)
    }
}

/// Internal client object exposed to generated Swift bindings.
#[derive(uniffi::Object)]
pub struct FfiSippClient {
    client: SharedClient,
    models: Arc<FfiModelStore>,
}

#[uniffi::export]
impl FfiSippClient {
    /// Open a client at an explicit native storage path.
    ///
    /// # Errors
    ///
    /// Returns a typed lifecycle or runtime error when the client cannot open.
    #[uniffi::constructor]
    pub fn new(
        storage_root: String,
        local_source_root: Option<String>,
    ) -> Result<Arc<Self>, FfiError> {
        let storage_root = PathBuf::from(storage_root);
        let client = match local_source_root {
            Some(local_source_root) => CoreSippClient::with_storage_root_and_local_source_root(
                storage_root,
                PathBuf::from(local_source_root),
            ),
            None => CoreSippClient::with_storage_root(storage_root),
        }
        .map_err(FfiError::from)?;
        let client = Arc::new(RwLock::new(client));
        let executor = Arc::new(BridgeExecutor::new()?);
        let models = Arc::new(FfiModelStore {
            client: Arc::clone(&client),
            executor,
        });
        Ok(Arc::new(Self { client, models }))
    }

    /// Return the client-owned model store.
    pub fn models(&self) -> Arc<FfiModelStore> {
        Arc::clone(&self.models)
    }

    /// Register or replace an endpoint.
    ///
    /// # Errors
    ///
    /// Returns a typed request, lifecycle, unsupported-operation, or runtime error.
    pub async fn add(&self, id: String, endpoint: FfiEndpoint) -> Result<FfiEndpointRef, FfiError> {
        self.client
            .write()
            .await
            .add(id, CoreEndpoint::from(endpoint))
            .await
            .map(|endpoint| FfiEndpointRef {
                id: endpoint.id().to_string(),
            })
            .map_err(FfiError::from)
    }

    /// Remove a registered endpoint.
    ///
    /// # Errors
    ///
    /// Returns a typed request or runtime error when removal fails.
    pub async fn remove(&self, id: String) -> Result<(), FfiError> {
        self.client
            .write()
            .await
            .remove(&id)
            .await
            .map_err(FfiError::from)
    }

    /// Start raw-prompt text generation on a registered endpoint.
    pub async fn query(&self, request: FfiQueryRequest) -> Arc<FfiTextRun> {
        let (context, request) = request.into_core();
        let run = self
            .client
            .read()
            .await
            .query_with_context(context, request);
        Arc::new(FfiTextRun::from_core(run))
    }

    /// Start chat generation on a registered endpoint.
    pub async fn chat(&self, request: FfiChatRequest) -> Arc<FfiTextRun> {
        let (context, request) = request.into_core();
        let run = self.client.read().await.chat_with_context(context, request);
        Arc::new(FfiTextRun::from_core(run))
    }

    /// Start single-input embedding on a registered endpoint.
    pub async fn embed(&self, request: FfiEmbedRequest) -> Arc<FfiEmbeddingRun> {
        let (context, request) = request.into_core();
        let run = self
            .client
            .read()
            .await
            .embed_with_context(context, request);
        Arc::new(FfiEmbeddingRun::from_core(run))
    }

    /// Start speech recognition on a registered endpoint.
    pub async fn listen(&self, request: FfiListenRequest) -> Arc<FfiTextRun> {
        let (context, request) = request.into_core();
        let run = self
            .client
            .read()
            .await
            .listen_with_context(context, request);
        Arc::new(FfiTextRun::from_core(run))
    }

    /// Start speech synthesis on a registered endpoint.
    pub async fn speak(&self, request: FfiSpeakRequest) -> Arc<FfiAudioRun> {
        let (context, request) = request.into_core();
        let run = self
            .client
            .read()
            .await
            .speak_with_context(context, request);
        Arc::new(FfiAudioRun::from_core(run))
    }
}
