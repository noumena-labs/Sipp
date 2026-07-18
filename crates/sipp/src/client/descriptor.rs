use std::path::PathBuf;

use crate::engine::NativeRuntimeConfig;
use crate::lifecycle::ModelSource;

use crate::client::GatewayDescriptor;
#[cfg(feature = "providers")]
use crate::client::ProviderDescriptor;

/// Default native registry and managed-asset root for local endpoints.
pub const DEFAULT_STORAGE_ROOT: &str = ".sipp-models";

/// Descriptor used by `SippClient::add` to register an endpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointDescriptor {
    /// Local GGUF model loaded into this process.
    Local(LocalDescriptor),
    /// First-party HTTP gateway endpoint.
    Gateway(GatewayDescriptor),
    /// Direct provider endpoint using caller-owned credentials.
    #[cfg(feature = "providers")]
    Provider(ProviderDescriptor),
}

impl From<LocalDescriptor> for EndpointDescriptor {
    fn from(descriptor: LocalDescriptor) -> Self {
        Self::Local(descriptor)
    }
}

impl From<GatewayDescriptor> for EndpointDescriptor {
    fn from(descriptor: GatewayDescriptor) -> Self {
        Self::Gateway(descriptor)
    }
}

#[cfg(feature = "providers")]
impl From<ProviderDescriptor> for EndpointDescriptor {
    fn from(descriptor: ProviderDescriptor) -> Self {
        Self::Provider(descriptor)
    }
}

/// Descriptor for a local GGUF model endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalDescriptor {
    pub(crate) source: ModelSource,
    /// Lifecycle registry and asset-store root.
    pub storage_root: PathBuf,
    /// Native runtime configuration.
    pub config: NativeRuntimeConfig,
}

impl LocalDescriptor {
    /// Create a local endpoint from host filesystem model files.
    pub fn files<P, I>(model_paths: I) -> Self
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = P>,
    {
        Self::from_source(ModelSource::Local {
            model_paths: model_paths.into_iter().map(Into::into).collect(),
            projector_path: None,
        })
    }

    /// Create a multimodal local endpoint from host filesystem files.
    pub fn files_with_projector<P, I>(model_paths: I, projector_path: impl Into<PathBuf>) -> Self
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = P>,
    {
        Self::from_source(ModelSource::Local {
            model_paths: model_paths.into_iter().map(Into::into).collect(),
            projector_path: Some(projector_path.into()),
        })
    }

    /// Create a local endpoint that acquires model files from HTTP(S) URLs.
    pub fn urls<U, I>(model_urls: I) -> Self
    where
        U: Into<String>,
        I: IntoIterator<Item = U>,
    {
        Self::from_source(ModelSource::Remote {
            model_urls: model_urls.into_iter().map(Into::into).collect(),
            projector_url: None,
        })
    }

    /// Create a multimodal local endpoint from HTTP(S) model URLs.
    pub fn urls_with_projector<U, I>(model_urls: I, projector_url: impl Into<String>) -> Self
    where
        U: Into<String>,
        I: IntoIterator<Item = U>,
    {
        Self::from_source(ModelSource::Remote {
            model_urls: model_urls.into_iter().map(Into::into).collect(),
            projector_url: Some(projector_url.into()),
        })
    }

    fn from_source(source: ModelSource) -> Self {
        Self {
            source,
            storage_root: DEFAULT_STORAGE_ROOT.into(),
            config: NativeRuntimeConfig::default(),
        }
    }
}
