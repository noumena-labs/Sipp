use serde::{Deserialize, Serialize};
use url::Url;

use crate::lifecycle::{ModelAssetKind, ModelError};

/// Role of one independently acquired remote asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RemoteAssetRole {
    Model,
    Shard,
    Projector,
}

impl RemoteAssetRole {
    pub(crate) const fn asset_kind(self) -> ModelAssetKind {
        match self {
            Self::Model => ModelAssetKind::Model,
            Self::Shard => ModelAssetKind::Shard,
            Self::Projector => ModelAssetKind::Projector,
        }
    }
}

/// Remote metadata used as the exact cache identity for one asset member.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteMetadata {
    pub(crate) url: String,
    pub(crate) name: String,
    pub(crate) bytes: u64,
    pub(crate) etag: Option<String>,
    pub(crate) last_modified: Option<String>,
}

/// Raw response headers reported by a platform HTTP executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteMetadataHeaders {
    pub(crate) content_length: Option<u64>,
    pub(crate) linked_size: Option<u64>,
    pub(crate) etag: Option<String>,
    pub(crate) linked_etag: Option<String>,
    pub(crate) last_modified: Option<String>,
}

/// Exact cached representation known before remote revalidation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteCacheCandidate {
    pub(crate) candidate_id: String,
    pub(crate) asset_ids: Vec<String>,
    pub(crate) metadata: RemoteMetadata,
}

/// One independently acquired URL and its exact cached representations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteAcquisitionRequest {
    pub(crate) member_id: u32,
    pub(crate) role: RemoteAssetRole,
    pub(crate) url: String,
    pub(crate) candidates: Vec<RemoteCacheCandidate>,
}

/// Remote operation phase used for retry and typed error classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RemoteFailurePhase {
    Metadata,
    Download,
    CacheValidation,
    Cleanup,
}

/// Raw failure category reported by a platform executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RemoteFailureKind {
    Transport,
    Http,
    InvalidResponse,
    Integrity,
    Storage,
}

/// Structured remote operation failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFailure {
    pub(crate) phase: RemoteFailurePhase,
    pub(crate) kind: RemoteFailureKind,
    pub(crate) status: Option<u16>,
    pub(crate) retry_after: Option<String>,
    pub(crate) reason: String,
}

impl RemoteFailure {
    pub(crate) fn model_error(self, url: &str) -> ModelError {
        let url = redacted_remote_url(url);
        let retry_after_ms = retry_after_ms(self.retry_after.as_deref());
        match self.phase {
            RemoteFailurePhase::Metadata => ModelError::RemoteMetadataUnavailable {
                url,
                status: self.status,
                retry_after_ms,
                reason: self.reason,
            },
            RemoteFailurePhase::Download => ModelError::RemoteDownloadFailed {
                url,
                status: self.status,
                retry_after_ms,
                reason: self.reason,
            },
            RemoteFailurePhase::CacheValidation => ModelError::RemoteIntegrityFailed {
                url,
                reason: self.reason,
            },
            RemoteFailurePhase::Cleanup => ModelError::RemoteCleanupFailed {
                url,
                reason: self.reason,
            },
        }
    }
}

pub(super) fn retry_after_ms(value: Option<&str>) -> Option<u64> {
    let value = value?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return seconds.checked_mul(1_000);
    }
    let retry_at = httpdate::parse_http_date(value).ok()?;
    let delay = retry_at.duration_since(std::time::SystemTime::now()).ok()?;
    u64::try_from(delay.as_millis()).ok()
}

/// Host operation selected by the Rust acquisition state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum RemoteAction {
    FetchMetadata {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        url: String,
    },
    Wait {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        delay_ms: u64,
    },
    ValidateCache {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        candidate: RemoteCacheCandidate,
    },
    Download {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        role: RemoteAssetRole,
        metadata: RemoteMetadata,
    },
    Cleanup {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        asset_ids: Vec<String>,
    },
}

impl RemoteAction {
    pub(crate) fn identity(&self) -> (&str, u32, u8) {
        match self {
            Self::FetchMetadata {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::Wait {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::ValidateCache {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::Download {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::Cleanup {
                acquisition_id,
                member_id,
                attempt,
                ..
            } => (acquisition_id, *member_id, *attempt),
        }
    }
}

/// Result of one host operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum RemoteAcquisitionEvent {
    MetadataSucceeded {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        headers: RemoteMetadataHeaders,
    },
    OperationFailed {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        failure: RemoteFailure,
    },
    WaitCompleted {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
    },
    CacheValidated {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        asset_ids: Vec<String>,
    },
    DownloadSucceeded {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        asset_ids: Vec<String>,
        created_asset_ids: Vec<String>,
    },
    CleanupSucceeded {
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
    },
    Cancelled {
        acquisition_id: String,
    },
}

impl RemoteAcquisitionEvent {
    pub(crate) fn identity(&self) -> (&str, Option<(u32, u8)>) {
        match self {
            Self::MetadataSucceeded {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::OperationFailed {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::WaitCompleted {
                acquisition_id,
                member_id,
                attempt,
            }
            | Self::CacheValidated {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::DownloadSucceeded {
                acquisition_id,
                member_id,
                attempt,
                ..
            }
            | Self::CleanupSucceeded {
                acquisition_id,
                member_id,
                attempt,
            } => (acquisition_id, Some((*member_id, *attempt))),
            Self::Cancelled { acquisition_id } => (acquisition_id, None),
        }
    }
}

/// Resolved assets for one remote source member.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteResolvedMember {
    pub(crate) member_id: u32,
    pub(crate) role: RemoteAssetRole,
    pub(crate) asset_ids: Vec<String>,
    pub(crate) created_asset_ids: Vec<String>,
}

/// Current externally visible state of an acquisition.
#[derive(Debug)]
pub(crate) enum RemoteAcquisitionProgress {
    Action(RemoteAction),
    Ready(Vec<RemoteResolvedMember>),
    Failed(ModelError),
    Cancelled,
}

pub(crate) fn canonical_remote_url(value: &str) -> Result<String, ModelError> {
    let parsed = Url::parse(value).map_err(|error| {
        ModelError::InvalidModelSource(format!("remote URL is invalid: {error}"))
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(ModelError::InvalidModelSource(
            "remote URL scheme must be http or https".to_string(),
        ));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(ModelError::InvalidModelSource(
            "remote URL must not contain credentials".to_string(),
        ));
    }
    Ok(parsed.to_string())
}

pub(crate) fn redacted_remote_url(value: &str) -> String {
    Url::parse(value).map_or_else(
        |_| "<invalid-remote-url>".to_string(),
        |mut url| {
            url.set_query(None);
            url.set_fragment(None);
            url.to_string()
        },
    )
}
