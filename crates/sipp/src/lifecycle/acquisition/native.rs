//! Native HTTP and filesystem executor for shared acquisition actions.

use std::collections::BTreeMap;

use reqwest::header::HeaderMap;
use tokio::io::AsyncWriteExt;

use crate::lifecycle::storage::{AcquisitionJournal, StorageBackend};
use crate::lifecycle::{AssetRecord, AssetStore, ModelError, RegistryManifest};

use super::{
    RemoteAcquisitionEvent, RemoteAction, RemoteFailure, RemoteFailureKind, RemoteFailurePhase,
    RemoteMetadataHeaders,
};

const X_LINKED_ETAG: &str = "x-linked-etag";
const X_LINKED_SIZE: &str = "x-linked-size";

pub(crate) struct NativeRemoteExecutor<B: StorageBackend> {
    client: reqwest::Client,
    assets: AssetStore<B>,
    journal: AcquisitionJournal<B>,
}

impl<B: StorageBackend> NativeRemoteExecutor<B> {
    pub(crate) fn new(
        assets: AssetStore<B>,
        journal: AcquisitionJournal<B>,
    ) -> Result<Self, ModelError> {
        let redirect = reqwest::redirect::Policy::custom(|attempt| {
            let downgrade = attempt.previous().last().is_some_and(|previous| {
                previous.scheme() == "https" && attempt.url().scheme() == "http"
            });
            if downgrade || attempt.previous().len() >= 10 {
                attempt.stop()
            } else {
                attempt.follow()
            }
        });
        let client = reqwest::Client::builder()
            .redirect(redirect)
            .build()
            .map_err(|error| ModelError::RemoteClient(error.to_string()))?;
        Ok(Self {
            client,
            assets,
            journal,
        })
    }

    pub(crate) async fn execute(
        &self,
        action: RemoteAction,
        manifest: &RegistryManifest,
        downloaded: &mut BTreeMap<String, AssetRecord>,
    ) -> RemoteAcquisitionEvent {
        match action {
            RemoteAction::FetchMetadata {
                acquisition_id,
                member_id,
                attempt,
                url,
            } => {
                self.fetch_metadata(acquisition_id, member_id, attempt, url)
                    .await
            }
            RemoteAction::Wait {
                acquisition_id,
                member_id,
                attempt,
                delay_ms,
            } => {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                RemoteAcquisitionEvent::WaitCompleted {
                    acquisition_id,
                    member_id,
                    attempt,
                }
            }
            RemoteAction::ValidateCache {
                acquisition_id,
                member_id,
                attempt,
                candidate,
            } => self.validate_cache(
                acquisition_id,
                member_id,
                attempt,
                candidate.asset_ids,
                manifest,
            ),
            RemoteAction::Download {
                acquisition_id,
                member_id,
                attempt,
                role,
                metadata,
            } => {
                match self
                    .download(
                        acquisition_id,
                        member_id,
                        attempt,
                        role.asset_kind(),
                        metadata,
                    )
                    .await
                {
                    NativeDownloadResult::Succeeded { event, record } => {
                        downloaded.insert(record.id.clone(), record);
                        event
                    }
                    NativeDownloadResult::Failed(event) => event,
                }
            }
            RemoteAction::Cleanup {
                acquisition_id,
                member_id,
                attempt,
                asset_ids,
            } => self.cleanup(acquisition_id, member_id, attempt, asset_ids, downloaded),
        }
    }

    async fn fetch_metadata(
        &self,
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        url: String,
    ) -> RemoteAcquisitionEvent {
        let response = match self.client.head(&url).send().await {
            Ok(response) => response,
            Err(_) => {
                return operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    transport_failure(RemoteFailurePhase::Metadata),
                );
            }
        };
        if !response.status().is_success() {
            return operation_failed(
                acquisition_id,
                member_id,
                attempt,
                http_failure(RemoteFailurePhase::Metadata, &response),
            );
        }
        RemoteAcquisitionEvent::MetadataSucceeded {
            acquisition_id,
            member_id,
            attempt,
            headers: metadata_headers(response.headers()),
        }
    }

    fn validate_cache(
        &self,
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        asset_ids: Vec<String>,
        manifest: &RegistryManifest,
    ) -> RemoteAcquisitionEvent {
        for asset_id in &asset_ids {
            let Some(record) = manifest.assets.get(asset_id) else {
                return operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    integrity_failure(
                        RemoteFailurePhase::CacheValidation,
                        "selected cache asset is absent from the registry",
                    ),
                );
            };
            if self.assets.validate_asset(record).is_err() {
                return operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    integrity_failure(
                        RemoteFailurePhase::CacheValidation,
                        "selected cache asset failed content validation",
                    ),
                );
            }
        }
        RemoteAcquisitionEvent::CacheValidated {
            acquisition_id,
            member_id,
            attempt,
            asset_ids,
        }
    }

    async fn download(
        &self,
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        kind: crate::lifecycle::ModelAssetKind,
        metadata: super::RemoteMetadata,
    ) -> NativeDownloadResult {
        let mut response = match self.client.get(&metadata.url).send().await {
            Ok(response) => response,
            Err(_) => {
                return NativeDownloadResult::Failed(operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    transport_failure(RemoteFailurePhase::Download),
                ));
            }
        };
        if !response.status().is_success() {
            return NativeDownloadResult::Failed(operation_failed(
                acquisition_id,
                member_id,
                attempt,
                http_failure(RemoteFailurePhase::Download, &response),
            ));
        }

        let staged_storage_path = self.assets.incoming_storage_path();
        if let Err(error) = self.journal.record_path(&staged_storage_path) {
            return NativeDownloadResult::Failed(operation_failed(
                acquisition_id,
                member_id,
                attempt,
                model_error_failure(RemoteFailurePhase::Download, error),
            ));
        }
        let staged_path = self.assets.resolve_storage_path(&staged_storage_path);
        let mut staged = match tokio::fs::File::create(&staged_path).await {
            Ok(staged) => staged,
            Err(error) => {
                return NativeDownloadResult::Failed(operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    storage_failure(RemoteFailurePhase::Download, &error.to_string()),
                ));
            }
        };
        loop {
            let chunk = match response.chunk().await {
                Ok(chunk) => chunk,
                Err(_) => {
                    return NativeDownloadResult::Failed(
                        self.remove_failed_download(
                            acquisition_id,
                            member_id,
                            attempt,
                            staged_path,
                            transport_failure(RemoteFailurePhase::Download),
                        )
                        .await,
                    );
                }
            };
            let Some(chunk) = chunk else {
                break;
            };
            if let Err(error) = staged.write_all(&chunk).await {
                return NativeDownloadResult::Failed(
                    self.remove_failed_download(
                        acquisition_id,
                        member_id,
                        attempt,
                        staged_path,
                        storage_failure(RemoteFailurePhase::Download, &error.to_string()),
                    )
                    .await,
                );
            }
        }
        if let Err(error) = staged.sync_all().await {
            return NativeDownloadResult::Failed(
                self.remove_failed_download(
                    acquisition_id,
                    member_id,
                    attempt,
                    staged_path,
                    storage_failure(RemoteFailurePhase::Download, &error.to_string()),
                )
                .await,
            );
        }

        let assets = self.assets.clone();
        let install_metadata = metadata.clone();
        let install_path = staged_path.clone();
        let journal = self.journal.clone();
        let install = tokio::task::spawn_blocking(move || {
            assets.install_remote_staged(&install_path, &install_metadata, kind, Some(&journal))
        })
        .await;
        let installed = match install {
            Ok(Ok(installed)) => installed,
            Ok(Err(error)) => {
                return NativeDownloadResult::Failed(
                    self.remove_failed_download(
                        acquisition_id,
                        member_id,
                        attempt,
                        staged_path,
                        model_error_failure(RemoteFailurePhase::Download, error),
                    )
                    .await,
                );
            }
            Err(_) => {
                return NativeDownloadResult::Failed(
                    self.remove_failed_download(
                        acquisition_id,
                        member_id,
                        attempt,
                        staged_path,
                        storage_failure(
                            RemoteFailurePhase::Download,
                            "remote asset installation task failed",
                        ),
                    )
                    .await,
                );
            }
        };
        if installed.already_present {
            if let Err(error) = tokio::fs::remove_file(&staged_path).await {
                return NativeDownloadResult::Failed(operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    storage_failure(RemoteFailurePhase::Download, &error.to_string()),
                ));
            }
        }
        let asset_ids = vec![installed.record.id.clone()];
        let created_asset_ids = if installed.already_present {
            Vec::new()
        } else {
            asset_ids.clone()
        };
        NativeDownloadResult::Succeeded {
            event: RemoteAcquisitionEvent::DownloadSucceeded {
                acquisition_id,
                member_id,
                attempt,
                asset_ids,
                created_asset_ids,
            },
            record: installed.record,
        }
    }

    async fn remove_failed_download(
        &self,
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        path: std::path::PathBuf,
        failure: RemoteFailure,
    ) -> RemoteAcquisitionEvent {
        match tokio::fs::remove_file(path).await {
            Ok(()) => operation_failed(acquisition_id, member_id, attempt, failure),
            Err(error) => operation_failed(
                acquisition_id,
                member_id,
                attempt,
                storage_failure(RemoteFailurePhase::Download, &error.to_string()),
            ),
        }
    }

    fn cleanup(
        &self,
        acquisition_id: String,
        member_id: u32,
        attempt: u8,
        asset_ids: Vec<String>,
        downloaded: &mut BTreeMap<String, AssetRecord>,
    ) -> RemoteAcquisitionEvent {
        for asset_id in asset_ids {
            let Some(record) = downloaded.remove(&asset_id) else {
                return operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    storage_failure(
                        RemoteFailurePhase::Cleanup,
                        "cleanup asset is absent from the acquisition",
                    ),
                );
            };
            if let Err(error) = self.assets.delete_asset(&record) {
                return operation_failed(
                    acquisition_id,
                    member_id,
                    attempt,
                    model_error_failure(RemoteFailurePhase::Cleanup, error),
                );
            }
        }
        RemoteAcquisitionEvent::CleanupSucceeded {
            acquisition_id,
            member_id,
            attempt,
        }
    }
}

enum NativeDownloadResult {
    Succeeded {
        event: RemoteAcquisitionEvent,
        record: AssetRecord,
    },
    Failed(RemoteAcquisitionEvent),
}

fn metadata_headers(headers: &HeaderMap) -> RemoteMetadataHeaders {
    RemoteMetadataHeaders {
        content_length: header_u64(headers, reqwest::header::CONTENT_LENGTH.as_str()),
        linked_size: header_u64(headers, X_LINKED_SIZE),
        etag: header_text(headers, reqwest::header::ETAG.as_str()),
        linked_etag: header_text(headers, X_LINKED_ETAG),
        last_modified: header_text(headers, reqwest::header::LAST_MODIFIED.as_str()),
    }
}

fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header_text(headers, name).and_then(|value| value.parse().ok())
}

fn operation_failed(
    acquisition_id: String,
    member_id: u32,
    attempt: u8,
    failure: RemoteFailure,
) -> RemoteAcquisitionEvent {
    RemoteAcquisitionEvent::OperationFailed {
        acquisition_id,
        member_id,
        attempt,
        failure,
        created_asset_ids: Vec::new(),
    }
}

fn http_failure(phase: RemoteFailurePhase, response: &reqwest::Response) -> RemoteFailure {
    RemoteFailure {
        phase,
        kind: RemoteFailureKind::Http,
        status: Some(response.status().as_u16()),
        retry_after: header_text(response.headers(), reqwest::header::RETRY_AFTER.as_str()),
        reason: format!("HTTP {}", response.status().as_u16()),
    }
}

fn transport_failure(phase: RemoteFailurePhase) -> RemoteFailure {
    RemoteFailure {
        phase,
        kind: RemoteFailureKind::Transport,
        status: None,
        retry_after: None,
        reason: "request transport failed".to_string(),
    }
}

fn integrity_failure(phase: RemoteFailurePhase, reason: &str) -> RemoteFailure {
    RemoteFailure {
        phase,
        kind: RemoteFailureKind::Integrity,
        status: None,
        retry_after: None,
        reason: reason.to_string(),
    }
}

fn storage_failure(phase: RemoteFailurePhase, reason: &str) -> RemoteFailure {
    RemoteFailure {
        phase,
        kind: RemoteFailureKind::Storage,
        status: None,
        retry_after: None,
        reason: reason.to_string(),
    }
}

fn model_error_failure(phase: RemoteFailurePhase, error: ModelError) -> RemoteFailure {
    match error {
        ModelError::RemoteIntegrityFailed { reason, .. } => integrity_failure(phase, &reason),
        error => RemoteFailure {
            phase,
            kind: RemoteFailureKind::Storage,
            status: None,
            retry_after: None,
            reason: error.to_string(),
        },
    }
}
