use std::collections::VecDeque;

use crate::lifecycle::ModelError;

use super::policy::retry_delay_ms;
use super::{
    canonical_remote_url, RemoteAcquisitionEvent, RemoteAcquisitionProgress,
    RemoteAcquisitionRequest, RemoteAction, RemoteCacheCandidate, RemoteFailure, RemoteFailureKind,
    RemoteFailurePhase, RemoteMetadata, RemoteMetadataHeaders, RemoteResolvedMember,
};

const EMPTY_REMOTE_SOURCE: &str = "remote model URLs must not be empty";

#[derive(Debug, Default)]
pub(crate) struct RemoteAcquisitionIds {
    last: u64,
}

impl RemoteAcquisitionIds {
    pub(crate) fn issue(&mut self) -> Result<String, ModelError> {
        self.last = self.last.checked_add(1).ok_or_else(|| {
            ModelError::RemoteClient("remote acquisition identifier space is exhausted".to_string())
        })?;
        Ok(self.last.to_string())
    }
}

#[derive(Debug, Clone)]
enum ActiveOperation {
    Metadata,
    Waiting {
        next: Box<ActiveOperation>,
        delay_ms: u64,
    },
    Cache(RemoteCacheCandidate),
    Download(RemoteMetadata),
    Cleanup {
        member_id: u32,
        url: String,
        asset_ids: Vec<String>,
    },
}

#[derive(Debug)]
struct CreatedAssets {
    member_id: u32,
    url: String,
    asset_ids: Vec<String>,
}

#[derive(Debug)]
enum Terminal {
    Failed(ModelError),
    Cancelled,
}

/// Deterministic Rust-owned acquisition state for remote model assets.
#[derive(Debug)]
pub(crate) struct RemoteAcquisition {
    id: String,
    requests: Vec<RemoteAcquisitionRequest>,
    request_index: usize,
    attempt: u8,
    operation: ActiveOperation,
    resolved: Vec<RemoteResolvedMember>,
    created: VecDeque<CreatedAssets>,
    terminal: Option<Terminal>,
}

impl RemoteAcquisition {
    pub(crate) fn new(
        id: String,
        mut requests: Vec<RemoteAcquisitionRequest>,
    ) -> Result<Self, ModelError> {
        if requests.is_empty() {
            return Err(ModelError::InvalidModelSource(
                EMPTY_REMOTE_SOURCE.to_string(),
            ));
        }
        for request in &mut requests {
            request.url = canonical_remote_url(&request.url)?;
            for candidate in &mut request.candidates {
                candidate.metadata.url = canonical_remote_url(&candidate.metadata.url)?;
            }
        }
        Ok(Self {
            id,
            requests,
            request_index: 0,
            attempt: 1,
            operation: ActiveOperation::Metadata,
            resolved: Vec::new(),
            created: VecDeque::new(),
            terminal: None,
        })
    }

    pub(crate) fn progress(&mut self) -> RemoteAcquisitionProgress {
        if matches!(self.operation, ActiveOperation::Cleanup { .. }) {
            return RemoteAcquisitionProgress::Action(self.action());
        }
        if let Some(terminal) = self.terminal.take() {
            return match terminal {
                Terminal::Failed(error) => RemoteAcquisitionProgress::Failed(error),
                Terminal::Cancelled => RemoteAcquisitionProgress::Cancelled,
            };
        }
        if self.request_index == self.requests.len() {
            return RemoteAcquisitionProgress::Ready(self.resolved.clone());
        }
        RemoteAcquisitionProgress::Action(self.action())
    }

    pub(crate) fn advance(
        &mut self,
        event: RemoteAcquisitionEvent,
    ) -> Result<RemoteAcquisitionProgress, ModelError> {
        self.validate_event(&event)?;
        if matches!(event, RemoteAcquisitionEvent::Cancelled { .. }) {
            self.begin_terminal(Terminal::Cancelled);
            return Ok(self.progress());
        }
        match (&self.operation, event) {
            (
                ActiveOperation::Metadata,
                RemoteAcquisitionEvent::MetadataSucceeded { headers, .. },
            ) => {
                if let Err(failure) = self.accept_metadata(headers) {
                    self.fail(failure);
                }
            }
            (
                ActiveOperation::Metadata,
                RemoteAcquisitionEvent::OperationFailed { failure, .. },
            )
            | (
                ActiveOperation::Download(_),
                RemoteAcquisitionEvent::OperationFailed { failure, .. },
            ) => {
                self.accept_failure(failure);
            }
            (
                ActiveOperation::Waiting { next, .. },
                RemoteAcquisitionEvent::WaitCompleted { .. },
            ) => {
                self.attempt += 1;
                self.operation = *next.clone();
            }
            (
                ActiveOperation::Cache(candidate),
                RemoteAcquisitionEvent::CacheValidated { asset_ids, .. },
            ) => {
                if asset_ids == candidate.asset_ids {
                    self.complete_member(asset_ids, Vec::new());
                } else {
                    self.fail(integrity_failure(
                        "validated cache assets do not match the selected candidate",
                    ));
                }
            }
            (
                ActiveOperation::Cache(_),
                RemoteAcquisitionEvent::OperationFailed { failure, .. },
            ) => {
                self.fail(failure);
            }
            (
                ActiveOperation::Download(_),
                RemoteAcquisitionEvent::DownloadSucceeded {
                    asset_ids,
                    created_asset_ids,
                    ..
                },
            ) => {
                if asset_ids.is_empty() {
                    self.fail(integrity_failure("download produced no assets"));
                } else {
                    self.complete_member(asset_ids, created_asset_ids);
                }
            }
            (ActiveOperation::Cleanup { .. }, RemoteAcquisitionEvent::CleanupSucceeded { .. }) => {
                self.created.pop_front();
                self.continue_cleanup();
            }
            (
                ActiveOperation::Cleanup { url, .. },
                RemoteAcquisitionEvent::OperationFailed { failure, .. },
            ) => {
                self.created.clear();
                self.terminal = Some(Terminal::Failed(failure.model_error(url)));
                self.operation = ActiveOperation::Metadata;
            }
            _ => {
                return Err(ModelError::InvalidModelSource(
                    "remote acquisition event does not match the active operation".to_string(),
                ));
            }
        }
        Ok(self.progress())
    }

    pub(crate) fn validate_event(&self, event: &RemoteAcquisitionEvent) -> Result<(), ModelError> {
        self.validate_event_identity(event)?;
        self.validate_failure_phase(event)
    }

    fn action(&self) -> RemoteAction {
        let request = self.request();
        match &self.operation {
            ActiveOperation::Metadata => RemoteAction::FetchMetadata {
                acquisition_id: self.id.clone(),
                member_id: request.member_id,
                attempt: self.attempt,
                url: request.url.clone(),
            },
            ActiveOperation::Waiting { delay_ms, .. } => RemoteAction::Wait {
                acquisition_id: self.id.clone(),
                member_id: request.member_id,
                attempt: self.attempt,
                delay_ms: *delay_ms,
            },
            ActiveOperation::Cache(candidate) => RemoteAction::ValidateCache {
                acquisition_id: self.id.clone(),
                member_id: request.member_id,
                attempt: self.attempt,
                candidate: candidate.clone(),
            },
            ActiveOperation::Download(metadata) => RemoteAction::Download {
                acquisition_id: self.id.clone(),
                member_id: request.member_id,
                attempt: self.attempt,
                role: request.role,
                metadata: metadata.clone(),
            },
            ActiveOperation::Cleanup {
                member_id,
                asset_ids,
                ..
            } => RemoteAction::Cleanup {
                acquisition_id: self.id.clone(),
                member_id: *member_id,
                attempt: self.attempt,
                asset_ids: asset_ids.clone(),
            },
        }
    }

    fn accept_metadata(&mut self, headers: RemoteMetadataHeaders) -> Result<(), RemoteFailure> {
        let request = self.request();
        let metadata = metadata_from_headers(&request.url, headers)?;
        let matching: Vec<_> = request
            .candidates
            .iter()
            .filter(|candidate| metadata_identity_matches(&candidate.metadata, &metadata))
            .cloned()
            .collect();
        self.operation = match matching.as_slice() {
            [] => ActiveOperation::Download(metadata),
            [candidate] => ActiveOperation::Cache(candidate.clone()),
            _ => {
                return Err(integrity_failure(
                    "multiple cache candidates have the same remote identity",
                ));
            }
        };
        Ok(())
    }

    fn accept_failure(&mut self, failure: RemoteFailure) {
        if let Some(delay_ms) = retry_delay_ms(&failure, self.attempt) {
            self.operation = ActiveOperation::Waiting {
                next: Box::new(self.operation.clone()),
                delay_ms,
            };
        } else {
            self.fail(failure);
        }
    }

    fn fail(&mut self, failure: RemoteFailure) {
        let url = self.request().url.clone();
        self.begin_terminal(Terminal::Failed(failure.model_error(&url)));
    }

    pub(crate) fn fail_finalization(&mut self, error: ModelError) -> RemoteAcquisitionProgress {
        self.begin_terminal(Terminal::Failed(error));
        self.progress()
    }

    fn complete_member(&mut self, asset_ids: Vec<String>, created_asset_ids: Vec<String>) {
        let request = self.request().clone();
        if !created_asset_ids.is_empty() {
            self.created.push_back(CreatedAssets {
                member_id: request.member_id,
                url: request.url.clone(),
                asset_ids: created_asset_ids.clone(),
            });
        }
        self.resolved.push(RemoteResolvedMember {
            member_id: request.member_id,
            role: request.role,
            asset_ids,
            created_asset_ids,
        });
        self.request_index += 1;
        self.attempt = 1;
        self.operation = ActiveOperation::Metadata;
    }

    fn begin_terminal(&mut self, terminal: Terminal) {
        self.terminal = Some(terminal);
        self.continue_cleanup();
    }

    fn continue_cleanup(&mut self) {
        self.operation = self
            .created
            .front()
            .map_or(ActiveOperation::Metadata, |created| {
                ActiveOperation::Cleanup {
                    member_id: created.member_id,
                    url: created.url.clone(),
                    asset_ids: created.asset_ids.clone(),
                }
            });
    }

    fn request(&self) -> &RemoteAcquisitionRequest {
        let index = self.request_index.min(self.requests.len() - 1);
        &self.requests[index]
    }

    fn validate_event_identity(&self, event: &RemoteAcquisitionEvent) -> Result<(), ModelError> {
        let (received_id, member_attempt) = event.identity();
        if received_id != self.id {
            return Err(ModelError::StaleAcquisitionResult {
                expected: self.id.clone(),
                received: received_id.to_string(),
            });
        }
        if let Some((member_id, attempt)) = member_attempt {
            let action = self.action();
            let (expected_id, expected_member, expected_attempt) = action.identity();
            if member_id != expected_member || attempt != expected_attempt {
                return Err(ModelError::StaleAcquisitionResult {
                    expected: format!("{expected_id}:{expected_member}:{expected_attempt}"),
                    received: format!("{received_id}:{member_id}:{attempt}"),
                });
            }
        }
        Ok(())
    }

    fn validate_failure_phase(&self, event: &RemoteAcquisitionEvent) -> Result<(), ModelError> {
        let RemoteAcquisitionEvent::OperationFailed { failure, .. } = event else {
            return Ok(());
        };
        let expected = match self.operation {
            ActiveOperation::Metadata => RemoteFailurePhase::Metadata,
            ActiveOperation::Cache(_) => RemoteFailurePhase::CacheValidation,
            ActiveOperation::Download(_) => RemoteFailurePhase::Download,
            ActiveOperation::Cleanup { .. } => RemoteFailurePhase::Cleanup,
            ActiveOperation::Waiting { .. } => {
                return Err(ModelError::InvalidModelSource(
                    "a wait action cannot report an operation failure".to_string(),
                ));
            }
        };
        if failure.phase != expected {
            return Err(ModelError::InvalidModelSource(format!(
                "remote acquisition failure phase {:?} does not match active phase {:?}",
                failure.phase, expected
            )));
        }
        Ok(())
    }
}

fn metadata_identity_matches(left: &RemoteMetadata, right: &RemoteMetadata) -> bool {
    left.url == right.url
        && left.bytes == right.bytes
        && left.etag == right.etag
        && left.last_modified == right.last_modified
}

fn metadata_from_headers(
    url: &str,
    headers: RemoteMetadataHeaders,
) -> Result<RemoteMetadata, RemoteFailure> {
    let bytes = headers
        .linked_size
        .or(headers.content_length)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| invalid_metadata("Content-Length or X-Linked-Size is required"))?;
    let etag = non_empty(headers.linked_etag).or_else(|| non_empty(headers.etag));
    let last_modified = non_empty(headers.last_modified);
    if etag.is_none() && last_modified.is_none() {
        return Err(invalid_metadata(
            "ETag, X-Linked-Etag, or Last-Modified is required",
        ));
    }
    let parsed = url::Url::parse(url).map_err(|error| invalid_metadata(&error.to_string()))?;
    let name = parsed
        .path_segments()
        .and_then(Iterator::last)
        .filter(|name| !name.is_empty())
        .unwrap_or("model.gguf")
        .to_string();
    Ok(RemoteMetadata {
        url: url.to_string(),
        name,
        bytes,
        etag,
        last_modified,
    })
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| (!value.trim().is_empty()).then_some(value))
}

fn invalid_metadata(reason: &str) -> RemoteFailure {
    RemoteFailure {
        phase: RemoteFailurePhase::Metadata,
        kind: RemoteFailureKind::InvalidResponse,
        status: None,
        retry_after: None,
        reason: reason.to_string(),
    }
}

fn integrity_failure(reason: &str) -> RemoteFailure {
    RemoteFailure {
        phase: RemoteFailurePhase::CacheValidation,
        kind: RemoteFailureKind::Integrity,
        status: None,
        retry_after: None,
        reason: reason.to_string(),
    }
}
