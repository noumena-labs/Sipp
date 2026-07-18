//! Tests the shared remote acquisition state machine.
//!
//! Covers retry policy, exact cache identity, stale events, and transactional
//! cleanup through pure protocol events without HTTP or filesystem access.

use super::*;
use crate::lifecycle::ModelError;

fn request(member_id: u32, candidates: Vec<RemoteCacheCandidate>) -> RemoteAcquisitionRequest {
    RemoteAcquisitionRequest {
        member_id,
        url: format!("https://example.test/model-{member_id}.gguf"),
        candidates,
    }
}

fn metadata(url: &str, etag: &str) -> RemoteMetadata {
    RemoteMetadata {
        url: url.to_string(),
        name: "model.gguf".to_string(),
        bytes: 8,
        etag: Some(etag.to_string()),
        last_modified: None,
    }
}

fn headers(etag: &str) -> RemoteMetadataHeaders {
    RemoteMetadataHeaders {
        content_length: Some(8),
        linked_size: None,
        etag: Some(etag.to_string()),
        linked_etag: None,
        last_modified: None,
    }
}

fn failure(
    acquisition_id: &str,
    member_id: u32,
    attempt: u8,
    phase: RemoteFailurePhase,
    status: u16,
) -> RemoteAcquisitionEvent {
    RemoteAcquisitionEvent::OperationFailed {
        acquisition_id: acquisition_id.to_string(),
        member_id,
        attempt,
        failure: RemoteFailure {
            phase,
            kind: RemoteFailureKind::Http,
            status: Some(status),
            retry_after: None,
            reason: format!("HTTP {status}"),
        },
        created_asset_ids: Vec::new(),
    }
}

#[test]
fn retries_503_with_shared_attempt_and_backoff_policy() {
    let mut acquisition =
        RemoteAcquisition::new("lease-1".to_string(), vec![request(0, Vec::new())])
            .expect("acquisition");

    let progress = acquisition
        .advance(failure("lease-1", 0, 1, RemoteFailurePhase::Metadata, 503))
        .expect("retry");
    assert!(matches!(
        progress,
        RemoteAcquisitionProgress::Action(RemoteAction::Wait {
            attempt: 1,
            delay_ms: 250,
            ..
        })
    ));

    let progress = acquisition
        .advance(RemoteAcquisitionEvent::WaitCompleted {
            acquisition_id: "lease-1".to_string(),
            member_id: 0,
            attempt: 1,
        })
        .expect("next attempt");
    assert!(matches!(
        progress,
        RemoteAcquisitionProgress::Action(RemoteAction::FetchMetadata { attempt: 2, .. })
    ));
}

#[test]
fn final_503_fails_even_when_a_stale_cache_candidate_exists() {
    let url = "https://example.test/model-0.gguf";
    let candidate = RemoteCacheCandidate {
        candidate_id: "candidate-a".to_string(),
        asset_ids: vec!["asset-a".to_string()],
        metadata: metadata(url, "old"),
    };
    let mut acquisition =
        RemoteAcquisition::new("lease-2".to_string(), vec![request(0, vec![candidate])])
            .expect("acquisition");

    for attempt in 1..=4 {
        let progress = acquisition
            .advance(failure(
                "lease-2",
                0,
                attempt,
                RemoteFailurePhase::Metadata,
                503,
            ))
            .expect("failure result");
        if attempt == 4 {
            assert!(matches!(
                progress,
                RemoteAcquisitionProgress::Failed(ModelError::RemoteMetadataUnavailable {
                    status: Some(503),
                    ..
                })
            ));
            break;
        }
        assert!(matches!(
            progress,
            RemoteAcquisitionProgress::Action(RemoteAction::Wait { .. })
        ));
        acquisition
            .advance(RemoteAcquisitionEvent::WaitCompleted {
                acquisition_id: "lease-2".to_string(),
                member_id: 0,
                attempt,
            })
            .expect("next attempt");
    }
}

#[test]
fn cache_is_selected_only_for_an_exact_remote_identity() {
    let url = "https://example.test/model-0.gguf";
    let candidate = RemoteCacheCandidate {
        candidate_id: "candidate-a".to_string(),
        asset_ids: vec!["asset-a".to_string()],
        metadata: metadata(url, "current"),
    };
    let mut exact = RemoteAcquisition::new(
        "lease-3".to_string(),
        vec![request(0, vec![candidate.clone()])],
    )
    .expect("exact acquisition");
    let exact_progress = exact
        .advance(RemoteAcquisitionEvent::MetadataSucceeded {
            acquisition_id: "lease-3".to_string(),
            member_id: 0,
            attempt: 1,
            headers: headers("current"),
        })
        .expect("metadata");
    assert!(matches!(
        exact_progress,
        RemoteAcquisitionProgress::Action(RemoteAction::ValidateCache { .. })
    ));

    let mut changed =
        RemoteAcquisition::new("lease-4".to_string(), vec![request(0, vec![candidate])])
            .expect("changed acquisition");
    let changed_progress = changed
        .advance(RemoteAcquisitionEvent::MetadataSucceeded {
            acquisition_id: "lease-4".to_string(),
            member_id: 0,
            attempt: 1,
            headers: headers("changed"),
        })
        .expect("metadata");
    assert!(matches!(
        changed_progress,
        RemoteAcquisitionProgress::Action(RemoteAction::Download { .. })
    ));
}

#[test]
fn linked_huggingface_headers_describe_the_downloaded_blob() {
    let mut acquisition =
        RemoteAcquisition::new("lease-hf".to_string(), vec![request(0, Vec::new())])
            .expect("acquisition");

    let progress = acquisition
        .advance(RemoteAcquisitionEvent::MetadataSucceeded {
            acquisition_id: "lease-hf".to_string(),
            member_id: 0,
            attempt: 1,
            headers: RemoteMetadataHeaders {
                content_length: Some(1_038),
                linked_size: Some(428_730_208),
                etag: Some("\"redirect\"".to_string()),
                linked_etag: Some(
                    "\"7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed\""
                        .to_string(),
                ),
                last_modified: None,
            },
        })
        .expect("metadata");

    let RemoteAcquisitionProgress::Action(RemoteAction::Download { metadata, .. }) = progress
    else {
        panic!("expected download action");
    };
    assert_eq!(metadata.bytes, 428_730_208);
    assert_eq!(
        metadata.etag.as_deref(),
        Some("\"7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed\"")
    );
}

#[test]
fn stale_acquisition_results_and_wrong_failure_phases_are_rejected() {
    let mut acquisition =
        RemoteAcquisition::new("lease-5".to_string(), vec![request(0, Vec::new())])
            .expect("acquisition");

    let stale = acquisition
        .advance(failure(
            "lease-old",
            0,
            1,
            RemoteFailurePhase::Metadata,
            503,
        ))
        .expect_err("stale event");
    assert!(matches!(stale, ModelError::StaleAcquisitionResult { .. }));

    let wrong_phase = acquisition
        .advance(failure("lease-5", 0, 1, RemoteFailurePhase::Download, 503))
        .expect_err("wrong phase");
    assert!(matches!(wrong_phase, ModelError::InvalidModelSource(_)));
}

#[test]
fn failure_cleans_assets_created_by_earlier_members_before_returning() {
    let mut acquisition = RemoteAcquisition::new(
        "lease-6".to_string(),
        vec![request(0, Vec::new()), request(1, Vec::new())],
    )
    .expect("acquisition");
    acquisition
        .advance(RemoteAcquisitionEvent::MetadataSucceeded {
            acquisition_id: "lease-6".to_string(),
            member_id: 0,
            attempt: 1,
            headers: headers("first"),
        })
        .expect("metadata");
    acquisition
        .advance(RemoteAcquisitionEvent::DownloadSucceeded {
            acquisition_id: "lease-6".to_string(),
            member_id: 0,
            attempt: 1,
            asset_ids: vec!["asset-new".to_string()],
            created_asset_ids: vec!["asset-new".to_string()],
        })
        .expect("download");

    let progress = acquisition
        .advance(failure("lease-6", 1, 1, RemoteFailurePhase::Metadata, 400))
        .expect("terminal cleanup");
    assert!(matches!(
        progress,
        RemoteAcquisitionProgress::Action(RemoteAction::Cleanup {
            member_id: 0,
            ref asset_ids,
            ..
        }) if asset_ids == &["asset-new".to_string()]
    ));

    let progress = acquisition
        .advance(RemoteAcquisitionEvent::CleanupSucceeded {
            acquisition_id: "lease-6".to_string(),
            member_id: 0,
            attempt: 1,
        })
        .expect("cleanup");
    assert!(matches!(
        progress,
        RemoteAcquisitionProgress::Failed(ModelError::RemoteMetadataUnavailable {
            status: Some(400),
            ..
        })
    ));
}

#[test]
fn download_failure_cleans_assets_created_by_the_failed_operation() {
    let mut acquisition =
        RemoteAcquisition::new("lease-7".to_string(), vec![request(0, Vec::new())])
            .expect("acquisition");
    acquisition
        .advance(RemoteAcquisitionEvent::MetadataSucceeded {
            acquisition_id: "lease-7".to_string(),
            member_id: 0,
            attempt: 1,
            headers: headers("download"),
        })
        .expect("metadata");

    let progress = acquisition
        .advance(RemoteAcquisitionEvent::OperationFailed {
            acquisition_id: "lease-7".to_string(),
            member_id: 0,
            attempt: 1,
            failure: RemoteFailure {
                phase: RemoteFailurePhase::Download,
                kind: RemoteFailureKind::Storage,
                status: None,
                retry_after: None,
                reason: "classification failed".to_string(),
            },
            created_asset_ids: vec!["asset-new".to_string()],
        })
        .expect("download failure cleanup");
    assert!(matches!(
        progress,
        RemoteAcquisitionProgress::Action(RemoteAction::Cleanup {
            member_id: 0,
            ref asset_ids,
            ..
        }) if asset_ids == &["asset-new".to_string()]
    ));

    let progress = acquisition
        .advance(RemoteAcquisitionEvent::CleanupSucceeded {
            acquisition_id: "lease-7".to_string(),
            member_id: 0,
            attempt: 1,
        })
        .expect("cleanup");
    assert!(matches!(
        progress,
        RemoteAcquisitionProgress::Failed(ModelError::RemoteDownloadFailed {
            reason,
            ..
        }) if reason == "classification failed"
    ));
}
