//! Unit tests for the native remote acquisition executor.
//!
//! Covers timeout handling for deterministic local HTTP fixtures without live
//! network access or model loading.

use std::{collections::BTreeMap, time::Duration};

use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use crate::lifecycle::test_support::TempDir;
use crate::lifecycle::{AssetStore, RegistryManifest};

use super::*;

fn short_timeouts() -> NativeHttpTimeouts {
    NativeHttpTimeouts {
        connect: Duration::from_secs(1),
        read: Duration::from_millis(50),
        metadata_request: Duration::from_millis(50),
    }
}

#[tokio::test]
async fn metadata_request_timeout_reports_transport_failure() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/model.gguf"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;

    let root = TempDir::new("native-acquisition", "metadata-timeout");
    let assets = AssetStore::local(root.path.clone());
    let journal = assets.acquisition_journal("lease-metadata-timeout");
    let executor =
        NativeRemoteExecutor::with_timeouts(assets, journal, short_timeouts()).expect("executor");
    let mut downloaded = BTreeMap::new();
    let event = tokio::time::timeout(
        Duration::from_secs(1),
        executor.execute(
            RemoteAction::FetchMetadata {
                acquisition_id: "lease-metadata-timeout".to_string(),
                member_id: 0,
                attempt: 1,
                url: format!("{}/model.gguf", server.uri()),
            },
            &RegistryManifest::default(),
            &mut downloaded,
        ),
    )
    .await
    .expect("metadata request should return through its configured timeout");

    assert_transport_failure(event, RemoteFailurePhase::Metadata);
}

#[tokio::test]
async fn download_read_timeout_reports_transport_failure() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.gguf"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;

    let root = TempDir::new("native-acquisition", "download-timeout");
    let assets = AssetStore::local(root.path.clone());
    let journal = assets.acquisition_journal("lease-download-timeout");
    let executor =
        NativeRemoteExecutor::with_timeouts(assets, journal, short_timeouts()).expect("executor");
    let mut downloaded = BTreeMap::new();
    let event = tokio::time::timeout(
        Duration::from_secs(1),
        executor.execute(
            RemoteAction::Download {
                acquisition_id: "lease-download-timeout".to_string(),
                member_id: 0,
                attempt: 1,
                metadata: super::super::RemoteMetadata {
                    url: format!("{}/model.gguf", server.uri()),
                    name: "model.gguf".to_string(),
                    bytes: 1,
                    etag: None,
                    last_modified: None,
                },
            },
            &RegistryManifest::default(),
            &mut downloaded,
        ),
    )
    .await
    .expect("download request should return through its configured timeout");

    assert_transport_failure(event, RemoteFailurePhase::Download);
}

fn assert_transport_failure(event: RemoteAcquisitionEvent, phase: RemoteFailurePhase) {
    assert!(
        matches!(
            event,
            RemoteAcquisitionEvent::OperationFailed {
                failure: RemoteFailure {
                    phase: actual_phase,
                    kind: RemoteFailureKind::Transport,
                    status: None,
                    ..
                },
                ref created_asset_ids,
                ..
            } if actual_phase == phase && created_asset_ids.is_empty()
        ),
        "expected {phase:?} transport failure, got {event:?}"
    );
}
