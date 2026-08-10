//! Tests the Swift UniFFI lifecycle bridge.
//!
//! Covers value conversion, typed error metadata, executor-backed empty-store
//! operations, and input validation without loading a model.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use sipp::lifecycle::{ManagedModel, ModelError, ModelModality, ModelStatus};
use sipp::{EndpointError, SippError};

use super::*;

fn test_storage_root(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sipp-swift-{}-{name}", std::process::id()))
}

#[test]
fn managed_model_conversion_preserves_lifecycle_values() {
    let converted = FfiManagedModel::from(ManagedModel {
        id: "model-id".to_owned(),
        name: "Model".to_owned(),
        bytes: 42,
        modality: ModelModality::Vision,
        status: ModelStatus::NeedsProjector,
    });

    assert_eq!(converted.id, "model-id");
    assert_eq!(converted.name, "Model");
    assert_eq!(converted.bytes, 42);
    assert_eq!(converted.modality, FfiModelModality::Vision);
    assert_eq!(converted.status, FfiModelStatus::NeedsProjector);
}

#[test]
fn endpoint_conversion_preserves_the_selected_managed_model() {
    let model = ManagedModel {
        id: "model-id".to_owned(),
        name: "Model".to_owned(),
        bytes: 42,
        modality: ModelModality::Audio,
        status: ModelStatus::Ready,
    };
    let endpoint = FfiEndpoint {
        model: FfiManagedModel::from(model.clone()),
    };

    assert_eq!(
        CoreEndpoint::from(endpoint),
        CoreLocalEndpoint::new(&model).into()
    );
}

#[test]
fn model_modality_conversion_preserves_audio_categories() {
    assert_eq!(
        FfiModelModality::from(ModelModality::Audio),
        FfiModelModality::Audio
    );
    assert_eq!(
        FfiModelModality::from(ModelModality::Multimodal),
        FfiModelModality::Multimodal
    );
}

#[test]
fn lifecycle_error_conversion_preserves_stable_metadata() {
    let converted = FfiError::from(ModelError::RemoteMetadataUnavailable {
        url: "https://example.com/model.gguf".to_owned(),
        status: Some(503),
        retry_after_ms: Some(2500),
        reason: "unavailable".to_owned(),
    });

    let FfiError::ModelLifecycle {
        code,
        status,
        retry_after_ms,
        ..
    } = converted
    else {
        panic!("expected lifecycle error");
    };
    assert_eq!(code, "REMOTE_METADATA_UNAVAILABLE");
    assert_eq!(status, Some(503));
    assert_eq!(retry_after_ms, Some(2500));
}

#[test]
fn endpoint_error_conversion_preserves_structured_metadata() {
    let converted = FfiError::from(SippError::from(EndpointError {
        kind: "rate_limited".to_owned(),
        status: Some(429),
        code: Some("RATE_LIMITED".to_owned()),
        message: "slow down".to_owned(),
        retry_after: None,
        request_id: Some("upstream-request".to_owned()),
        raw: None,
    }));

    let FfiError::Endpoint {
        kind,
        status,
        code,
        request_id,
        ..
    } = converted
    else {
        panic!("expected endpoint error");
    };
    assert_eq!(kind, "rate_limited");
    assert_eq!(status, Some(429));
    assert_eq!(code.as_deref(), Some("RATE_LIMITED"));
    assert_eq!(request_id.as_deref(), Some("upstream-request"));
}

#[test]
fn bridge_executor_aborts_work_when_its_caller_is_cancelled() {
    struct DropSignal(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    let executor = Arc::new(BridgeExecutor::new().unwrap());
    let foreign_executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    foreign_executor.block_on(async {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
        let executor = Arc::clone(&executor);
        let operation = tokio::spawn(async move {
            executor
                .execute(async move {
                    let _drop_signal = DropSignal(Some(dropped_tx));
                    started_tx.send(()).unwrap();
                    futures::future::pending::<Result<(), FfiError>>().await
                })
                .await
        });

        started_rx.await.unwrap();
        operation.abort();
        assert!(operation.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), dropped_rx)
            .await
            .unwrap()
            .unwrap();
    });
}

#[test]
fn client_runs_model_store_and_validation_operations() {
    let storage_root = test_storage_root("client");
    let client = FfiSippClient::new(storage_root.display().to_string(), None).unwrap();
    let models = client.models();
    let foreign_executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    foreign_executor.block_on(async {
        assert!(models.list().await.unwrap().is_empty());

        let add_error = models.add(Vec::new()).await.unwrap_err();
        assert!(matches!(
            add_error,
            FfiError::ModelLifecycle { ref code, .. } if code == "INVALID_MODEL_SOURCE"
        ));

        let remove_error = client.remove(String::new()).await.unwrap_err();
        assert!(matches!(remove_error, FfiError::InvalidArgument { .. }));
    });

    drop(models);
    drop(client);
    std::fs::remove_dir_all(storage_root).unwrap();
}

#[test]
fn model_registration_reports_creation_for_bookmark_rollback() {
    let storage_root = test_storage_root("registration");
    let model_path = storage_root.with_extension("gguf");
    std::fs::write(&model_path, b"stable model bytes").unwrap();
    let client = FfiSippClient::new(storage_root.display().to_string(), None).unwrap();
    let models = client.models();
    let foreign_executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    foreign_executor.block_on(async {
        let first = models
            .add(vec![model_path.display().to_string()])
            .await
            .unwrap();
        let second = models
            .add(vec![model_path.display().to_string()])
            .await
            .unwrap();

        assert!(first.created);
        assert!(!second.created);
        assert_eq!(first.model.id, second.model.id);
    });

    drop(models);
    drop(client);
    std::fs::remove_dir_all(storage_root).unwrap();
    std::fs::remove_file(model_path).unwrap();
}
