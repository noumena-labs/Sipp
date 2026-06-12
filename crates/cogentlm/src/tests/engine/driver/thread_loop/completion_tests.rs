//! Tests the `engine::driver::thread_loop::completion` module in `cogentlm`.
//!
//! Covers driver futures, command handling, event emission, and request mapping with model-free channels or explicitly ignored model smoke tests.

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};

use futures::executor::block_on;
use futures_channel::oneshot;

use crate::engine::protocol::{
    EmbeddingCapabilities, EngineEvent, ModelCapabilities, ModelClass, ModelState, PoolingType,
};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus, ResponseOutput};

use super::super::{ActiveRequest, ActiveRequestOutput, EngineThreadState};

fn model_state() -> ModelState {
    ModelState {
        id: "model".to_string(),
        name: "model".to_string(),
        capabilities: ModelCapabilities {
            model_class: ModelClass::EncoderOnly,
            supports_text_generation: false,
            supports_embeddings: true,
            has_chat_template: false,
            embedding: Some(EmbeddingCapabilities {
                dimensions: 2,
                pooling: PoolingType::Mean,
            }),
        },
    }
}

#[test]
fn embedding_completion_is_forwarded_without_generation_mapping() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .request_queue
        .mark_completed(GenerateResponse::terminal(
            7,
            GenerateResponseStatus::Completed,
            ResponseOutput::Embedding {
                values: vec![0.6, 0.8],
                pooling: PoolingType::Mean,
                normalized: true,
            },
            "",
        ));

    let (response_tx, response_rx) = oneshot::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let mut active_requests = HashMap::new();
    active_requests.insert(
        7,
        ActiveRequest {
            output: ActiveRequestOutput::Embedding,
            response_tx,
            token: None,
        },
    );
    let mut state = EngineThreadState {
        runtime: Some(runtime),
        active_requests,
        model_state: model_state(),
        event_subscribers: Arc::new(Mutex::new(vec![event_tx])),
    };

    state.complete_finished_requests();

    let response = block_on(response_rx)
        .expect("response")
        .expect("embedding ok");
    assert!(matches!(response.output, ResponseOutput::Embedding { .. }));
    assert!(matches!(
        event_rx.recv().expect("completion event"),
        EngineEvent::RequestCompleted { request_id } if request_id == "7"
    ));
}

#[test]
fn wrong_completed_output_variant_is_a_failed_request() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .request_queue
        .mark_completed(GenerateResponse::terminal(
            7,
            GenerateResponseStatus::Completed,
            ResponseOutput::Text("text".to_string()),
            "",
        ));

    let (response_tx, response_rx) = oneshot::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let mut active_requests = HashMap::new();
    active_requests.insert(
        7,
        ActiveRequest {
            output: ActiveRequestOutput::Embedding,
            response_tx,
            token: None,
        },
    );
    let mut state = EngineThreadState {
        runtime: Some(runtime),
        active_requests,
        model_state: model_state(),
        event_subscribers: Arc::new(Mutex::new(vec![event_tx])),
    };

    state.complete_finished_requests();

    let error = block_on(response_rx)
        .expect("response")
        .expect_err("wrong output");
    assert!(error.to_string().contains("text output"));
    assert!(matches!(
        event_rx.recv().expect("failure event"),
        EngineEvent::RequestFailed { request_id, .. } if request_id == "7"
    ));
}

#[test]
fn cancelled_completion_uses_fallback_error_and_failure_event() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .request_queue
        .mark_completed(GenerateResponse::cancelled(7, ""));

    let (response_tx, response_rx) = oneshot::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let mut active_requests = HashMap::new();
    active_requests.insert(
        7,
        ActiveRequest {
            output: ActiveRequestOutput::Text,
            response_tx,
            token: None,
        },
    );
    let mut state = EngineThreadState {
        runtime: Some(runtime),
        active_requests,
        model_state: model_state(),
        event_subscribers: Arc::new(Mutex::new(vec![event_tx])),
    };

    state.complete_finished_requests();

    let error = block_on(response_rx)
        .expect("response")
        .expect_err("cancelled response");
    assert!(error.to_string().contains("request cancelled"));
    assert!(matches!(
        event_rx.recv().expect("failure event"),
        EngineEvent::RequestFailed { request_id, error }
            if request_id == "7" && error == "request cancelled"
    ));
}

#[test]
fn pending_completion_returns_error_without_failure_event() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .request_queue
        .mark_completed(GenerateResponse::terminal(
            7,
            GenerateResponseStatus::Pending,
            ResponseOutput::Text(String::new()),
            "",
        ));

    let (response_tx, response_rx) = oneshot::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let mut active_requests = HashMap::new();
    active_requests.insert(
        7,
        ActiveRequest {
            output: ActiveRequestOutput::Text,
            response_tx,
            token: None,
        },
    );
    let mut state = EngineThreadState {
        runtime: Some(runtime),
        active_requests,
        model_state: model_state(),
        event_subscribers: Arc::new(Mutex::new(vec![event_tx])),
    };

    state.complete_finished_requests();

    let error = block_on(response_rx)
        .expect("response")
        .expect_err("pending response");
    assert!(error.to_string().contains("pending response"));
    assert!(event_rx.try_recv().is_err());
}
