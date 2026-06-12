//! Tests the `engine::driver::thread_loop` module in `cogentlm`.
//!
//! Covers driver futures, command handling, event emission, and request mapping with model-free channels or explicitly ignored model smoke tests.

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};

use futures::executor::block_on;
use futures_channel::oneshot;

use crate::engine::protocol::{
    EngineEvent, EngineStatus, ModelCapabilities, ModelClass, ModelState,
};
use crate::engine::{EmbedOptions, EmbedRequest, QueryRequest};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;

use super::*;

fn model_state() -> ModelState {
    ModelState {
        id: "model".to_string(),
        name: "model".to_string(),
        capabilities: ModelCapabilities {
            model_class: ModelClass::DecoderOnly,
            supports_text_generation: true,
            supports_embeddings: false,
            has_chat_template: false,
            embedding: None,
        },
    }
}

fn thread_state(event_tx: mpsc::Sender<EngineEvent>) -> EngineThreadState {
    EngineThreadState {
        runtime: Some(test_runtime(NativeRuntimeConfig::default())),
        active_requests: HashMap::new(),
        model_state: model_state(),
        event_subscribers: Arc::new(Mutex::new(vec![event_tx])),
    }
}

#[test]
fn active_request_status_reports_running_only_when_requests_are_active() {
    assert_eq!(active_request_status(false), EngineStatus::Ready);
    assert_eq!(active_request_status(true), EngineStatus::Running);
}

#[test]
fn get_state_command_returns_ready_snapshot_for_idle_thread() {
    let (event_tx, _event_rx) = mpsc::channel();
    let mut state = thread_state(event_tx);
    let (response_tx, response_rx) = oneshot::channel();

    assert!(state.process_command(EngineThreadCommand::GetState(response_tx)));

    let snapshot = block_on(response_rx)
        .expect("state response")
        .expect("state ok");
    assert_eq!(snapshot.status, EngineStatus::Ready);
    assert_eq!(snapshot.model.as_ref().expect("model").id, "model");
}

#[test]
fn close_command_drops_runtime_acks_and_emits_closed_event() {
    let (event_tx, event_rx) = mpsc::channel();
    let mut state = thread_state(event_tx);
    let (ack_tx, ack_rx) = oneshot::channel();

    assert!(!state.process_command(EngineThreadCommand::Close(Some(ack_tx))));

    block_on(ack_rx)
        .expect("close ack response")
        .expect("close ack ok");
    assert!(state.runtime.is_none());
    assert!(matches!(
        event_rx.recv().expect("closed event"),
        EngineEvent::Closed
    ));
}

#[test]
fn current_state_rejects_closed_runtime() {
    let (event_tx, _event_rx) = mpsc::channel();
    let mut state = thread_state(event_tx);
    state.runtime = None;

    let error = state.current_state().expect_err("closed runtime");

    assert!(error.to_string().contains("runtime is closed"));
}

#[test]
fn generate_command_on_closed_runtime_returns_ready_error() {
    let (event_tx, _event_rx) = mpsc::channel();
    let mut state = thread_state(event_tx);
    state.runtime = None;
    let (response_tx, response_rx) = oneshot::channel();

    assert!(state.process_command(EngineThreadCommand::Generate(
        QueryRequest::new("hello"),
        response_tx,
        None,
    )));

    let error = block_on(response_rx)
        .expect("response")
        .expect_err("closed runtime");
    assert!(error.to_string().contains("runtime is closed"));
    assert!(state.active_requests.is_empty());
}

#[test]
fn embed_command_on_not_ready_runtime_returns_error_without_tracking_request() {
    let (event_tx, event_rx) = mpsc::channel();
    let mut state = thread_state(event_tx);
    let (response_tx, response_rx) = oneshot::channel();

    assert!(state.process_command(EngineThreadCommand::Embed(
        EmbedRequest {
            input: "hello".to_string(),
            options: EmbedOptions::default(),
        },
        response_tx,
    )));

    let error = block_on(response_rx)
        .expect("response")
        .expect_err("not ready runtime");
    assert!(error.to_string().contains("unsupported operation embed"));
    assert!(state.active_requests.is_empty());
    assert!(event_rx.try_recv().is_err());
}

#[test]
fn dropped_response_receiver_cancels_active_request_and_emits_failure() {
    let (event_tx, event_rx) = mpsc::channel();
    let mut state = thread_state(event_tx);
    let (response_tx, response_rx) = oneshot::channel();
    drop(response_rx);
    state.active_requests.insert(
        7,
        ActiveRequest {
            output: ActiveRequestOutput::Text,
            response_tx,
            token: None,
        },
    );

    state.step_active_requests();

    assert!(state.active_requests.is_empty());
    assert!(matches!(
        event_rx.recv().expect("failure event"),
        EngineEvent::RequestFailed { request_id, error }
            if request_id == "7" && error == "request cancelled"
    ));
}
