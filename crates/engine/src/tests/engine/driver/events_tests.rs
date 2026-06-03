//! Tests the `engine::driver::events` module in `cogentlm-engine`.
//!
//! Covers driver futures, command handling, event emission, and request mapping with model-free channels or explicitly ignored model smoke tests.

use std::sync::{mpsc, Arc, Mutex};

use crate::engine::protocol::{
    EngineEvent, EngineStatus, ModelCapabilities, ModelClass, ModelState,
};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;

use super::{build_engine_state_with_status, emit_event};

#[test]
fn emit_event_drops_closed_subscribers() {
    let subscribers = Arc::new(Mutex::new(Vec::new()));
    let (closed_tx, closed_rx) = mpsc::channel();
    drop(closed_rx);
    let (open_tx, open_rx) = mpsc::channel();
    subscribers.lock().unwrap().push(closed_tx);
    subscribers.lock().unwrap().push(open_tx);

    emit_event(&subscribers, EngineEvent::Closed);

    assert!(matches!(open_rx.recv().unwrap(), EngineEvent::Closed));
    assert_eq!(subscribers.lock().unwrap().len(), 1);
}

#[test]
fn build_engine_state_uses_explicit_status_and_model_snapshot() {
    let runtime = test_runtime(NativeRuntimeConfig::default());
    let model = ModelState {
        id: "model-id".to_string(),
        name: "model-name".to_string(),
        capabilities: ModelCapabilities {
            model_class: ModelClass::DecoderOnly,
            supports_text_generation: true,
            supports_embeddings: false,
            has_chat_template: false,
            embedding: None,
        },
    };

    let state = build_engine_state_with_status(&runtime, &model, Some(EngineStatus::Running));

    assert_eq!(state.status, EngineStatus::Running);
    assert_eq!(state.model.as_ref().expect("model").id, "model-id");
    assert!(state.runtime.is_some());
}
