//! Tests the `runtime::inference_runtime::request::lifecycle` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::GenerateRequest;
use crate::runtime::request::GenerateResponse;
use crate::runtime::scheduler::SlotPhase;
use crate::runtime::session::KvCacheAdmission;

#[test]
fn cancel_request_marks_active_slot_clone() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut request = GenerateRequest::new(7, "ctx");
    request.prompt_tokens = vec![1, 2, 3];
    assert!(runtime.request_queue.push(request.clone()));

    runtime.slot_scheduler.resize(1, &mut runtime.kv_cache);
    runtime.slot_scheduler.slots[0].attach_request(request, KvCacheAdmission::default());
    runtime.slot_scheduler.slots[0].phase = SlotPhase::Decode;

    assert!(runtime.cancel_request(7));

    assert!(runtime
        .request_queue
        .requests
        .get(&7)
        .is_some_and(|request| request.cancel_requested));
    assert!(runtime.slot_scheduler.slots[0]
        .request()
        .is_some_and(|request| request.cancel_requested));
}

#[test]
fn take_completed_response_removes_observability_bookkeeping() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.committed_observability_request_ids.insert(7);
    runtime.request_queue.completed_responses.insert(
        7,
        GenerateResponse {
            request_id: 7,
            ..GenerateResponse::default()
        },
    );

    let response = runtime
        .take_completed_response(7)
        .expect("completed response");

    assert_eq!(response.request_id, 7);
    assert!(!runtime.committed_observability_request_ids.contains(&7));
    assert!(runtime.request_queue.completed_responses.is_empty());
}
