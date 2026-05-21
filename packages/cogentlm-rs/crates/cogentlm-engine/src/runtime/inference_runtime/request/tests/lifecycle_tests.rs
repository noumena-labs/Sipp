use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::tests::runtime_tests::test_runtime;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::SlotPhase;
use crate::runtime::session::SequenceState;

#[test]
fn cancel_request_marks_active_slot_clone() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut request = GenerateRequest::new(7, "ctx");
    request.prompt_tokens = vec![1, 2, 3];
    assert!(runtime.request_queue.push(request.clone()));

    runtime.slot_scheduler.resize(1);
    runtime.slot_scheduler.mutable_slots()[0].attach_request(request, SequenceState::default());
    runtime.slot_scheduler.mutable_slots()[0].phase = SlotPhase::Decode;

    assert!(runtime.cancel_request(7));

    assert!(runtime
        .request_queue
        .find(7)
        .is_some_and(|request| request.cancel_requested));
    assert!(runtime.slot_scheduler.slots()[0]
        .request()
        .is_some_and(|request| request.cancel_requested));
}
