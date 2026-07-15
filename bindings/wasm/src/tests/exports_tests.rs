//! Tests the WASM `exports` module.
//!
//! Covers engine borrow scope and owned-string error envelopes through
//! in-process calls without model loading or browser host callbacks.

use super::*;

#[test]
fn mutable_engine_operation_does_not_hold_refcell_borrow() {
    CURRENT_ENGINE.with(|current| {
        *current.borrow_mut() = Some(Box::new(BrowserEngine::create()));
    });

    let can_borrow_inside_operation = with_current_engine_mut(false, |_| {
        CURRENT_ENGINE.with(|current| current.try_borrow_mut().is_ok())
    });

    assert!(can_borrow_inside_operation);
    assert!(current_engine_initialized());
    close_current_engine();
}

#[test]
fn remote_acquisition_export_rejects_missing_command_json() {
    let response = unsafe { CE_ModelServiceRemoteAcquisitionCommand(0, ptr::null()) };
    assert!(!response.is_null());

    let response_json = unsafe { CStr::from_ptr(response).to_string_lossy().into_owned() };
    unsafe { CE_FreeString(response) };

    assert!(response_json.contains("\"ok\":false"));
    assert!(response_json.contains("\"INVALID_MODEL_SOURCE\""));
    assert!(response_json.contains("remote acquisition command JSON is missing"));
}
