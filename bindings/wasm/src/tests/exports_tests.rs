//! Tests the WASM `exports` module.
//!
//! Covers engine borrow scope and owned-string error envelopes through
//! in-process calls without model loading or browser host callbacks.

use super::*;

fn take_response(response: *mut c_char) -> String {
    assert!(!response.is_null());
    let response_json = unsafe { CStr::from_ptr(response).to_string_lossy().into_owned() };
    unsafe { CE_FreeString(response) };
    response_json
}

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
    let response_json =
        take_response(unsafe { CE_ModelServiceRemoteAcquisitionCommand(0, ptr::null()) });

    assert!(response_json.contains("\"ok\":false"));
    assert!(response_json.contains("\"INVALID_MODEL_SOURCE\""));
    assert!(response_json.contains("remote acquisition command JSON is missing"));
}

#[test]
fn model_install_export_rejects_missing_source_json() {
    let response_json = take_response(unsafe { CE_ModelServiceInstall(0, ptr::null()) });

    assert!(response_json.contains("\"ok\":false"));
    assert!(response_json.contains("\"INVALID_MODEL_SOURCE\""));
    assert!(response_json.contains("install source JSON is missing"));
}

#[test]
fn model_install_export_invokes_the_lifecycle_service() {
    let config = CString::new("{}").expect("config");
    let created = take_response(unsafe { CE_ModelServiceCreate(config.as_ptr()) });
    let created: Value = serde_json::from_str(&created).expect("create response");
    let service = created["value"]["handle"].as_u64().expect("service handle") as usize;

    let source = CString::new(r#"{"assets":[],"classified":[]}"#).expect("source");
    let response = take_response(unsafe { CE_ModelServiceInstall(service, source.as_ptr()) });

    assert!(response.contains("\"ok\":false"));
    assert!(response.contains("\"INVALID_MODEL_PAIRING\""));
    assert_eq!(CE_ModelServiceClose(service), 1);
}
