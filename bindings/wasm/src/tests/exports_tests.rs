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
fn session_mutation_does_not_hold_refcell_borrow() {
    CURRENT_ENGINE.with(|current| {
        *current.borrow_mut() = Some(Box::new(BrowserEngine::create()));
    });
    let generation = CURRENT_ENGINE.with(|current| current.borrow().as_ref().expect("engine").id());

    let can_borrow_inside_operation = with_session_engine_mut(generation, false, |_| {
        CURRENT_ENGINE.with(|current| current.try_borrow_mut().is_ok())
    });

    assert!(can_borrow_inside_operation);
    assert!(current_engine_initialized());
    close_current_engine();
}

#[test]
fn runtime_session_export_rejects_an_empty_runtime() {
    close_current_engine();

    let session = CE_GetRuntimeSessionJson();

    assert!(session.is_null());
    LAST_ENGINE_ERROR.with(|error| {
        assert_eq!(
            error.borrow().as_str(),
            "browser runtime session is not loaded"
        );
    });
}

#[test]
fn invalid_activation_does_not_destroy_the_current_engine() {
    let current = BrowserEngine::create();
    let current_id = current.id();
    CURRENT_ENGINE.with(|engine| {
        *engine.borrow_mut() = Some(Box::new(current));
    });
    let model_path = CString::new("/models/model.gguf").expect("model path");
    let runtime = CString::new("{}").expect("runtime");
    let session = CString::new("{}").expect("session");

    let status = unsafe { CE_Init(model_path.as_ptr(), runtime.as_ptr(), session.as_ptr()) };

    assert_eq!(status, STATUS_INVALID_ARGUMENTS);
    CURRENT_ENGINE.with(|engine| {
        assert_eq!(
            engine.borrow().as_ref().map(|engine| engine.id()),
            Some(current_id)
        );
    });
    close_current_engine();
}

#[test]
fn stale_request_generation_does_not_touch_the_current_engine() {
    let current = BrowserEngine::create();
    let current_id = current.id();
    CURRENT_ENGINE.with(|engine| {
        *engine.borrow_mut() = Some(Box::new(current));
    });

    let status = CE_GetCompletedRequestStatus(current_id.wrapping_add(1), 1);

    assert_eq!(status, STATUS_STALE_RUNTIME_SESSION);
    assert!(LAST_ENGINE_ERROR.with(|error| error.borrow().contains("is stale")));
    CURRENT_ENGINE.with(|engine| {
        assert_eq!(
            engine.borrow().as_ref().map(|engine| engine.id()),
            Some(current_id)
        );
    });
    close_current_engine();
}

#[test]
fn closing_the_engine_keeps_the_catalog_service_alive() {
    let config = CString::new("{}").expect("config");
    let created = take_response(unsafe { CE_ModelServiceCreate(config.as_ptr()) });
    let created: Value = serde_json::from_str(&created).expect("create response");
    let service = created["value"]["handle"].as_u64().expect("service handle") as usize;
    CURRENT_ENGINE.with(|current| {
        *current.borrow_mut() = Some(Box::new(BrowserEngine::create()));
    });

    CE_Close();
    let listed = take_response(CE_ModelServiceList(service));

    assert!(listed.contains("\"ok\":true"));
    assert!(!current_engine_initialized());
    assert_eq!(CE_ModelServiceClose(service), 1);
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
    assert!(response.contains("\"INVALID_MODEL_SOURCE\""));
    assert!(response.contains("no model assets were provided"));
    assert_eq!(CE_ModelServiceClose(service), 1);
}
