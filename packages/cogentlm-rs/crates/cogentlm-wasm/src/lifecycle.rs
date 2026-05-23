use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cogentlm_engine::lifecycle::{
    browser_lifecycle_error_response, browser_lifecycle_response_json,
    browser_lifecycle_success_response, BrowserCommitLoadRequest, BrowserCreateConfig,
    BrowserLifecycleEnvelope, BrowserLifecycleService, BrowserLoadOptions, BrowserLoadSource,
    BrowserObservabilityEventType, ModelError,
};
use serde::Serialize;
use serde_json::Value;

use crate::ffi::{into_c_string, read_optional_c_string};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLifecycleCreateResponse {
    handle: usize,
    manifest: cogentlm_engine::lifecycle::BrowserRegistryManifest,
    snapshot: cogentlm_engine::lifecycle::BrowserObservabilitySnapshot,
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_create_json(config_json: *const c_char) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        into_c_string(browser_lifecycle_response_json(create_service(config_json)))
    }))
    .unwrap_or_else(|_| {
        into_c_string(browser_lifecycle_response_json::<Value>(
            browser_lifecycle_error_response(ModelError::StorageCorrupt(
                "browser lifecycle service creation panicked".to_string(),
            )),
        ))
    })
}

#[no_mangle]
/// # Safety
/// `service` must be null or a live handle returned by
/// `cogentlm_model_service_create_json`. A non-null handle is consumed and must
/// not be reused.
pub unsafe extern "C" fn cogentlm_model_service_close(
    service: *mut BrowserLifecycleService,
) -> i32 {
    if service.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe {
        let mut service = Box::from_raw(service);
        let _ = service.close();
        1
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_list_json(
    service: *mut BrowserLifecycleService,
) -> *mut c_char {
    service_response(service, |service| service.list())
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_current_json(
    service: *mut BrowserLifecycleService,
) -> *mut c_char {
    service_response(service, |service| service.current())
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_manifest_json(
    service: *mut BrowserLifecycleService,
) -> *mut c_char {
    service_response(service, |service| service.manifest.clone())
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_prepare_load_json(
    service: *mut BrowserLifecycleService,
    source_json: *const c_char,
    options_json: *const c_char,
) -> *mut c_char {
    service_result_response(service, |service| {
        let source = parse_json_arg::<BrowserLoadSource>(source_json, "model source")?;
        let options = parse_json_arg::<BrowserLoadOptions>(options_json, "load options")?;
        service.prepare_load(source, options)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_commit_load_json(
    service: *mut BrowserLifecycleService,
    commit_json: *const c_char,
) -> *mut c_char {
    service_result_response(service, |service| {
        let request = parse_json_arg::<BrowserCommitLoadRequest>(commit_json, "load commit")?;
        service.commit_load(request)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_abort_load_json(
    service: *mut BrowserLifecycleService,
    error_json: *const c_char,
) -> *mut c_char {
    service_response(service, |service| {
        let message = read_optional_c_string(error_json)
            .filter(|value| !value.trim().is_empty())
            .and_then(|value| {
                serde_json::from_str::<Value>(&value)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("message")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .or(Some(value))
            });
        service.abort_load(message)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_remove_json(
    service: *mut BrowserLifecycleService,
    model_id: *const c_char,
) -> *mut c_char {
    service_result_response(service, |service| {
        let model_id = read_required_c_string(model_id, "model id")?;
        service.remove(model_id.trim())
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_unload_json(
    service: *mut BrowserLifecycleService,
) -> *mut c_char {
    service_response(service, |service| service.unload())
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_snapshot_json(
    service: *mut BrowserLifecycleService,
) -> *mut c_char {
    service_response(service, |service| service.snapshot.clone())
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_drain_events_json(
    service: *mut BrowserLifecycleService,
) -> *mut c_char {
    service_response(service, |service| service.drain_events())
}

#[no_mangle]
pub extern "C" fn cogentlm_model_service_record_event_json(
    service: *mut BrowserLifecycleService,
    event_type: *const c_char,
    patch_json: *const c_char,
) -> *mut c_char {
    service_result_response(service, |service| {
        let event_type = read_required_c_string(event_type, "event type")?;
        let event_type =
            serde_json::from_value::<BrowserObservabilityEventType>(Value::String(event_type))
                .map_err(ModelError::from)?;
        let patch = parse_json_arg::<Value>(patch_json, "event patch")?;
        service.record_event(event_type, patch)
    })
}

fn create_service(
    config_json: *const c_char,
) -> BrowserLifecycleEnvelope<BrowserLifecycleCreateResponse> {
    let config = match parse_json_arg::<BrowserCreateConfig>(config_json, "service config") {
        Ok(config) => config,
        Err(error) => return browser_lifecycle_error_response(error),
    };
    match BrowserLifecycleService::create(config) {
        Ok(service) => {
            let service = Box::new(service);
            let handle = Box::into_raw(service);
            let service_ref = unsafe { &*handle };
            browser_lifecycle_success_response(BrowserLifecycleCreateResponse {
                handle: handle as usize,
                manifest: service_ref.manifest.clone(),
                snapshot: service_ref.snapshot.clone(),
            })
        }
        Err(error) => browser_lifecycle_error_response(error),
    }
}

fn service_response<T>(
    service: *mut BrowserLifecycleService,
    operation: impl FnOnce(&mut BrowserLifecycleService) -> T,
) -> *mut c_char
where
    T: Serialize,
{
    service_result_response(service, |service| Ok(operation(service)))
}

fn service_result_response<T>(
    service: *mut BrowserLifecycleService,
    operation: impl FnOnce(&mut BrowserLifecycleService) -> Result<T, ModelError>,
) -> *mut c_char
where
    T: Serialize,
{
    catch_unwind(AssertUnwindSafe(|| {
        let response = match service_mut(service) {
            Ok(service) => match operation(service) {
                Ok(value) => browser_lifecycle_success_response(value),
                Err(error) => browser_lifecycle_error_response(error),
            },
            Err(error) => browser_lifecycle_error_response(error),
        };
        into_c_string(browser_lifecycle_response_json(response))
    }))
    .unwrap_or_else(|_| {
        into_c_string(browser_lifecycle_response_json::<Value>(
            browser_lifecycle_error_response(ModelError::StorageCorrupt(
                "browser lifecycle service operation panicked".to_string(),
            )),
        ))
    })
}

fn service_mut(
    service: *mut BrowserLifecycleService,
) -> Result<&'static mut BrowserLifecycleService, ModelError> {
    if service.is_null() {
        return Err(ModelError::StorageUnavailable(
            "browser lifecycle service handle is null".to_string(),
        ));
    }
    Ok(unsafe { &mut *service })
}

fn parse_json_arg<T>(value: *const c_char, label: &str) -> Result<T, ModelError>
where
    T: for<'de> serde::Deserialize<'de>,
{
    let text = read_required_c_string(value, label)?;
    serde_json::from_str::<T>(&text).map_err(ModelError::from)
}

fn read_required_c_string(value: *const c_char, label: &str) -> Result<String, ModelError> {
    let value = read_optional_c_string(value)
        .ok_or_else(|| ModelError::InvalidModelSource(format!("{label} is not valid UTF-8")))?;
    if value.trim().is_empty() {
        return Err(ModelError::InvalidModelSource(format!("{label} is empty")));
    }
    Ok(value)
}
