use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use serde_json::Value;
use sipp::lifecycle::{
    browser_lifecycle_error_response, browser_lifecycle_response_json,
    browser_lifecycle_success_response, BrowserCommitLoadRequest, BrowserCreateConfig,
    BrowserInstallSource, BrowserLifecycleEnvelope, BrowserLifecycleService, BrowserLoadOptions,
    BrowserLoadSource, BrowserObservabilityEventType, BrowserRemoteCommand, BrowserRemoveRequest,
    ModelError,
};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLifecycleCreateResponse {
    handle: usize,
    manifest: sipp::lifecycle::BrowserRegistryManifest,
    snapshot: sipp::lifecycle::BrowserObservabilitySnapshot,
}

static NEXT_SERVICE_ID: AtomicUsize = AtomicUsize::new(1);
static SERVICES: OnceLock<Mutex<HashMap<usize, BrowserLifecycleService>>> = OnceLock::new();

pub(crate) fn model_service_create_json(config_json: &str) -> String {
    let response = catch_unwind(AssertUnwindSafe(|| create_service(config_json)))
        .unwrap_or_else(|_| browser_lifecycle_error_response(lifecycle_panic_error("creation")));
    browser_lifecycle_response_json(response)
}

pub(crate) fn model_service_close(service: usize) -> i32 {
    if service == 0 {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let Ok(mut services) = services().lock() else {
            return 0;
        };
        let Some(mut service) = services.remove(&service) else {
            return 0;
        };
        let _ = service.close();
        1
    }))
    .unwrap_or(0)
}

pub(crate) fn model_service_list_json(service: usize) -> String {
    service_response(service, |service| service.list())
}

pub(crate) fn model_service_manifest_json(service: usize) -> String {
    service_response(service, |service| service.manifest.clone())
}

pub(crate) fn model_service_prepare_load_json(
    service: usize,
    source_json: &str,
    options_json: &str,
) -> String {
    service_result_response(service, |service| {
        let source = parse_json_arg::<BrowserLoadSource>(source_json)?;
        let options = parse_json_arg::<BrowserLoadOptions>(options_json)?;
        service.prepare_load(source, options)
    })
}

pub(crate) fn model_service_install_json(service: usize, source_json: &str) -> String {
    service_result_response(service, |service| {
        let source = parse_json_arg::<BrowserInstallSource>(source_json)?;
        service.install(source)
    })
}

pub(crate) fn model_service_remote_acquisition_command_json(
    service: usize,
    command_json: &str,
) -> String {
    service_result_response(service, |service| {
        let command = parse_json_arg::<BrowserRemoteCommand>(command_json)?;
        service.remote_command(command)
    })
}

pub(crate) fn model_service_commit_load_json(service: usize, commit_json: &str) -> String {
    service_result_response(service, |service| {
        let request = parse_json_arg::<BrowserCommitLoadRequest>(commit_json)?;
        service.commit_load(request)
    })
}

pub(crate) fn model_service_remove_json(service: usize, request_json: &str) -> String {
    service_result_response(service, |service| {
        let request = parse_json_arg::<BrowserRemoveRequest>(request_json)?;
        service.remove(request)
    })
}

pub(crate) fn model_service_snapshot_json(service: usize) -> String {
    service_response(service, |service| service.snapshot.clone())
}

pub(crate) fn model_service_drain_events_json(service: usize) -> String {
    service_response(service, |service| service.drain_events())
}

pub(crate) fn model_service_record_event_json(
    service: usize,
    event_type: &str,
    patch_json: &str,
) -> String {
    service_result_response(service, |service| {
        let event_type = serde_json::from_value::<BrowserObservabilityEventType>(Value::String(
            event_type.to_string(),
        ))
        .map_err(ModelError::from)?;
        let patch = parse_json_arg::<Value>(patch_json)?;
        service.record_event(event_type, patch)
    })
}

fn create_service(config_json: &str) -> BrowserLifecycleEnvelope<BrowserLifecycleCreateResponse> {
    let config = match parse_json_arg::<BrowserCreateConfig>(config_json) {
        Ok(config) => config,
        Err(error) => return browser_lifecycle_error_response(error),
    };
    match BrowserLifecycleService::create(config) {
        Ok(service) => {
            let handle = NEXT_SERVICE_ID.fetch_add(1, Ordering::Relaxed);
            let manifest = service.manifest.clone();
            let snapshot = service.snapshot.clone();
            let Ok(mut services) = services().lock() else {
                return browser_lifecycle_error_response(registry_unavailable_error());
            };
            services.insert(handle, service);
            browser_lifecycle_success_response(BrowserLifecycleCreateResponse {
                handle,
                manifest,
                snapshot,
            })
        }
        Err(error) => browser_lifecycle_error_response(error),
    }
}

fn service_response<T>(
    service: usize,
    operation: impl FnOnce(&mut BrowserLifecycleService) -> T,
) -> String
where
    T: Serialize,
{
    service_result_response(service, |service| Ok(operation(service)))
}

fn service_result_response<T>(
    service: usize,
    operation: impl FnOnce(&mut BrowserLifecycleService) -> Result<T, ModelError>,
) -> String
where
    T: Serialize,
{
    let response = catch_unwind(AssertUnwindSafe(|| {
        let Ok(mut services) = services().lock() else {
            return browser_lifecycle_error_response(registry_unavailable_error());
        };
        match services.get_mut(&service) {
            Some(service) => match operation(service) {
                Ok(value) => browser_lifecycle_success_response(value),
                Err(error) => browser_lifecycle_error_response(error),
            },
            None => browser_lifecycle_error_response(ModelError::StorageUnavailable(
                "browser lifecycle service handle is missing".to_string(),
            )),
        }
    }))
    .unwrap_or_else(|_| browser_lifecycle_error_response(lifecycle_panic_error("operation")));
    browser_lifecycle_response_json(response)
}

fn services() -> &'static Mutex<HashMap<usize, BrowserLifecycleService>> {
    SERVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registry_unavailable_error() -> ModelError {
    ModelError::StorageCorrupt("browser lifecycle service registry is unavailable".to_string())
}

fn lifecycle_panic_error(operation: &'static str) -> ModelError {
    ModelError::StorageCorrupt(format!("browser lifecycle service {operation} panicked"))
}

fn parse_json_arg<T>(value: &str) -> Result<T, ModelError>
where
    T: for<'de> serde::Deserialize<'de>,
{
    serde_json::from_str::<T>(value).map_err(ModelError::from)
}

#[cfg(test)]
#[path = "tests/lifecycle_tests.rs"]
mod lifecycle_tests;
