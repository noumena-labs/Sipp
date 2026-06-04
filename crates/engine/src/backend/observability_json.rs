use serde_json::Value;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/backend/observability_json_tests.rs"]
mod observability_json_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) const KEY_AVAILABLE_BACKENDS: &str = "availableBackends";
pub(crate) const KEY_COMPILED: &str = "compiled";
pub(crate) const KEY_DEVICE_ID: &str = "deviceId";
pub(crate) const KEY_DEVICES: &str = "devices";
pub(crate) const KEY_DYNAMIC_BACKEND_LOADING: &str = "dynamicBackendLoading";
pub(crate) const KEY_GPU_OFFLOAD_SUPPORTED: &str = "gpuOffloadSupported";
pub(crate) const KEY_MEMORY_FREE_BYTES: &str = "memoryFreeBytes";
pub(crate) const KEY_MEMORY_TOTAL_BYTES: &str = "memoryTotalBytes";
pub(crate) const KEY_NAME: &str = "name";
pub(crate) const KEY_TYPE: &str = "type";

pub(crate) const DEVICE_TYPE_GPU: &str = "GPU";
pub(crate) const DEVICE_TYPE_IGPU: &str = "IGPU";

pub(crate) fn json_array<'value>(value: &'value Value, key: &str) -> Option<&'value [Value]> {
    value.get(key).and_then(Value::as_array).map(Vec::as_slice)
}

pub(crate) fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

pub(crate) fn json_str<'value>(value: &'value Value, key: &str) -> Option<&'value str> {
    value.get(key).and_then(Value::as_str)
}

pub(crate) fn json_string_or(value: &Value, key: &str, fallback: &str) -> String {
    json_str(value, key).unwrap_or(fallback).to_string()
}

pub(crate) fn json_strings(items: &[Value], key: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if let Some(value) = json_str(item, key) {
            out.push(value.to_string());
        }
    }
    out
}

pub(crate) fn json_array_strings(value: &Value, array_key: &str, item_key: &str) -> Vec<String> {
    json_array(value, array_key).map_or_else(Vec::new, |items| json_strings(items, item_key))
}

pub(crate) fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}
