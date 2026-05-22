use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cogentlm_shard::{
    detect_model_from_gguf_bytes, inspect_gguf_metadata, GgufError, GgufMetadataInspection,
    ModelDetection,
};
use serde::Serialize;

const CODE_INVALID_GGUF: &str = "INVALID_GGUF";
const CODE_UNSUPPORTED_GGUF_VERSION: &str = "UNSUPPORTED_GGUF_VERSION";
const CODE_GGUF_METADATA_TOO_LARGE: &str = "GGUF_METADATA_TOO_LARGE";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GgufJsonResponse<T>
where
    T: Serialize,
{
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<GgufJsonError>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GgufJsonError {
    code: &'static str,
    message: String,
}

#[no_mangle]
pub extern "C" fn cogentlm_inspect_gguf_metadata_json(
    bytes_ptr: *const u8,
    bytes_len: usize,
) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        let response = with_bytes(bytes_ptr, bytes_len, |bytes| inspect_gguf_metadata(bytes));
        into_c_string(response_json(response))
    }))
    .unwrap_or_else(|_| {
        into_c_string(response_json::<Option<GgufMetadataInspection>>(
            error_response(
                CODE_INVALID_GGUF,
                "GGUF metadata inspection panicked".to_string(),
            ),
        ))
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_detect_model_from_gguf_bytes_json(
    name: *const c_char,
    bytes_ptr: *const u8,
    bytes_len: usize,
) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(name) = read_optional_c_string(name) else {
            return into_c_string(response_json::<ModelDetection>(error_response(
                CODE_INVALID_GGUF,
                "model file name is not valid UTF-8".to_string(),
            )));
        };
        let response = with_bytes(bytes_ptr, bytes_len, |bytes| {
            detect_model_from_gguf_bytes(name, bytes)
        });
        into_c_string(response_json(response))
    }))
    .unwrap_or_else(|_| {
        into_c_string(response_json::<ModelDetection>(error_response(
            CODE_INVALID_GGUF,
            "GGUF model detection panicked".to_string(),
        )))
    })
}

fn with_bytes<T>(
    bytes_ptr: *const u8,
    bytes_len: usize,
    operation: impl FnOnce(&[u8]) -> Result<T, GgufError>,
) -> GgufJsonResponse<T>
where
    T: Serialize,
{
    if bytes_ptr.is_null() && bytes_len > 0 {
        return error_response(CODE_INVALID_GGUF, "GGUF byte pointer is null".to_string());
    }
    let bytes = if bytes_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) }
    };
    match operation(bytes) {
        Ok(value) => GgufJsonResponse {
            ok: true,
            value: Some(value),
            error: None,
        },
        Err(error) => gguf_error_response(error),
    }
}

fn gguf_error_response<T>(error: GgufError) -> GgufJsonResponse<T>
where
    T: Serialize,
{
    let code = match error {
        GgufError::Invalid(_) => CODE_INVALID_GGUF,
        GgufError::UnsupportedVersion(_) => CODE_UNSUPPORTED_GGUF_VERSION,
        GgufError::MetadataTooLarge { .. } => CODE_GGUF_METADATA_TOO_LARGE,
        GgufError::Io(_) | GgufError::AlreadySplit(_) => CODE_INVALID_GGUF,
    };
    error_response(code, error.to_string())
}

fn error_response<T>(code: &'static str, message: String) -> GgufJsonResponse<T>
where
    T: Serialize,
{
    GgufJsonResponse {
        ok: false,
        value: None,
        error: Some(GgufJsonError { code, message }),
    }
}

fn response_json<T>(response: GgufJsonResponse<T>) -> String
where
    T: Serialize,
{
    serde_json::to_string(&response).unwrap_or_else(|_| {
        "{\"ok\":false,\"error\":{\"code\":\"INVALID_GGUF\",\"message\":\"failed to serialize GGUF response\"}}".to_string()
    })
}

fn read_optional_c_string(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return Some(String::new());
    }
    Some(
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn into_c_string(value: String) -> *mut c_char {
    let sanitized = value.replace('\0', "");
    CString::new(sanitized)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_gguf_detection_returns_unknown_model() {
        let name = CString::new("bad.bin").expect("name");

        let ptr =
            cogentlm_detect_model_from_gguf_bytes_json(name.as_ptr(), b"not-a-gguf".as_ptr(), 10);
        let response = read_response(ptr);

        assert_eq!(response["ok"], true);
        assert_eq!(response["value"]["inspection"]["role"], "unknown");
        assert_eq!(response["value"]["detectionMethod"], "none");
    }

    #[test]
    fn truncated_gguf_metadata_returns_typed_error() {
        let bytes = [
            b'G', b'G', b'U', b'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        ];

        let ptr = cogentlm_inspect_gguf_metadata_json(bytes.as_ptr(), bytes.len());
        let response = read_response(ptr);

        assert_eq!(response["ok"], false);
        assert_eq!(response["error"]["code"], CODE_INVALID_GGUF);
    }

    fn read_response(ptr: *mut c_char) -> serde_json::Value {
        assert!(!ptr.is_null());
        let raw = unsafe { CString::from_raw(ptr) };
        serde_json::from_slice(raw.as_bytes()).expect("response json")
    }
}
