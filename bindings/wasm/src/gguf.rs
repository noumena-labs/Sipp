#[cfg(test)]
use cogentlm_shard::inspect_gguf_metadata;
use cogentlm_shard::{detect_model_from_gguf_bytes, GgufError};
use serde::Serialize;

use crate::ffi::serialize_json_response;

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

#[cfg(test)]
pub(crate) fn inspect_gguf_metadata_json(bytes: &[u8]) -> String {
    let response = with_bytes(bytes, inspect_gguf_metadata);
    serialize_json_response(&response)
}

pub(crate) fn detect_model_from_gguf_bytes_json(name: &str, bytes: &[u8]) -> String {
    let response = with_bytes(bytes, |bytes| detect_model_from_gguf_bytes(name, bytes));
    serialize_json_response(&response)
}

fn with_bytes<T>(
    bytes: &[u8],
    operation: impl FnOnce(&[u8]) -> Result<T, GgufError>,
) -> GgufJsonResponse<T>
where
    T: Serialize,
{
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

#[cfg(test)]
#[path = "tests/gguf_tests.rs"]
mod gguf_tests;
