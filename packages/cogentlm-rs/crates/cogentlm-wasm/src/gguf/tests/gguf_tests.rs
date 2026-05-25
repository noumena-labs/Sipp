use std::ffi::CString;
use std::os::raw::c_char;

use super::super::*;

#[test]
fn non_gguf_detection_returns_unknown_model() {
    let name = CString::new("bad.bin").expect("name");

    let ptr = cogentlm_detect_model_from_gguf_bytes_json(name.as_ptr(), b"not-a-gguf".as_ptr(), 10);
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
