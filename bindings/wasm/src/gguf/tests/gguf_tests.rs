use super::super::*;

#[test]
fn non_gguf_detection_returns_unknown_model() {
    let response = read_response(detect_model_from_gguf_bytes_json("bad.bin", b"not-a-gguf"));

    assert_eq!(response["ok"], true);
    assert_eq!(response["value"]["inspection"]["role"], "unknown");
    assert_eq!(response["value"]["detectionMethod"], "none");
}

#[test]
fn truncated_gguf_metadata_returns_typed_error() {
    let bytes = [
        b'G', b'G', b'U', b'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    ];

    let response = read_response(inspect_gguf_metadata_json(&bytes));

    assert_eq!(response["ok"], false);
    assert_eq!(response["error"]["code"], CODE_INVALID_GGUF);
}

fn read_response(text: String) -> serde_json::Value {
    serde_json::from_str(&text).expect("response json")
}
