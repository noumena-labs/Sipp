use std::ffi::CString;
use std::os::raw::c_char;

use super::super::*;

fn classified_json() -> CString {
    CString::new(
        r#"[
            {
                "assetId":"asset-model",
                "name":"base.gguf",
                "inspection":{
                    "version":1,
                    "role":"model",
                    "architecture":"lfm2",
                    "visionCapable":true,
                    "compatibleVisionProjectorTypes":["lfm2"],
                    "providedVisionProjectorType":null
                }
            },
            {
                "assetId":"asset-projector",
                "name":"mmproj.gguf",
                "inspection":{
                    "version":1,
                    "role":"projector",
                    "architecture":"clip",
                    "visionCapable":false,
                    "compatibleVisionProjectorTypes":[],
                    "providedVisionProjectorType":"lfm2"
                }
            }
        ]"#,
    )
    .expect("classified json")
}

fn read_owned(ptr: *mut c_char) -> serde_json::Value {
    assert!(!ptr.is_null());
    let raw = unsafe { CString::from_raw(ptr) };
    let text = raw.to_string_lossy().into_owned();
    serde_json::from_str(&text).expect("response json")
}

#[test]
fn validates_explicit_projector_pairing() {
    let classified = classified_json();
    let explicit = CString::new("asset-projector").expect("projector id");

    let response = read_owned(cogentlm_pairing_validate_json(
        classified.as_ptr(),
        explicit.as_ptr(),
    ));

    assert_eq!(response["ok"], true);
    assert_eq!(response["plan"]["projectorAssetId"], "asset-projector");
    assert_eq!(response["plan"]["status"], "ready");
}

#[test]
fn returns_typed_error_for_invalid_json() {
    let invalid = CString::new("{").expect("invalid json");

    let response = read_owned(cogentlm_pairing_validate_json(
        invalid.as_ptr(),
        std::ptr::null(),
    ));

    assert_eq!(response["ok"], false);
    assert_eq!(response["error"]["code"], CODE_INVALID_MODEL_SOURCE);
}
