use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cogentlm_engine::lifecycle::{ClassifiedAsset, ModelError, PairingPlan, PairingResolver};
use serde::Serialize;

use crate::ffi::{into_c_string, read_optional_c_string, serialize_json_response};

const CODE_INVALID_MODEL_SOURCE: &str = "INVALID_MODEL_SOURCE";
const CODE_INVALID_MODEL_PAIRING: &str = "INVALID_MODEL_PAIRING";
const CODE_MODEL_BROKEN: &str = "MODEL_BROKEN";
const PAIRING_SERIALIZATION_FALLBACK: &str =
    "{\"ok\":false,\"error\":{\"code\":\"MODEL_BROKEN\",\"message\":\"failed to serialize pairing validation response\"}}";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PairingValidateResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<PairingPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<PairingValidateError>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PairingValidateError {
    code: &'static str,
    message: String,
}

#[no_mangle]
pub extern "C" fn cogentlm_pairing_validate_json(
    classified_json: *const c_char,
    explicit_projector_id: *const c_char,
) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        into_c_string(response_json(validate_pairing(
            classified_json,
            explicit_projector_id,
        )))
    }))
    .unwrap_or_else(|_| {
        into_c_string(response_json(error_response(
            CODE_MODEL_BROKEN,
            "pairing validation panicked".to_string(),
        )))
    })
}

fn validate_pairing(
    classified_json: *const c_char,
    explicit_projector_id: *const c_char,
) -> PairingValidateResponse {
    let Some(raw_classified) = read_optional_c_string(classified_json) else {
        return error_response(
            CODE_INVALID_MODEL_SOURCE,
            "classified asset JSON is not valid UTF-8".to_string(),
        );
    };
    if raw_classified.trim().is_empty() {
        return error_response(
            CODE_INVALID_MODEL_SOURCE,
            "classified asset JSON is empty".to_string(),
        );
    }

    let classified = match serde_json::from_str::<Vec<ClassifiedAsset>>(raw_classified.trim()) {
        Ok(classified) => classified,
        Err(error) => {
            return error_response(
                CODE_INVALID_MODEL_SOURCE,
                format!("classified asset JSON is invalid: {error}"),
            );
        }
    };

    let explicit_projector_id = read_optional_c_string(explicit_projector_id)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let result = match explicit_projector_id.as_deref() {
        Some(projector_id) => PairingResolver::resolve_explicit(&classified, projector_id),
        None => PairingResolver::resolve(&classified),
    };

    match result {
        Ok(plan) => PairingValidateResponse {
            ok: true,
            plan: Some(plan),
            error: None,
        },
        Err(error) => model_error_response(error),
    }
}

fn model_error_response(error: ModelError) -> PairingValidateResponse {
    let code = match error {
        ModelError::InvalidModelSource(_) => CODE_INVALID_MODEL_SOURCE,
        ModelError::InvalidModelPairing(_) => CODE_INVALID_MODEL_PAIRING,
        _ => CODE_MODEL_BROKEN,
    };
    error_response(code, error.to_string())
}

fn error_response(code: &'static str, message: String) -> PairingValidateResponse {
    PairingValidateResponse {
        ok: false,
        plan: None,
        error: Some(PairingValidateError { code, message }),
    }
}

fn response_json(response: PairingValidateResponse) -> String {
    serialize_json_response(&response, PAIRING_SERIALIZATION_FALLBACK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

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
}
