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
        into_c_string(serialize_json_response(
            &validate_pairing(classified_json, explicit_projector_id),
            PAIRING_SERIALIZATION_FALLBACK,
        ))
    }))
    .unwrap_or_else(|_| {
        into_c_string(serialize_json_response(
            &error_response(CODE_MODEL_BROKEN, "pairing validation panicked".to_string()),
            PAIRING_SERIALIZATION_FALLBACK,
        ))
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

#[cfg(test)]
mod tests {
    mod pairing_tests;
}
