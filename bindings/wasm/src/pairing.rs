use serde::Serialize;
use sipp::lifecycle::{ClassifiedAsset, ModelError, PairingPlan, PairingResolver};

use crate::ffi::serialize_json_response;

const CODE_INVALID_MODEL_SOURCE: &str = "INVALID_MODEL_SOURCE";
const CODE_INVALID_MODEL_PAIRING: &str = "INVALID_MODEL_PAIRING";
const CODE_MODEL_BROKEN: &str = "MODEL_BROKEN";
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

pub(crate) fn pairing_validate_json(classified_json: &str) -> String {
    serialize_json_response(&validate_pairing(classified_json))
}

fn validate_pairing(raw_classified: &str) -> PairingValidateResponse {
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

    match PairingResolver::resolve(&classified) {
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
#[path = "tests/pairing_tests.rs"]
mod pairing_tests;
