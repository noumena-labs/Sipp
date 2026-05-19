//! Numeric coercions and Rust-error → napi-error mapping for the bindings.

use napi::{Error, Result, Status};

use super::{JS_MAX_SAFE_INTEGER_F64, JS_MAX_SAFE_INTEGER_U64};

pub(super) fn invalid_arg(message: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, message.into())
}

pub(super) fn i64_to_u32(value: i64, field: &'static str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_arg(format!("{field} must fit in an unsigned 32-bit integer")))
}

pub(super) fn f64_to_f32(value: f64, field: &'static str) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(invalid_arg(format!(
            "{field} must be a finite 32-bit float"
        )));
    }
    Ok(value as f32)
}

pub(super) fn optional_f64_to_f32(value: Option<f64>, field: &'static str) -> Result<Option<f32>> {
    value.map(|value| f64_to_f32(value, field)).transpose()
}

pub(super) fn u64_to_js_safe_number(value: u64) -> f64 {
    value.min(JS_MAX_SAFE_INTEGER_U64) as f64
}

pub(super) fn i64_to_js_safe_number(value: i64) -> f64 {
    if value <= 0 {
        return 0.0;
    }
    u64::try_from(value)
        .map(u64_to_js_safe_number)
        .unwrap_or(JS_MAX_SAFE_INTEGER_F64)
}

pub(super) fn finite_nonnegative_f64_to_u64(value: f64, field: &'static str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > JS_MAX_SAFE_INTEGER_F64
    {
        return Err(invalid_arg(format!(
            "{field} must be a finite non-negative safe integer"
        )));
    }
    Ok(value as u64)
}

pub(super) fn finite_nonnegative_f64_to_usize(value: f64, field: &'static str) -> Result<usize> {
    let value = finite_nonnegative_f64_to_u64(value, field)?;
    usize::try_from(value)
        .map_err(|_| invalid_arg(format!("{field} must fit in this platform's pointer width")))
}

pub(super) fn napi_error(message: impl ToString) -> Error {
    Error::new(Status::GenericFailure, message)
}

pub(super) fn core_error(error: cogentlm_core::Error) -> Error {
    match error {
        cogentlm_core::Error::InvalidRequest(message)
        | cogentlm_core::Error::InvalidConfig(message) => invalid_arg(message),
        other => napi_error(other.to_string()),
    }
}

pub(super) fn model_error(error: cogentlm_core::ModelError) -> Error {
    match error {
        cogentlm_core::ModelError::InvalidModelSource(message)
        | cogentlm_core::ModelError::InvalidModelPairing(message) => invalid_arg(message),
        cogentlm_core::ModelError::UnsupportedGgufVersion(version) => {
            invalid_arg(format!("unsupported GGUF version {version}"))
        }
        other => napi_error(other.to_string()),
    }
}
