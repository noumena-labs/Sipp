use std::ffi::CStr;

use crate::error::{Error, Result};

use super::ensure_backend_initialized;

pub fn backend_observability_json(include_details: bool) -> Result<String> {
    ensure_backend_initialized();
    let value = unsafe { cogentlm_sys::cogent_backend_observability_json(include_details) };
    if value.is_null() {
        return Err(Error::NullPointer("cogent_backend_observability_json"));
    }

    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        cogentlm_sys::cogent_free_string(value);
    }
    Ok(result)
}
