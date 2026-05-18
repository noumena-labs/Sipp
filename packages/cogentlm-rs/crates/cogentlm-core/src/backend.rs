use std::sync::Once;

use std::ffi::CStr;

use crate::error::{Error, Result};

static INIT_BACKEND: Once = Once::new();

pub(crate) fn ensure_backend_initialized() {
    INIT_BACKEND.call_once(|| unsafe {
        cogentlm_sys::llama_backend_init();
        cogentlm_sys::cogent_backend_load_all();
    });
}

pub fn set_llama_log_quiet(quiet: bool) {
    unsafe {
        cogentlm_sys::cogent_set_llama_log_quiet(quiet);
    }
}

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
