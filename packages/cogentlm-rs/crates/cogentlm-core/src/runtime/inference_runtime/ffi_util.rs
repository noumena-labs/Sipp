//! Thin helpers around the llama.cpp / cogent shim FFI boundary.
//!
//! These wrap raw `*mut c_char` patterns into `Result<String>` / `Error` so
//! the runtime code paths stay free of unsafe boilerplate.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};

pub(super) fn c_strings_from_args(args: &[String]) -> Result<Vec<CString>> {
    args.iter()
        .map(|arg| CString::new(arg.as_str()).map_err(Error::from))
        .collect()
}

pub(super) fn c_ptrs_from_strings(args: &[CString]) -> Vec<*const c_char> {
    args.iter().map(|arg| arg.as_ptr()).collect()
}

/// Consumes a shim-owned error string (freeing it) and folds it into an
/// `Error::RuntimeCommand`. Falls back to `fallback` on null.
pub(super) fn runtime_command_from_shim_error(value: *mut c_char, fallback: &'static str) -> Error {
    if value.is_null() {
        return Error::RuntimeCommand(fallback.to_string());
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        ffi::cogent_free_string(value);
    }
    Error::RuntimeCommand(result)
}

/// Consumes a shim-owned string (freeing it) and returns an owned `String`.
/// Returns `Error::NullPointer(name)` if the pointer is null.
pub(super) fn owned_shim_string(value: *mut c_char, name: &'static str) -> Result<String> {
    if value.is_null() {
        return Err(Error::NullPointer(name));
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        ffi::cogent_free_string(value);
    }
    Ok(result)
}
