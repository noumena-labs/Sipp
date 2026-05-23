use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use serde::Serialize;

pub(crate) fn read_optional_c_string(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return Some(String::new());
    }
    // SAFETY: Callers pass a NUL-terminated C string pointer that remains valid
    // for the duration of this call.
    Some(
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned(),
    )
}

pub(crate) fn into_c_string(value: String) -> *mut c_char {
    CString::new(value.replace('\0', ""))
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

pub(crate) fn serialize_json_response<T>(response: &T, fallback: &'static str) -> String
where
    T: Serialize,
{
    serde_json::to_string(response).unwrap_or_else(|_| fallback.to_string())
}

/// # Safety
/// `value` must be null or a pointer returned by `CString::into_raw` from this
/// module. Each non-null pointer must be freed at most once.
pub(crate) unsafe fn free_c_string(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    // SAFETY: The caller guarantees `value` came from `CString::into_raw` and
    // has not already been reclaimed.
    unsafe {
        drop(CString::from_raw(value));
    }
}
