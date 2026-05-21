use std::ffi::CStr;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};

pub fn default_media_marker() -> Result<String> {
    let marker = unsafe { ffi::cogent_mtmd_default_marker() };
    if marker.is_null() {
        return Err(Error::NullPointer("cogent_mtmd_default_marker"));
    }
    Ok(unsafe { CStr::from_ptr(marker) }
        .to_string_lossy()
        .into_owned())
}
