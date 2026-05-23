use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use sha2::{Digest, Sha256};

use crate::ffi::into_c_string;

pub struct BrowserSha256Hasher {
    hasher: Sha256,
}

#[no_mangle]
pub extern "C" fn cogentlm_sha256_create() -> *mut BrowserSha256Hasher {
    catch_unwind(AssertUnwindSafe(|| {
        Box::into_raw(Box::new(BrowserSha256Hasher {
            hasher: Sha256::new(),
        }))
    }))
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn cogentlm_sha256_update(
    hasher: *mut BrowserSha256Hasher,
    bytes_ptr: *const u8,
    bytes_len: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(hasher) = hasher_mut(hasher) else {
            return -1;
        };
        let Some(bytes) = bytes_arg(bytes_ptr, bytes_len) else {
            return -1;
        };
        hasher.hasher.update(bytes);
        0
    }))
    .unwrap_or(-1)
}

#[no_mangle]
/// # Safety
/// `hasher` must be null or a live handle returned by `cogentlm_sha256_create`.
/// A non-null handle is consumed and must not be reused.
pub unsafe extern "C" fn cogentlm_sha256_finalize(hasher: *mut BrowserSha256Hasher) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        if hasher.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: `hasher` must be a live handle returned by
        // `cogentlm_sha256_create`. Finalize consumes that handle exactly once.
        let hasher = unsafe { Box::from_raw(hasher) };
        into_c_string(hex_lower(&hasher.hasher.finalize()))
    }))
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
/// # Safety
/// `hasher` must be null or a live handle returned by `cogentlm_sha256_create`.
/// A non-null handle is consumed and must not be reused.
pub unsafe extern "C" fn cogentlm_sha256_close(hasher: *mut BrowserSha256Hasher) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if hasher.is_null() {
            return 0;
        }
        // SAFETY: `hasher` must be a live handle returned by
        // `cogentlm_sha256_create`. Close consumes the handle without using it.
        drop(unsafe { Box::from_raw(hasher) });
        1
    }))
    .unwrap_or(0)
}

fn hasher_mut(hasher: *mut BrowserSha256Hasher) -> Option<&'static mut BrowserSha256Hasher> {
    if hasher.is_null() {
        return None;
    }
    // SAFETY: The caller must pass a live hasher handle created by
    // `cogentlm_sha256_create` and must not call into this handle concurrently.
    Some(unsafe { &mut *hasher })
}

fn bytes_arg(bytes_ptr: *const u8, bytes_len: usize) -> Option<&'static [u8]> {
    if bytes_ptr.is_null() && bytes_len > 0 {
        return None;
    }
    Some(if bytes_len == 0 {
        &[]
    } else {
        // SAFETY: The wasm/C wrapper passes a pointer into wasm linear memory
        // that is valid for `bytes_len` bytes for the duration of this call.
        unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) }
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn streaming_sha256_matches_known_digest() {
        let hasher = cogentlm_sha256_create();
        assert!(!hasher.is_null());
        assert_eq!(cogentlm_sha256_update(hasher, b"abc".as_ptr(), 3), 0);

        let ptr = unsafe { cogentlm_sha256_finalize(hasher) };
        assert!(!ptr.is_null());
        let digest = unsafe { CString::from_raw(ptr) }
            .to_string_lossy()
            .into_owned();

        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
