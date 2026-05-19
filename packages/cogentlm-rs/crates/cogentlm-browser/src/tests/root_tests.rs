//! Unit tests for the parent module.

use super::*;

#[test]
fn create_assigns_nonzero_ids() {
    let first = cogentlm_browser_engine_create();
    let second = cogentlm_browser_engine_create();
    assert!(!first.is_null());
    assert!(!second.is_null());
    assert_ne!(cogentlm_browser_engine_id(first), 0);
    assert_ne!(
        cogentlm_browser_engine_id(first),
        cogentlm_browser_engine_id(second)
    );
    assert_eq!(cogentlm_browser_engine_close(first), STATUS_OK);
    assert_eq!(cogentlm_browser_engine_close(second), STATUS_OK);
}

#[test]
fn close_rejects_null() {
    assert_eq!(
        cogentlm_browser_engine_close(std::ptr::null_mut()),
        STATUS_INVALID_ARGUMENTS
    );
}

#[test]
fn copies_nul_terminated_bytes() {
    let mut out = [0_u8; 6];
    assert_eq!(
        copy_bytes_with_nul(b"hello", out.as_mut_ptr(), out.len()),
        5
    );
    assert_eq!(&out, b"hello\0");
}

#[test]
fn copy_bytes_rejects_missing_nul_capacity() {
    let mut out = [0_u8; 5];
    assert_eq!(
        copy_bytes_with_nul(b"hello", out.as_mut_ptr(), out.len()),
        STATUS_INVALID_ARGUMENTS
    );
}
