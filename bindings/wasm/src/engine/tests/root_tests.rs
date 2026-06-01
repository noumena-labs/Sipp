//! Unit tests for the parent module.

use super::*;

#[test]
fn create_assigns_nonzero_ids() {
    let first = BrowserEngine::create();
    let second = BrowserEngine::create();
    assert_ne!(first.id(), 0);
    assert_ne!(first.id(), second.id());
}

#[test]
fn copies_nul_terminated_bytes() {
    let mut out = [0_u8; 6];
    assert_eq!(copy_bytes_with_nul(b"hello", &mut out), 5);
    assert_eq!(&out, b"hello\0");
}

#[test]
fn copy_bytes_rejects_missing_nul_capacity() {
    let mut out = [0_u8; 5];
    assert_eq!(
        copy_bytes_with_nul(b"hello", &mut out),
        STATUS_INVALID_ARGUMENTS
    );
}
