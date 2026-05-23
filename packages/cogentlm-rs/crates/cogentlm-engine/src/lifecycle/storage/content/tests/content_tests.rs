//! Unit tests for the parent module.

use super::super::*;

#[test]
fn inspection_prefix_capacity_clamps_to_prefix_limit() {
    assert_eq!(inspection_prefix_capacity(0), 0);
    assert_eq!(inspection_prefix_capacity(1024), 1024);
    assert_eq!(
        inspection_prefix_capacity((INSPECTION_PREFIX_BYTES as u64) + 1),
        INSPECTION_PREFIX_BYTES
    );
    assert_eq!(
        inspection_prefix_capacity(u64::MAX),
        INSPECTION_PREFIX_BYTES
    );
}

#[test]
fn hash_reader_collects_only_requested_prefix_bytes() {
    let mut source = &b"abcdef"[..];
    let (hash, prefix) = hash_reader(&mut source, 3).expect("hash");

    assert_eq!(hash, hex_lower(&Sha256::digest(b"abcdef")));
    assert_eq!(prefix, b"abc");

    let mut source = &b"abcdef"[..];
    let (_, prefix) = hash_reader(&mut source, 0).expect("hash without prefix");

    assert!(prefix.is_empty());
}
