use std::ffi::CString;

use super::super::*;

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
