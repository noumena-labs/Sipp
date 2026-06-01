use super::super::*;

#[test]
fn incremental_sha256_matches_known_digest() {
    let mut hasher = BrowserSha256Hasher::new();
    hasher.update(b"abc");

    assert_eq!(
        hasher.finalize_hex(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
