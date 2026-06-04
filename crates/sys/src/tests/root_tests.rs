//! Tests the `cogentlm-sys` crate root public aliases.
//!
//! Covers ABI-sized token and sequence identifiers with deterministic,
//! model-free assertions.

use super::{llama_seq_id, llama_token, LLAMA_TOKEN_NULL};

#[test]
fn llama_aliases_match_expected_i32_abi() {
    assert_eq!(
        std::mem::size_of::<llama_token>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(
        std::mem::size_of::<llama_seq_id>(),
        std::mem::size_of::<i32>()
    );

    let token: llama_token = -7;
    let seq_id: llama_seq_id = 13;
    assert_eq!(token, -7_i32);
    assert_eq!(seq_id, 13_i32);
}

#[test]
fn llama_token_null_matches_llama_sentinel() {
    let token: llama_token = LLAMA_TOKEN_NULL;
    assert_eq!(token, -1);
}
