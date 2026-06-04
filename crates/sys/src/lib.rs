pub mod bridge;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[allow(dead_code)]
#[path = "../build_support/mod.rs"]
mod build_support;

#[cfg(test)]
#[path = "tests/build_support/common.rs"]
mod build_support_test_common;

#[cfg(test)]
#[path = "tests/bridge_tests.rs"]
mod bridge_tests;

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[allow(non_camel_case_types)]
pub type llama_token = i32;

#[allow(non_camel_case_types)]
pub type llama_seq_id = i32;

pub const LLAMA_TOKEN_NULL: llama_token = -1;
