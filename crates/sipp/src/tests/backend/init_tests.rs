//! Tests the `backend::init` module in `sipp`.
//!
//! Covers default logging policy without initializing native backend state or
//! loading model files.

use super::*;

#[test]
fn backend_logging_is_quiet_by_default() {
    assert!(default_llama_log_quiet_for_tests());
}
