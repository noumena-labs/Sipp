//! Tests the `runtime::inference_runtime::native` module in
//! `cogentlm-engine`.
//!
//! Covers empty native-runtime convenience paths with fake handles, including
//! scalar token defaults and model-free RuntimeNotReady error propagation.

use crate::error::Error;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;

#[test]
fn empty_runtime_returns_empty_bos_and_eos_text_without_native_lookup() {
    let runtime = test_runtime(NativeRuntimeConfig::default());

    assert_eq!(runtime.get_bos_text().expect("bos text"), "");
    assert_eq!(runtime.get_eos_text().expect("eos text"), "");
}

#[test]
fn empty_runtime_template_methods_report_not_ready() {
    let runtime = test_runtime(NativeRuntimeConfig::default());

    assert!(matches!(
        runtime.chat_template_source(),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.apply_chat_template_json("[]", true),
        Err(Error::RuntimeNotReady)
    ));
}
