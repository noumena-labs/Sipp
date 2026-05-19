//! Unit tests for the parent module.

use super::*;

#[test]
fn i64_to_u32_rejects_out_of_range_seed_values() {
    assert_eq!(i64_to_u32(0, "seed").expect("zero"), 0);
    assert_eq!(
        i64_to_u32(i64::from(u32::MAX), "seed").expect("max"),
        u32::MAX
    );
    assert!(i64_to_u32(-1, "seed").is_err());
    assert!(i64_to_u32(i64::from(u32::MAX) + 1, "seed").is_err());
}

#[test]
fn f64_to_f32_rejects_non_finite_and_out_of_range_values() {
    assert_eq!(f64_to_f32(0.5, "temperature").expect("finite"), 0.5_f32);
    assert!(f64_to_f32(f64::NAN, "temperature").is_err());
    assert!(f64_to_f32(f64::INFINITY, "temperature").is_err());
    assert!(f64_to_f32(f64::from(f32::MAX) * 2.0, "temperature").is_err());
    assert!(f64_to_f32(f64::from(f32::MIN) * 2.0, "temperature").is_err());
}

#[test]
fn sampling_config_rejects_non_finite_float_inputs() {
    let config = SamplingRuntimeConfig {
        temperature: Some(f64::NAN),
        ..Default::default()
    };
    assert!(config.to_core().is_err());
}

#[test]
fn placement_config_rejects_non_finite_tensor_split() {
    let config = ModelPlacementConfig {
        tensor_split: Some(vec![1.0, f64::INFINITY]),
        ..Default::default()
    };
    assert!(config.to_core().is_err());
}

#[test]
fn u64_to_js_safe_number_clamps_at_number_safe_integer() {
    assert_eq!(u64_to_js_safe_number(42), 42.0);
    assert_eq!(
        u64_to_js_safe_number(JS_MAX_SAFE_INTEGER_U64),
        JS_MAX_SAFE_INTEGER_F64
    );
    assert_eq!(u64_to_js_safe_number(u64::MAX), JS_MAX_SAFE_INTEGER_F64);
}

#[test]
fn i64_to_js_safe_number_clamps_negative_and_large_values() {
    assert_eq!(i64_to_js_safe_number(-1), 0.0);
    assert_eq!(i64_to_js_safe_number(42), 42.0);
    assert_eq!(i64_to_js_safe_number(i64::MAX), JS_MAX_SAFE_INTEGER_F64);
}

#[test]
fn finite_nonnegative_f64_to_u64_requires_safe_integer() {
    assert_eq!(
        finite_nonnegative_f64_to_u64(JS_MAX_SAFE_INTEGER_F64, "bytes").expect("max safe"),
        JS_MAX_SAFE_INTEGER_U64
    );
    assert!(finite_nonnegative_f64_to_u64(1.5, "bytes").is_err());
    assert!(finite_nonnegative_f64_to_u64(-1.0, "bytes").is_err());
    assert!(finite_nonnegative_f64_to_u64(f64::NAN, "bytes").is_err());
    assert!(finite_nonnegative_f64_to_u64(JS_MAX_SAFE_INTEGER_F64 + 2.0, "bytes").is_err());
}

#[test]
fn finite_nonnegative_f64_to_u64_parses_exact_integer_boundaries() {
    assert_eq!(finite_nonnegative_f64_to_u64(0.0, "bytes").unwrap(), 0);
    assert_eq!(finite_nonnegative_f64_to_u64(42.0, "bytes").unwrap(), 42);
    assert_eq!(
        finite_nonnegative_f64_to_u64(JS_MAX_SAFE_INTEGER_F64, "bytes").unwrap(),
        JS_MAX_SAFE_INTEGER_U64
    );
}
