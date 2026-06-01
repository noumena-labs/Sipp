//! Unit tests for the parent module.

use super::*;

#[test]
fn py_i64_to_u32_rejects_out_of_range_seed_values() {
    assert_eq!(py_i64_to_u32(0).expect("zero"), 0);
    assert_eq!(py_i64_to_u32(i64::from(u32::MAX)).expect("max"), u32::MAX);
    assert!(py_i64_to_u32(-1).is_err());
    assert!(py_i64_to_u32(i64::from(u32::MAX) + 1).is_err());
}

#[test]
fn py_finite_f32_rejects_non_finite_values() {
    assert_eq!(py_finite_f32(0.5, "temperature").expect("finite"), 0.5);
    assert!(py_finite_f32(f32::NAN, "temperature").is_err());
    assert!(py_finite_f32(f32::INFINITY, "temperature").is_err());
    assert!(py_optional_finite_f32(Some(f32::NEG_INFINITY), "temperature").is_err());
}

#[test]
fn py_sampling_config_rejects_non_finite_float_inputs() {
    let config = PySamplingRuntimeConfig::new(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(f32::NAN),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        false,
        None,
        true,
    );
    assert!(config.is_err());
}

#[test]
fn py_placement_config_rejects_non_finite_tensor_split() {
    let config = PyModelPlacementConfig::new(
        None,
        None,
        None,
        None,
        Some(vec![1.0, f32::INFINITY]),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(config.is_err());
}
