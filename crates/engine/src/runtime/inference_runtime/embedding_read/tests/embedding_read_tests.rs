use super::super::*;
use crate::engine::protocol::PoolingType;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn l2_normalize_unit_vector_is_idempotent() {
    let mut values = vec![1.0, 0.0, 0.0];
    l2_normalize(&mut values);
    assert!(approx_eq(values[0], 1.0));
    assert!(approx_eq(values[1], 0.0));
    assert!(approx_eq(values[2], 0.0));
}

#[test]
fn l2_normalize_scales_to_unit_length() {
    let mut values = vec![3.0, 4.0];
    l2_normalize(&mut values);
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    assert!(approx_eq(sum_sq.sqrt(), 1.0));
    assert!(approx_eq(values[0], 0.6));
    assert!(approx_eq(values[1], 0.8));
}

#[test]
fn l2_normalize_zero_vector_is_left_alone() {
    let mut values = vec![0.0, 0.0, 0.0];
    l2_normalize(&mut values);
    for value in &values {
        assert_eq!(*value, 0.0);
    }
}

#[test]
fn apply_normalization_skips_rank_pooling() {
    let output = apply_normalization(vec![3.0, 4.0], PoolingType::Rank, true);
    assert!(!output.normalized);
    assert_eq!(output.pooling, PoolingType::Rank);
    assert_eq!(output.values, vec![3.0, 4.0]);
}

#[test]
fn apply_normalization_respects_normalize_request() {
    let output = apply_normalization(vec![3.0, 4.0], PoolingType::Mean, false);
    assert!(!output.normalized);
    assert_eq!(output.values, vec![3.0, 4.0]);

    let output = apply_normalization(vec![3.0, 4.0], PoolingType::Mean, true);
    assert!(output.normalized);
    assert!(approx_eq(output.values[0], 0.6));
    assert!(approx_eq(output.values[1], 0.8));
}
