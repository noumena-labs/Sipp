//! Tests the `runtime::residency` module in `cogentlm-engine`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use std::time::Duration;

use super::*;
use crate::runtime::numeric::duration_millis_u64;

fn backend_json(free: u64) -> String {
    serde_json::json!({
        "devices": [
            {
                "type": "GPU",
                "name": "CUDA0",
                "deviceId": "0000:01:00.0",
                "memoryFreeBytes": free
            },
            {
                "type": "CPU",
                "name": "CPU",
                "memoryFreeBytes": 1
            }
        ]
    })
    .to_string()
}

fn test_lease_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "cogentlm-rs-residency-test-{}-{}",
        name,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    root
}

#[test]
fn gpu_runtime_acquires_and_releases_lease() {
    let root = test_lease_root("release");
    let config = NativeRuntimeConfig::default();
    let lease = acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");
    assert_eq!(lease.as_ref().map(ResidencyLease::len), Some(1));
    assert_eq!(fs::read_dir(&root).expect("lease dir").count(), 1);

    drop(lease);

    assert_eq!(fs::read_dir(&root).expect("lease dir").count(), 0);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn gpu_runtime_rejects_when_lease_slots_are_full() {
    let root = test_lease_root("full");
    let config = NativeRuntimeConfig::default();
    let first = acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");
    assert!(first.is_some());

    let error = acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root)
        .expect_err("second lease should fail");
    assert!(matches!(error, Error::RuntimeCommand(message) if message.contains("already has")));

    drop(first);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn stale_lease_file_is_removed_and_reused() {
    let root = test_lease_root("stale");
    let config = NativeRuntimeConfig::default();
    fs::create_dir_all(&root).expect("lease dir");
    let path = root.join("gpu-0000_01_00_0-slot-0.lock");
    let stale_pid = u32::MAX;
    fs::write(
        &path,
        serde_json::json!({
            "pid": stale_pid,
            "device": "0000_01_00_0",
            "slot": 0,
            "createdAtUnixMs": 1,
        })
        .to_string(),
    )
    .expect("stale lease");

    let lease = acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");

    assert_eq!(lease.as_ref().map(ResidencyLease::len), Some(1));
    let payload = fs::read_to_string(&path).expect("new lease payload");
    let value: Value = serde_json::from_str(&payload).expect("lease json");
    assert_eq!(
        value.get("pid").and_then(Value::as_u64),
        Some(u64::from(std::process::id()))
    );
    drop(lease);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn cpu_runtime_skips_gpu_lease() {
    let root = test_lease_root("cpu");
    let mut config = NativeRuntimeConfig::default();
    config.placement.gpu_layers = GpuLayerConfig::Count(0);

    let lease = acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");

    assert!(lease.is_none());
    assert!(!root.exists());
}

#[test]
fn duration_millis_saturates_at_u64_max() {
    let oversized = Duration::from_millis(u64::MAX).saturating_add(Duration::from_millis(1));

    assert_eq!(duration_millis_u64(oversized), u64::MAX);
}

#[test]
fn vram_margin_rejects_too_little_free_memory() {
    let root = test_lease_root("margin");
    let mut config = NativeRuntimeConfig::default();
    config.residency.gpu_memory_safety_margin_bytes = 128;

    let error = acquire_residency_lease_in(&config, &backend_json(127), &root)
        .expect_err("margin should reject");

    assert!(matches!(error, Error::RuntimeCommand(message) if message.contains("safety margin")));
    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn oversized_pid_is_not_treated_as_running_on_unix() {
    assert!(!process_is_running(u32::MAX));
}
