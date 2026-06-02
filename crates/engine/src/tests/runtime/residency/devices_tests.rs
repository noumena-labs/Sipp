//! Tests the `runtime::residency::devices` module in `cogentlm-engine`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use super::*;

#[test]
fn parses_gpu_lease_devices_with_stable_order_and_capacity() {
    let raw = serde_json::json!({
        "devices": [
            {"type": "CPU", "name": "cpu"},
            {"type": "GPU", "name": "GPU B", "memoryFreeBytes": 20},
            {"type": "IGPU", "memoryFreeBytes": 10},
            {"type": "GPU", "name": "GPU B", "memoryFreeBytes": 30}
        ]
    })
    .to_string();

    let devices = parse_gpu_lease_devices(&raw).expect("devices");

    assert_eq!(
        devices,
        vec![
            GpuLeaseDevice {
                key: "GPU_B".to_string(),
                free_bytes: Some(20),
            },
            GpuLeaseDevice {
                key: "gpu-2".to_string(),
                free_bytes: Some(10),
            },
        ]
    );
    assert!(devices.capacity() >= 4);
}

#[test]
fn gpu_lease_device_helpers_preserve_identity_and_dedup_rules() {
    let device = serde_json::json!({
        "type": "GPU",
        "deviceId": "pci:0",
        "name": "fallback"
    });
    let cpu = serde_json::json!({ "type": "CPU", "name": "cpu" });
    let unnamed = serde_json::json!({ "type": "IGPU" });

    assert!(is_gpu_lease_device(&device));
    assert!(!is_gpu_lease_device(&cpu));
    assert_eq!(gpu_lease_device_identity(&device, 3), "pci:0");
    assert_eq!(gpu_lease_device_identity(&unnamed, 3), "gpu-3");

    let mut devices = vec![
        GpuLeaseDevice {
            key: "b".to_string(),
            free_bytes: Some(2),
        },
        GpuLeaseDevice {
            key: "a".to_string(),
            free_bytes: Some(1),
        },
        GpuLeaseDevice {
            key: "b".to_string(),
            free_bytes: Some(3),
        },
    ];

    sort_unique_lease_devices(&mut devices);

    assert_eq!(
        devices,
        vec![
            GpuLeaseDevice {
                key: "a".to_string(),
                free_bytes: Some(1),
            },
            GpuLeaseDevice {
                key: "b".to_string(),
                free_bytes: Some(2),
            },
        ]
    );
}
