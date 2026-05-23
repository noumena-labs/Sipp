use serde_json::Value;

use crate::backend::{
    json_array, json_str, json_u64, DEVICE_TYPE_GPU, DEVICE_TYPE_IGPU, KEY_DEVICES, KEY_DEVICE_ID,
    KEY_MEMORY_FREE_BYTES, KEY_NAME, KEY_TYPE,
};
use crate::error::Result;

use super::runtime_action_failed;

const UNKNOWN_DEVICE_COMPONENT: &str = "unknown";
const PARSE_BACKEND_RESIDENCY_FAILED: &str = "failed to parse backend residency";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GpuLeaseDevice {
    pub key: String,
    pub free_bytes: Option<u64>,
}

pub(super) fn parse_gpu_lease_devices(raw: &str) -> Result<Vec<GpuLeaseDevice>> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| runtime_action_failed(PARSE_BACKEND_RESIDENCY_FAILED, error))?;
    let Some(devices) = json_array(&value, KEY_DEVICES) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::with_capacity(devices.len());
    for (index, device) in devices.iter().enumerate() {
        if !is_gpu_lease_device(device) {
            continue;
        }
        out.push(GpuLeaseDevice {
            key: sanitize_lease_component(&gpu_lease_device_identity(device, index)),
            free_bytes: json_u64(device, KEY_MEMORY_FREE_BYTES),
        });
    }
    sort_unique_lease_devices(&mut out);
    Ok(out)
}

fn is_gpu_lease_device(device: &Value) -> bool {
    matches!(
        json_str(device, KEY_TYPE),
        Some(DEVICE_TYPE_GPU | DEVICE_TYPE_IGPU)
    )
}

fn gpu_lease_device_identity(device: &Value, index: usize) -> String {
    json_str(device, KEY_DEVICE_ID)
        .or_else(|| json_str(device, KEY_NAME))
        .map(str::to_string)
        .unwrap_or_else(|| format!("gpu-{index}"))
}

fn sort_unique_lease_devices(devices: &mut Vec<GpuLeaseDevice>) {
    devices.sort_by(|left, right| left.key.cmp(&right.key));
    devices.dedup_by(|left, right| left.key == right.key);
}

fn sanitize_lease_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        UNKNOWN_DEVICE_COMPONENT.to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
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
}
