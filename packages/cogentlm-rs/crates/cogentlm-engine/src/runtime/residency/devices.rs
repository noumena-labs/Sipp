use serde_json::Value;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GpuLeaseDevice {
    pub key: String,
    pub free_bytes: Option<u64>,
}

pub(super) fn parse_gpu_lease_devices(raw: &str) -> Result<Vec<GpuLeaseDevice>> {
    let value: Value = serde_json::from_str(raw).map_err(|error| {
        Error::RuntimeCommand(format!("failed to parse backend residency: {error}"))
    })?;
    let Some(devices) = value.get("devices").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::with_capacity(devices.len());
    for (index, device) in devices.iter().enumerate() {
        let device_type = device
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if device_type != "GPU" && device_type != "IGPU" {
            continue;
        }
        let identity = device
            .get("deviceId")
            .and_then(Value::as_str)
            .or_else(|| device.get("name").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| format!("gpu-{index}"));
        out.push(GpuLeaseDevice {
            key: sanitize_lease_component(&identity),
            free_bytes: device.get("memoryFreeBytes").and_then(Value::as_u64),
        });
    }
    out.sort_by(|left, right| left.key.cmp(&right.key));
    out.dedup_by(|left, right| left.key == right.key);
    Ok(out)
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
        "unknown".to_string()
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
}
