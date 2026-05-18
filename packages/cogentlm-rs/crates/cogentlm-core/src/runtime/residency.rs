use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::error::{Error, Result};
use crate::runtime::config::{GpuLayerConfig, NativeRuntimeConfig};

#[derive(Debug)]
pub(crate) struct ResidencyLease {
    files: Vec<ResidencyLeaseFile>,
}

impl ResidencyLease {
    fn new(files: Vec<ResidencyLeaseFile>) -> Self {
        Self { files }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.files.len()
    }
}

impl Drop for ResidencyLease {
    fn drop(&mut self) {
        for lease in self.files.drain(..) {
            drop(lease.file);
            let _ = fs::remove_file(lease.path);
        }
    }
}

#[derive(Debug)]
struct ResidencyLeaseFile {
    path: PathBuf,
    file: File,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuLeaseDevice {
    key: String,
    free_bytes: Option<u64>,
}

pub(crate) fn acquire_residency_lease(
    config: &NativeRuntimeConfig,
    backend_observability_json: &str,
) -> Result<Option<ResidencyLease>> {
    acquire_residency_lease_in(
        config,
        backend_observability_json,
        &default_residency_lease_root(),
    )
}

fn acquire_residency_lease_in(
    config: &NativeRuntimeConfig,
    backend_observability_json: &str,
    lease_root: &Path,
) -> Result<Option<ResidencyLease>> {
    if config.placement.gpu_layers == GpuLayerConfig::Count(0) {
        return Ok(None);
    }

    let devices = parse_gpu_lease_devices(backend_observability_json)?;
    if devices.is_empty() {
        return Ok(None);
    }

    enforce_vram_margin(&devices, config.residency.gpu_memory_safety_margin_bytes)?;
    if !config.residency.require_gpu_lease {
        return Ok(None);
    }

    fs::create_dir_all(lease_root).map_err(|error| {
        Error::RuntimeCommand(format!("failed to create residency lease dir: {error}"))
    })?;

    let mut leases = Vec::with_capacity(devices.len());
    for device in &devices {
        match acquire_device_lease(
            lease_root,
            device,
            config.residency.max_gpu_models_per_device.max(1),
        ) {
            Ok(lease) => leases.push(lease),
            Err(error) => {
                drop(ResidencyLease::new(leases));
                return Err(error);
            }
        }
    }

    Ok(Some(ResidencyLease::new(leases)))
}

fn parse_gpu_lease_devices(raw: &str) -> Result<Vec<GpuLeaseDevice>> {
    let value: Value = serde_json::from_str(raw).map_err(|error| {
        Error::RuntimeCommand(format!("failed to parse backend residency: {error}"))
    })?;
    let Some(devices) = value.get("devices").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
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

fn enforce_vram_margin(devices: &[GpuLeaseDevice], margin: u64) -> Result<()> {
    if margin == 0 {
        return Ok(());
    }
    let max_free = devices
        .iter()
        .filter_map(|device| device.free_bytes)
        .max()
        .unwrap_or(u64::MAX);
    if max_free < margin {
        return Err(Error::RuntimeCommand(format!(
            "gpu residency rejected: max free VRAM {max_free} bytes is below safety margin {margin} bytes"
        )));
    }
    Ok(())
}

fn acquire_device_lease(
    lease_root: &Path,
    device: &GpuLeaseDevice,
    max_models: usize,
) -> Result<ResidencyLeaseFile> {
    for slot in 0..max_models {
        let path = lease_root.join(format!("gpu-{}-slot-{slot}.lock", device.key));
        for attempt in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let payload = serde_json::json!({
                        "pid": std::process::id(),
                        "device": device.key,
                        "slot": slot,
                        "createdAtUnixMs": now_unix_ms(),
                    });
                    writeln!(file, "{payload}").map_err(|error| {
                        Error::RuntimeCommand(format!("failed to write residency lease: {error}"))
                    })?;
                    return Ok(ResidencyLeaseFile { path, file });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if attempt == 0 && remove_stale_lease_file(&path)? {
                        continue;
                    }
                    break;
                }
                Err(error) => {
                    return Err(Error::RuntimeCommand(format!(
                        "failed to acquire gpu residency lease for {}: {error}",
                        device.key
                    )));
                }
            }
        }
    }

    Err(Error::RuntimeCommand(format!(
        "gpu residency rejected: device {} already has {} active model lease(s)",
        device.key, max_models
    )))
}

fn remove_stale_lease_file(path: &Path) -> Result<bool> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(_) => return Ok(false),
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return Ok(false);
    };
    let Some(pid) = value.get("pid").and_then(Value::as_u64) else {
        return Ok(false);
    };
    let Ok(pid) = u32::try_from(pid) else {
        return remove_lease_file(path);
    };
    if process_is_running(pid) {
        return Ok(false);
    }
    remove_lease_file(path)
}

fn remove_lease_file(path: &Path) -> Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(Error::RuntimeCommand(format!(
            "failed to remove stale gpu residency lease {}: {error}",
            path.display()
        ))),
    }
}

#[cfg(windows)]
fn process_is_running(pid: u32) -> bool {
    use std::ffi::c_void;

    if pid == 0 {
        return false;
    }

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;
    const ERROR_INVALID_PARAMETER: u32 = 87;

    extern "system" {
        fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: i32, dwProcessId: u32) -> *mut c_void;
        fn GetExitCodeProcess(hProcess: *mut c_void, lpExitCode: *mut u32) -> i32;
        fn CloseHandle(hObject: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return GetLastError() != ERROR_INVALID_PARAMETER;
        }
        let mut exit_code = 0;
        let ok = GetExitCodeProcess(handle, &mut exit_code);
        let _ = CloseHandle(handle);
        ok != 0 && exit_code == STILL_ACTIVE
    }
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }

    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }

    unsafe { kill(pid as i32, 0) == 0 }
}

#[cfg(not(any(unix, windows)))]
fn process_is_running(_pid: u32) -> bool {
    true
}

fn default_residency_lease_root() -> PathBuf {
    std::env::temp_dir().join("cogentlm-rs-residency")
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

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let lease =
            acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");
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
        let first =
            acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");
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

        let lease =
            acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");

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

        let lease =
            acquire_residency_lease_in(&config, &backend_json(u64::MAX), &root).expect("lease");

        assert!(lease.is_none());
        assert!(!root.exists());
    }

    #[test]
    fn vram_margin_rejects_too_little_free_memory() {
        let root = test_lease_root("margin");
        let mut config = NativeRuntimeConfig::default();
        config.residency.gpu_memory_safety_margin_bytes = 128;

        let error = acquire_residency_lease_in(&config, &backend_json(127), &root)
            .expect_err("margin should reject");

        assert!(
            matches!(error, Error::RuntimeCommand(message) if message.contains("safety margin"))
        );
        let _ = fs::remove_dir_all(root);
    }
}
