//! GPU residency leases: parses backend devices and enforces VRAM/model-count limits per device.

use std::fmt::Display;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{Error, Result};
use crate::runtime::config::{GpuLayerConfig, NativeRuntimeConfig};
use crate::runtime::numeric::unix_time_ms;

mod devices;

use devices::{parse_gpu_lease_devices, GpuLeaseDevice};

const GPU_RESIDENCY_REJECTED: &str = "gpu residency rejected";
const CREATE_LEASE_DIR_FAILED: &str = "failed to create residency lease dir";
const WRITE_LEASE_FAILED: &str = "failed to write residency lease";

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

    fs::create_dir_all(lease_root)
        .map_err(|error| runtime_action_failed(CREATE_LEASE_DIR_FAILED, error))?;

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
        return Err(residency_rejected(format!(
            "max free VRAM {max_free} bytes is below safety margin {margin} bytes"
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
                        "createdAtUnixMs": unix_time_ms(),
                    });
                    writeln!(file, "{payload}")
                        .map_err(|error| runtime_action_failed(WRITE_LEASE_FAILED, error))?;
                    return Ok(ResidencyLeaseFile { path, file });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if attempt == 0 && remove_stale_lease_file(&path)? {
                        continue;
                    }
                    break;
                }
                Err(error) => {
                    return Err(runtime_command(format!(
                        "failed to acquire gpu residency lease for {}: {error}",
                        device.key
                    )));
                }
            }
        }
    }

    Err(residency_rejected(format!(
        "device {} already has {} active model lease(s)",
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
        Err(error) => Err(runtime_command(format!(
            "failed to remove stale gpu residency lease {}: {error}",
            path.display()
        ))),
    }
}

fn runtime_command(message: impl Into<String>) -> Error {
    Error::RuntimeCommand(message.into())
}

fn runtime_action_failed(action: &str, error: impl Display) -> Error {
    runtime_command(format!("{action}: {error}"))
}

fn residency_rejected(reason: impl Display) -> Error {
    runtime_command(format!("{GPU_RESIDENCY_REJECTED}: {reason}"))
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

    // SAFETY: The declarations above mirror the Win32 process-query APIs. The
    // handle returned by OpenProcess is checked for null, used only with the
    // same APIs, and closed exactly once on the success path.
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
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };

    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }

    // SAFETY: kill(pid, 0) performs existence/permission probing without
    // delivering a signal. The pid was validated as a positive platform pid.
    unsafe { kill(pid, 0) == 0 }
}

#[cfg(not(any(unix, windows)))]
fn process_is_running(_pid: u32) -> bool {
    true
}

fn default_residency_lease_root() -> PathBuf {
    std::env::temp_dir().join("cogentlm-rs-residency")
}

#[cfg(test)]
#[path = "../../tests/runtime/residency_tests.rs"]
mod residency_tests;
