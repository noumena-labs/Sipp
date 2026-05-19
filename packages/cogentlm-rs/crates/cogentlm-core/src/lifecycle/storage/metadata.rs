use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) fn modified_unix_ms(metadata: &fs::Metadata) -> Option<u64> {
    metadata.modified().ok().map(system_time_unix_ms)
}

pub(crate) fn now_unix_ms() -> u64 {
    system_time_unix_ms(SystemTime::now())
}

fn system_time_unix_ms(value: SystemTime) -> u64 {
    value
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

pub(super) fn normalize_asset_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.gguf")
        .trim();
    if name.is_empty() {
        "model.gguf".to_string()
    } else {
        name.replace(['\\', '/', ':', '*', '?', '"', '<', '>', '|'], "-")
    }
}

pub(super) fn unique_temp_suffix() -> String {
    format!(
        "{}-{}",
        now_unix_ms(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    )
}
