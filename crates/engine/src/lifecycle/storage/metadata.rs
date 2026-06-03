use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::defaults::DEFAULT_MODEL_FILE_NAME;
use crate::runtime::numeric::{system_time_unix_ms, unix_time_ms};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/storage/metadata_tests.rs"]
mod metadata_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) fn modified_unix_ms(metadata: &fs::Metadata) -> Option<u64> {
    metadata.modified().ok().map(system_time_unix_ms)
}

pub(crate) fn now_unix_ms() -> u64 {
    unix_time_ms()
}

pub(super) fn normalize_asset_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(DEFAULT_MODEL_FILE_NAME)
        .trim();
    if name.is_empty() {
        DEFAULT_MODEL_FILE_NAME.to_string()
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
