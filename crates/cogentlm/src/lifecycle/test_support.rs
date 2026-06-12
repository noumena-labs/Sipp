use std::fs;
use std::path::PathBuf;

use super::storage::now_unix_ms;

pub(super) struct TempDir {
    pub(super) path: PathBuf,
}

impl TempDir {
    pub(super) fn new(scope: &str, name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("cogentlm-engine-{scope}-{name}-{}", now_unix_ms()));
        fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(super) fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

pub(super) fn some_string(value: &str) -> Option<String> {
    Some(value.to_string())
}

pub(super) fn gguf_name(id: &str) -> String {
    format!("{id}.gguf")
}

pub(super) fn gguf_path(id: &str) -> PathBuf {
    PathBuf::from(gguf_name(id))
}
