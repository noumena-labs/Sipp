//! Filesystem layout for stored model assets and registry manifests.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use super::{
    detect_model_from_gguf_bytes, AssetRecord, AssetRole, AssetSource, ModelAssetKind, ModelError,
};

mod content;
mod metadata;

pub(crate) use content::hash_file;
use content::inspect_local_path;
pub(crate) use metadata::{modified_unix_ms, now_unix_ms};
use metadata::{normalize_asset_name, unique_temp_suffix};

const ASSETS_DIR: &str = "assets";
const INCOMING_DIR: &str = ".incoming";
const REGISTRY_FILE_NAME: &str = "registry.json";
pub(super) const COPY_BUFFER_BYTES: usize = 1024 * 1024;

pub trait StorageBackend: Clone + Send + Sync + 'static {
    fn root(&self) -> &Path;

    fn manifest_path(&self) -> PathBuf {
        self.root().join(REGISTRY_FILE_NAME)
    }

    fn asset_storage_path(&self, asset_id: &str) -> PathBuf {
        PathBuf::from(ASSETS_DIR).join(asset_id)
    }

    fn asset_path(&self, asset_id: &str) -> PathBuf {
        self.root().join(self.asset_storage_path(asset_id))
    }

    fn resolve_storage_path(&self, storage_path: &Path) -> PathBuf {
        if storage_path.is_absolute() {
            storage_path.to_path_buf()
        } else {
            self.root().join(storage_path)
        }
    }

    fn ensure_layout(&self) -> Result<(), ModelError> {
        fs::create_dir_all(self.root().join(ASSETS_DIR))?;
        fs::create_dir_all(self.root().join(INCOMING_DIR))?;
        Ok(())
    }

    fn atomic_write(&self, path: &Path, bytes: &[u8]) -> Result<(), ModelError> {
        let parent = path.parent().ok_or_else(|| {
            ModelError::StorageUnavailable(format!(
                "storage path has no parent: {}",
                path.display()
            ))
        })?;
        fs::create_dir_all(parent)?;

        let tmp_path = parent.join(format!(
            ".{}.tmp-{}",
            REGISTRY_FILE_NAME,
            unique_temp_suffix()
        ));
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(bytes)?;
            tmp.sync_all()?;
        }

        match fs::rename(&tmp_path, path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                fs::remove_file(path)?;
                fs::rename(&tmp_path, path)?;
                Ok(())
            }
            Err(error) => {
                let _ = fs::remove_file(&tmp_path);
                Err(ModelError::Io(error))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalStorageBackend {
    root: PathBuf,
}

impl LocalStorageBackend {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl StorageBackend for LocalStorageBackend {
    fn root(&self) -> &Path {
        &self.root
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetInstallResult {
    pub record: AssetRecord,
    pub already_present: bool,
}

#[derive(Debug, Clone)]
pub struct AssetStore<B = LocalStorageBackend> {
    backend: B,
}

impl AssetStore<LocalStorageBackend> {
    pub fn local(root: impl Into<PathBuf>) -> Self {
        Self::new(LocalStorageBackend::new(root))
    }
}

impl<B: StorageBackend> AssetStore<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn install_local_path(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<AssetInstallResult, ModelError> {
        self.install_local_path_as(path, None)
    }

    pub fn install_local_path_as(
        &self,
        path: impl AsRef<Path>,
        kind: Option<ModelAssetKind>,
    ) -> Result<AssetInstallResult, ModelError> {
        self.backend.ensure_layout()?;
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() {
            return Err(ModelError::InvalidModelSource(format!(
                "model asset is not a file: {}",
                path.display()
            )));
        }

        let name = normalize_asset_name(path);
        let source_path = canonicalize_existing_path(path)?;
        let source_modified_unix_ms = modified_unix_ms(&metadata);
        let (hash, prefix) = inspect_local_path(path)?;
        let id = format!("asset-{hash}");
        let storage_path = self.backend.asset_storage_path(&id);
        let final_path = self.backend.asset_path(&id);
        let already_present = final_path.exists();

        if already_present {
            let existing_bytes = fs::metadata(&final_path)?.len();
            if existing_bytes != metadata.len() {
                return Err(ModelError::StorageCorrupt(format!(
                    "asset {} has hash match but byte-size mismatch",
                    id
                )));
            }
            let existing_hash = hash_file(&final_path)?;
            if existing_hash != hash {
                return Err(ModelError::StorageCorrupt(format!(
                    "asset {} has hash match but content mismatch",
                    id
                )));
            }
        } else {
            let tmp_path = self.incoming_path();
            stage_local_path(path, &tmp_path)?;
            if let Some(parent) = final_path.parent() {
                fs::create_dir_all(parent)?;
            }
            match fs::rename(&tmp_path, &final_path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    fs::remove_file(tmp_path)?;
                }
                Err(error) => return Err(ModelError::Io(error)),
            }
        }

        let detection = detect_model_from_gguf_bytes(&name, &prefix)?;
        let inspection = detection.inspection;
        let inferred_kind = kind.unwrap_or(match inspection.role {
            AssetRole::Projector => ModelAssetKind::Projector,
            AssetRole::Model | AssetRole::Unknown => ModelAssetKind::Model,
        });

        Ok(AssetInstallResult {
            record: AssetRecord {
                id,
                kind: inferred_kind,
                name,
                hash,
                bytes: metadata.len(),
                storage_path,
                source: AssetSource::Local {
                    path: source_path,
                    modified_unix_ms: source_modified_unix_ms,
                },
                ref_count: 0,
                created_at_unix_ms: now_unix_ms(),
                inspection: Some(inspection),
            },
            already_present,
        })
    }

    pub fn resolve_asset_path(&self, record: &AssetRecord) -> Result<PathBuf, ModelError> {
        let path = self.backend.resolve_storage_path(&record.storage_path);
        let metadata = fs::metadata(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ModelError::AssetMissing(record.id.clone())
            } else {
                ModelError::Io(error)
            }
        })?;
        if !metadata.is_file() || metadata.len() != record.bytes {
            return Err(ModelError::AssetMissing(record.id.clone()));
        }
        Ok(path)
    }

    pub fn delete_asset(&self, record: &AssetRecord) -> Result<(), ModelError> {
        let path = self.backend.resolve_storage_path(&record.storage_path);
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(ModelError::Io(error)),
        }
    }

    fn incoming_path(&self) -> PathBuf {
        self.backend
            .root()
            .join(INCOMING_DIR)
            .join(format!("asset-{}.tmp", unique_temp_suffix()))
    }
}

fn stage_local_path(source_path: &Path, tmp_path: &Path) -> Result<(), ModelError> {
    if let Some(parent) = tmp_path.parent() {
        fs::create_dir_all(parent)?;
    }

    if fs::hard_link(source_path, tmp_path).is_ok() {
        return Ok(());
    }

    let copy_result = (|| -> Result<(), ModelError> {
        let mut source = File::open(source_path)?;
        let mut tmp = File::create(tmp_path)?;
        let mut buffer = vec![0u8; COPY_BUFFER_BYTES];

        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            tmp.write_all(&buffer[..read])?;
        }
        tmp.sync_all()?;
        Ok(())
    })();

    if copy_result.is_err() {
        let _ = fs::remove_file(tmp_path);
    }
    copy_result
}

fn canonicalize_existing_path(path: &Path) -> Result<PathBuf, ModelError> {
    fs::canonicalize(path).map_err(ModelError::from)
}

#[cfg(test)]
mod tests {
    mod storage_tests;
}
