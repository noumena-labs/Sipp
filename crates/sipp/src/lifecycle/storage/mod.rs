//! Filesystem layout for stored model assets and registry manifests.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use super::util::{invalid_source, storage_corrupt};
use super::{
    detect_model_from_gguf_bytes, AssetRecord, AssetRole, AssetSource, LocalPathAnchor,
    ModelAssetKind, ModelError,
};
use crate::defaults::BYTES_PER_MIB;

mod content;
mod journal;
mod metadata;

#[cfg(not(target_family = "wasm"))]
pub(crate) use content::hash_file;
use content::inspect_local_path;
#[cfg(not(target_family = "wasm"))]
pub(crate) use journal::AcquisitionJournal;
#[cfg(not(target_family = "wasm"))]
use metadata::normalize_asset_file_name;
pub(crate) use metadata::{modified_unix_ms, now_unix_ms};
use metadata::{normalize_asset_name, unique_temp_suffix};

const ASSETS_DIR: &str = "assets";
const INCOMING_DIR: &str = ".incoming";
const JOURNALS_DIR: &str = "journals";
const REGISTRY_FILE_NAME: &str = "registry.json";
const ASSET_ID_PREFIX: &str = "asset-";
pub(super) const COPY_BUFFER_BYTES: usize = BYTES_PER_MIB;

#[cfg(not(target_family = "wasm"))]
fn asset_integrity_error(asset_id: &str, reason: &str) -> ModelError {
    storage_corrupt(format!("asset {asset_id} has hash match but {reason}"))
}

fn asset_missing(asset_id: &str) -> ModelError {
    ModelError::AssetMissing(asset_id.to_string())
}

fn incoming_asset_file_name() -> String {
    format!("{ASSET_ID_PREFIX}{}.tmp", unique_temp_suffix())
}

pub trait StorageBackend: Clone + Send + Sync + 'static {
    fn root(&self) -> &Path;

    /// Return the relocatable root used by sandbox-owned local model files.
    fn local_source_root(&self) -> Option<&Path> {
        None
    }

    fn manifest_path(&self) -> PathBuf {
        self.root().join(REGISTRY_FILE_NAME)
    }

    fn asset_storage_path(&self, asset_id: &str) -> PathBuf {
        PathBuf::from(ASSETS_DIR).join(asset_id)
    }

    fn asset_path(&self, asset_id: &str) -> PathBuf {
        self.root().join(self.asset_storage_path(asset_id))
    }

    fn incoming_storage_path(&self) -> PathBuf {
        PathBuf::from(INCOMING_DIR).join(incoming_asset_file_name())
    }

    fn incoming_journal_dir(&self) -> PathBuf {
        self.root().join(INCOMING_DIR).join(JOURNALS_DIR)
    }

    fn incoming_journal_path(&self, acquisition_id: &str) -> PathBuf {
        self.incoming_journal_dir()
            .join(format!("{acquisition_id}.json"))
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
        fs::create_dir_all(self.incoming_journal_dir())?;
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
    local_source_root: Option<PathBuf>,
}

impl LocalStorageBackend {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            local_source_root: None,
        }
    }

    pub(crate) fn with_local_source_root(
        root: impl Into<PathBuf>,
        local_source_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            root: root.into(),
            local_source_root: Some(local_source_root.into()),
        }
    }
}

impl StorageBackend for LocalStorageBackend {
    fn root(&self) -> &Path {
        &self.root
    }

    fn local_source_root(&self) -> Option<&Path> {
        self.local_source_root.as_deref()
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

    pub fn register_local_path(&self, path: impl AsRef<Path>) -> Result<AssetRecord, ModelError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() {
            return Err(invalid_source(format!(
                "model asset is not a file: {}",
                path.display()
            )));
        }

        let source_path = canonicalize_existing_path(path)?;
        let source_modified_unix_ms = modified_unix_ms(&metadata);
        let name = normalize_asset_name(path);
        let (hash, prefix) = inspect_local_path(path)?;
        let inspection = detect_model_from_gguf_bytes(&name, &prefix)?.inspection;
        let kind = match inspection.role {
            AssetRole::Projector => ModelAssetKind::Projector,
            AssetRole::Model | AssetRole::Unknown => ModelAssetKind::Model,
        };
        let (stored_path, anchor) = self.local_source_locator(&source_path)?;

        Ok(AssetRecord {
            id: format!("{ASSET_ID_PREFIX}{hash}"),
            kind,
            name,
            hash,
            bytes: metadata.len(),
            storage_path: stored_path.clone(),
            source: AssetSource::Local {
                path: stored_path,
                anchor,
                modified_unix_ms: source_modified_unix_ms,
            },
            ref_count: 0,
            created_at_unix_ms: now_unix_ms(),
            inspection: Some(inspection),
        })
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn install_remote_staged(
        &self,
        staged_path: &Path,
        metadata: &super::acquisition::RemoteMetadata,
        journal: Option<&AcquisitionJournal<B>>,
    ) -> Result<AssetInstallResult, ModelError> {
        let bytes = fs::metadata(staged_path)?.len();
        if bytes != metadata.bytes {
            return Err(ModelError::RemoteIntegrityFailed {
                url: metadata.url.clone(),
                reason: format!(
                    "downloaded byte length is {bytes}, expected {}",
                    metadata.bytes
                ),
            });
        }
        self.install_managed_path(
            staged_path,
            normalize_asset_file_name(&metadata.name),
            AssetSource::Remote {
                url: metadata.url.clone(),
                etag: metadata.etag.clone(),
                last_modified: metadata.last_modified.clone(),
            },
            journal,
        )
    }

    pub fn resolve_asset_path(&self, record: &AssetRecord) -> Result<PathBuf, ModelError> {
        let path = match &record.source {
            AssetSource::Local { path, anchor, .. } => {
                self.resolve_local_source_path(path, *anchor)?
            }
            AssetSource::Remote { .. } => self.backend.resolve_storage_path(&record.storage_path),
        };
        let metadata = fs::metadata(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                asset_missing(&record.id)
            } else {
                ModelError::Io(error)
            }
        })?;
        if !metadata.is_file() || metadata.len() != record.bytes {
            return Err(asset_missing(&record.id));
        }
        Ok(path)
    }

    pub(crate) fn requires_external_access(&self, record: &AssetRecord) -> bool {
        self.backend.local_source_root().is_some()
            && matches!(
                &record.source,
                AssetSource::Local {
                    anchor: LocalPathAnchor::Absolute,
                    ..
                }
            )
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn validate_asset(&self, record: &AssetRecord) -> Result<(), ModelError> {
        let path = self.resolve_asset_path(record)?;
        if hash_file(&path)? != record.hash {
            return Err(asset_integrity_error(&record.id, "content mismatch"));
        }
        Ok(())
    }

    pub fn delete_managed_asset(&self, record: &AssetRecord) -> Result<(), ModelError> {
        if matches!(record.source, AssetSource::Local { .. }) {
            return Ok(());
        }
        let path = self.backend.resolve_storage_path(&record.storage_path);
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(ModelError::Io(error)),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn incoming_storage_path(&self) -> PathBuf {
        self.backend.incoming_storage_path()
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn resolve_storage_path(&self, storage_path: &Path) -> PathBuf {
        self.backend.resolve_storage_path(storage_path)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn acquisition_journal(
        &self,
        acquisition_id: impl Into<String>,
    ) -> AcquisitionJournal<B> {
        AcquisitionJournal::new(self.backend.clone(), acquisition_id.into())
    }

    pub(crate) fn recover_acquisition_journals(
        &self,
        manifest: &super::RegistryManifest,
    ) -> Result<(), ModelError> {
        journal::recover_acquisition_journals(&self.backend, manifest)
    }

    fn local_source_locator(
        &self,
        source_path: &Path,
    ) -> Result<(PathBuf, LocalPathAnchor), ModelError> {
        let Some(source_root) = self.backend.local_source_root() else {
            return Ok((source_path.to_path_buf(), LocalPathAnchor::Absolute));
        };
        let source_root = canonicalize_existing_path(source_root)?;
        let Ok(relative_path) = source_path.strip_prefix(source_root) else {
            return Ok((source_path.to_path_buf(), LocalPathAnchor::Absolute));
        };
        if relative_path.as_os_str().is_empty() {
            return Err(invalid_source(
                "model asset path resolves to the local source root",
            ));
        }
        Ok((relative_path.to_path_buf(), LocalPathAnchor::SourceRoot))
    }

    fn resolve_local_source_path(
        &self,
        path: &Path,
        anchor: LocalPathAnchor,
    ) -> Result<PathBuf, ModelError> {
        match anchor {
            LocalPathAnchor::Absolute => Ok(path.to_path_buf()),
            LocalPathAnchor::SourceRoot => {
                if path.is_absolute()
                    || path.components().any(|component| {
                        matches!(
                            component,
                            Component::ParentDir | Component::RootDir | Component::Prefix(_)
                        )
                    })
                {
                    return Err(storage_corrupt(format!(
                        "local source-root path is invalid: {}",
                        path.display()
                    )));
                }
                let source_root = self.backend.local_source_root().ok_or_else(|| {
                    ModelError::StorageUnavailable(
                        "registry contains a source-root-relative model but no local source root was configured"
                            .to_string(),
                    )
                })?;
                Ok(source_root.join(path))
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn install_managed_path(
        &self,
        path: &Path,
        name: String,
        source: AssetSource,
        journal: Option<&AcquisitionJournal<B>>,
    ) -> Result<AssetInstallResult, ModelError> {
        self.backend.ensure_layout()?;
        let metadata = fs::metadata(path)?;
        let (hash, prefix) = inspect_local_path(path)?;
        let id = format!("{ASSET_ID_PREFIX}{hash}");
        let storage_path = self.backend.asset_storage_path(&id);
        let final_path = self.backend.asset_path(&id);
        let already_present = final_path.exists();

        if already_present {
            validate_existing_asset(&final_path, metadata.len(), &hash, &id)?;
        }

        let inspection = detect_model_from_gguf_bytes(&name, &prefix)?.inspection;
        let kind = match inspection.role {
            AssetRole::Projector => ModelAssetKind::Projector,
            AssetRole::Model | AssetRole::Unknown => ModelAssetKind::Model,
        };

        if !already_present {
            if let Some(journal) = journal {
                journal.record_path(&storage_path)?;
            }
            publish_staged_asset(path, &final_path)?;
        }

        Ok(AssetInstallResult {
            record: AssetRecord {
                id,
                kind,
                name,
                hash,
                bytes: metadata.len(),
                storage_path,
                source,
                ref_count: 0,
                created_at_unix_ms: now_unix_ms(),
                inspection: Some(inspection),
            },
            already_present,
        })
    }
}

#[cfg(not(target_family = "wasm"))]
fn validate_existing_asset(
    path: &Path,
    expected_bytes: u64,
    expected_hash: &str,
    asset_id: &str,
) -> Result<(), ModelError> {
    if fs::metadata(path)?.len() != expected_bytes {
        return Err(asset_integrity_error(asset_id, "byte-size mismatch"));
    }
    if hash_file(path)? != expected_hash {
        return Err(asset_integrity_error(asset_id, "content mismatch"));
    }
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
fn publish_staged_asset(tmp_path: &Path, final_path: &Path) -> Result<(), ModelError> {
    create_parent_dir(final_path)?;
    match fs::rename(tmp_path, final_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(tmp_path)?;
            Ok(())
        }
        Err(error) => Err(ModelError::Io(error)),
    }
}

#[cfg(not(target_family = "wasm"))]
fn create_parent_dir(path: &Path) -> Result<(), ModelError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn canonicalize_existing_path(path: &Path) -> Result<PathBuf, ModelError> {
    fs::canonicalize(path).map_err(ModelError::from)
}

#[cfg(test)]
#[path = "../../tests/lifecycle/storage_tests.rs"]
mod storage_tests;
