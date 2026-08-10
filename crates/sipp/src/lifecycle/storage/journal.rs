use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::lifecycle::util::storage_corrupt;
use crate::lifecycle::{AssetSource, ModelError, RegistryManifest};

use super::StorageBackend;

const JOURNAL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcquisitionJournalFile {
    version: u32,
    acquisition_id: String,
    storage_paths: BTreeSet<PathBuf>,
}

impl AcquisitionJournalFile {
    #[cfg(not(target_family = "wasm"))]
    fn empty(acquisition_id: String) -> Self {
        Self {
            version: JOURNAL_VERSION,
            acquisition_id,
            storage_paths: BTreeSet::new(),
        }
    }

    fn parse(path: &Path, bytes: &[u8]) -> Result<Self, ModelError> {
        let journal: Self = serde_json::from_slice(bytes).map_err(|error| {
            storage_corrupt(format!("failed to parse {}: {}", path.display(), error))
        })?;
        if journal.version != JOURNAL_VERSION {
            return Err(storage_corrupt(format!(
                "unsupported acquisition journal version {} in {}",
                journal.version,
                path.display()
            )));
        }
        for storage_path in &journal.storage_paths {
            validate_storage_path(storage_path)?;
        }
        Ok(journal)
    }
}

#[derive(Debug, Clone)]
#[cfg(not(target_family = "wasm"))]
pub(crate) struct AcquisitionJournal<B: StorageBackend> {
    backend: B,
    acquisition_id: String,
}

#[cfg(not(target_family = "wasm"))]
impl<B: StorageBackend> AcquisitionJournal<B> {
    pub(crate) fn new(backend: B, acquisition_id: String) -> Self {
        Self {
            backend,
            acquisition_id,
        }
    }

    pub(crate) fn record_path(&self, storage_path: &Path) -> Result<(), ModelError> {
        validate_storage_path(storage_path)?;
        let mut journal = self.read()?;
        journal.storage_paths.insert(storage_path.to_path_buf());
        self.write(&journal)
    }

    pub(crate) fn cleanup_uncommitted(
        &self,
        manifest: &RegistryManifest,
    ) -> Result<(), ModelError> {
        let journal = self.read()?;
        cleanup_entries(&self.backend, &journal, manifest)?;
        self.clear()
    }

    pub(crate) fn clear(&self) -> Result<(), ModelError> {
        remove_file_if_present(self.backend.incoming_journal_path(&self.acquisition_id))
    }

    fn read(&self) -> Result<AcquisitionJournalFile, ModelError> {
        let path = self.backend.incoming_journal_path(&self.acquisition_id);
        if !path.exists() {
            return Ok(AcquisitionJournalFile::empty(self.acquisition_id.clone()));
        }
        let journal = AcquisitionJournalFile::parse(&path, &fs::read(&path)?)?;
        if journal.acquisition_id != self.acquisition_id {
            return Err(storage_corrupt(format!(
                "acquisition journal id mismatch in {}",
                path.display()
            )));
        }
        Ok(journal)
    }

    fn write(&self, journal: &AcquisitionJournalFile) -> Result<(), ModelError> {
        self.backend.ensure_layout()?;
        let bytes = serde_json::to_vec_pretty(journal)?;
        self.backend.atomic_write(
            &self.backend.incoming_journal_path(&self.acquisition_id),
            &bytes,
        )
    }
}

pub(crate) fn recover_acquisition_journals<B: StorageBackend>(
    backend: &B,
    manifest: &RegistryManifest,
) -> Result<(), ModelError> {
    backend.ensure_layout()?;
    for entry in fs::read_dir(backend.incoming_journal_dir())? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        let path = entry.path();
        let journal = AcquisitionJournalFile::parse(&path, &fs::read(&path)?)?;
        cleanup_entries(backend, &journal, manifest)?;
        remove_file_if_present(path)?;
    }
    Ok(())
}

fn cleanup_entries<B: StorageBackend>(
    backend: &B,
    journal: &AcquisitionJournalFile,
    manifest: &RegistryManifest,
) -> Result<(), ModelError> {
    let protected: BTreeSet<_> = manifest
        .assets
        .values()
        .filter(|asset| matches!(&asset.source, AssetSource::Remote { .. }))
        .map(|asset| asset.storage_path.clone())
        .collect();
    for storage_path in &journal.storage_paths {
        if !protected.contains(storage_path) {
            remove_file_if_present(backend.resolve_storage_path(storage_path))?;
        }
    }
    Ok(())
}

fn validate_storage_path(path: &Path) -> Result<(), ModelError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(invalid_storage_path(path));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => return Err(invalid_storage_path(path)),
        }
    }
    Ok(())
}

fn invalid_storage_path(path: &Path) -> ModelError {
    storage_corrupt(format!(
        "invalid acquisition journal storage path: {}",
        path.display()
    ))
}

fn remove_file_if_present(path: PathBuf) -> Result<(), ModelError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ModelError::Io(error)),
    }
}
