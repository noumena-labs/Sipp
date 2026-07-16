use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(not(target_family = "wasm"))]
use crate::lifecycle::acquisition::native::NativeRemoteExecutor;
use crate::lifecycle::acquisition::{
    RemoteAcquisition, RemoteAcquisitionProgress, RemoteAcquisitionRequest, RemoteAssetRole,
    RemoteCacheCandidate, RemoteMetadata,
};
use crate::lifecycle::registry::model_entry_from_assets;
use crate::lifecycle::storage::{hash_file, modified_unix_ms, now_unix_ms, StorageBackend};
use crate::lifecycle::util::classified_asset;
use crate::lifecycle::{
    AssetRecord, AssetSource, ModelAssetKind, ModelError, ModelPairing, ModelPairingReason,
    ModelPairingState, ModelSource, ModelStatus, PairingResolver,
};

use super::helpers::{model_id_from_plan, same_path};
use super::{invalid_source, model_not_found, ModelService, ResolvedSource};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/service/source_resolution_tests.rs"]
mod source_resolution_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const MODEL_PATHS_REQUIRED: &str = "local model paths must not be empty";
const MODEL_URLS_REQUIRED: &str = "remote model URLs must not be empty";

#[derive(Debug)]
struct InstalledAsset {
    record: AssetRecord,
    created: bool,
}

impl<B: StorageBackend> ModelService<B> {
    pub(super) async fn resolve_source(
        &mut self,
        source: ModelSource,
    ) -> Result<ResolvedSource, ModelError> {
        match source {
            ModelSource::Installed { model_id } => self.resolve_installed(model_id),
            ModelSource::Local {
                model_paths,
                projector_path,
            } => self.resolve_local(model_paths, projector_path),
            ModelSource::Remote {
                model_urls,
                projector_url,
            } => self.resolve_remote(model_urls, projector_url).await,
        }
    }

    fn resolve_installed(&self, model_id: String) -> Result<ResolvedSource, ModelError> {
        if !self.registry.manifest.models.contains_key(&model_id) {
            return Err(model_not_found(&model_id));
        }
        Ok(ResolvedSource { entry_id: model_id })
    }

    fn resolve_local(
        &mut self,
        model_paths: Vec<PathBuf>,
        projector_path: Option<PathBuf>,
    ) -> Result<ResolvedSource, ModelError> {
        if model_paths.is_empty() {
            return Err(invalid_source(MODEL_PATHS_REQUIRED));
        }
        let model_kind = (model_paths.len() > 1).then_some(ModelAssetKind::Shard);
        let mut installed = Vec::new();
        for path in model_paths {
            match self.install_local_asset(path, model_kind) {
                Ok(asset) => installed.push(asset),
                Err(error) => return self.fail_local_install(installed, error),
            }
        }
        let explicit_projector_id = if let Some(path) = projector_path {
            match self.install_local_asset(path, Some(ModelAssetKind::Projector)) {
                Ok(projector) => {
                    let id = projector.record.id.clone();
                    installed.push(projector);
                    Some(id)
                }
                Err(error) => return self.fail_local_install(installed, error),
            }
        } else {
            None
        };
        self.commit_installed(installed, explicit_projector_id.as_deref())
    }

    fn fail_local_install<T>(
        &self,
        installed: Vec<InstalledAsset>,
        error: ModelError,
    ) -> Result<T, ModelError> {
        for asset in installed.iter().filter(|asset| asset.created) {
            self.assets.delete_asset(&asset.record)?;
        }
        Err(error)
    }

    #[cfg(not(target_family = "wasm"))]
    async fn resolve_remote(
        &mut self,
        model_urls: Vec<String>,
        projector_url: Option<String>,
    ) -> Result<ResolvedSource, ModelError> {
        if model_urls.is_empty() {
            return Err(invalid_source(MODEL_URLS_REQUIRED));
        }
        let acquisition_id = self.acquisition_ids.issue()?;
        let journal = self.assets.acquisition_journal(acquisition_id.clone());
        let model_role = if model_urls.len() == 1 {
            RemoteAssetRole::Model
        } else {
            RemoteAssetRole::Shard
        };
        let mut sources: Vec<_> = model_urls
            .into_iter()
            .map(|url| (model_role, url))
            .collect();
        if let Some(projector_url) = projector_url {
            sources.push((RemoteAssetRole::Projector, projector_url));
        }
        let requests = remote_requests(&self.registry.manifest.assets, sources)?;

        let mut acquisition = RemoteAcquisition::new(acquisition_id, requests)?;
        let executor = NativeRemoteExecutor::new(self.assets.clone(), journal.clone())?;
        let mut downloaded = BTreeMap::new();
        let mut progress = acquisition.progress();
        loop {
            match progress {
                RemoteAcquisitionProgress::Action(action) => {
                    let event = executor
                        .execute(action, &self.registry.manifest, &mut downloaded)
                        .await;
                    progress = match acquisition.advance(event) {
                        Ok(progress) => progress,
                        Err(error) => {
                            journal.cleanup_uncommitted(&self.registry.manifest)?;
                            return Err(error);
                        }
                    };
                }
                RemoteAcquisitionProgress::Ready(resolved) => {
                    let records =
                        match resolved_records(&resolved, &downloaded, &self.registry.manifest) {
                            Ok(records) => records,
                            Err(error) => {
                                journal.cleanup_uncommitted(&self.registry.manifest)?;
                                return Err(error);
                            }
                        };
                    let installed: Vec<_> = records
                        .into_iter()
                        .map(|record| InstalledAsset {
                            created: resolved.iter().any(|member| {
                                member.created_asset_ids.iter().any(|id| id == &record.id)
                            }),
                            record,
                        })
                        .collect();
                    let projector_id = resolved
                        .iter()
                        .find(|member| member.role == RemoteAssetRole::Projector)
                        .and_then(|member| member.asset_ids.first())
                        .map(String::as_str);
                    return match self.commit_installed(installed, projector_id) {
                        Ok(resolved) => {
                            journal.clear()?;
                            Ok(resolved)
                        }
                        Err(error) => {
                            journal.cleanup_uncommitted(&self.registry.manifest)?;
                            Err(error)
                        }
                    };
                }
                RemoteAcquisitionProgress::Failed(error) => {
                    journal.cleanup_uncommitted(&self.registry.manifest)?;
                    return Err(error);
                }
                RemoteAcquisitionProgress::Cancelled => {
                    journal.cleanup_uncommitted(&self.registry.manifest)?;
                    return Err(ModelError::AcquisitionCancelled);
                }
            }
        }
    }

    #[cfg(target_family = "wasm")]
    async fn resolve_remote(
        &mut self,
        _model_urls: Vec<String>,
        _projector_url: Option<String>,
    ) -> Result<ResolvedSource, ModelError> {
        Err(ModelError::UnsupportedOperation {
            operation: "native model service remote acquisition",
            reason: "browser acquisition is driven through BrowserLifecycleService".to_string(),
        })
    }

    fn install_local_asset(
        &self,
        path: impl AsRef<Path>,
        kind: Option<ModelAssetKind>,
    ) -> Result<InstalledAsset, ModelError> {
        let path = path.as_ref();
        if let Some(record) = self.find_cached_local_asset(path, kind)? {
            return Ok(InstalledAsset {
                record,
                created: false,
            });
        }
        let installed = self.assets.install_local_path_as(path, kind)?;
        Ok(InstalledAsset {
            record: installed.record,
            created: !installed.already_present,
        })
    }

    fn find_cached_local_asset(
        &self,
        path: &Path,
        kind: Option<ModelAssetKind>,
    ) -> Result<Option<AssetRecord>, ModelError> {
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() {
            return Ok(None);
        }

        let source_path = fs::canonicalize(path)?;
        let source_modified_unix_ms = modified_unix_ms(&metadata);
        for record in self.registry.manifest.assets.values() {
            if cached_local_record_matches(
                record,
                kind,
                metadata.len(),
                &source_path,
                source_modified_unix_ms,
            ) && self.assets.resolve_asset_path(record).is_ok()
                && hash_file(path).is_ok_and(|hash| hash == record.hash)
            {
                return Ok(Some(record.clone()));
            }
        }
        Ok(None)
    }

    fn commit_installed(
        &mut self,
        installed: Vec<InstalledAsset>,
        explicit_projector_id: Option<&str>,
    ) -> Result<ResolvedSource, ModelError> {
        let previous = self.registry.manifest.clone();
        let result = self.register_installed_assets(
            &installed
                .iter()
                .map(|asset| asset.record.clone())
                .collect::<Vec<_>>(),
            explicit_projector_id,
        );
        if result.is_err() {
            self.registry.manifest = previous;
            for asset in installed.iter().filter(|asset| asset.created) {
                self.assets.delete_asset(&asset.record)?;
            }
        }
        result
    }

    fn register_installed_assets(
        &mut self,
        installed: &[AssetRecord],
        explicit_projector_id: Option<&str>,
    ) -> Result<ResolvedSource, ModelError> {
        let classified: Vec<_> = installed
            .iter()
            .map(|record| {
                classified_asset(
                    record.id.clone(),
                    record.name.clone(),
                    record.inspection.clone(),
                )
            })
            .collect();
        let plan = if let Some(projector_id) = explicit_projector_id {
            PairingResolver::resolve_explicit(&classified, projector_id)?
        } else {
            PairingResolver::resolve(&classified)?
        };
        for record in installed {
            self.registry.upsert_asset(record.clone())?;
        }
        let entry_id = model_id_from_plan(&plan);
        let mut entry = model_entry_from_assets(&entry_id, &plan.name, &plan);
        entry.pairing = Some(ModelPairing {
            state: if plan.status == ModelStatus::Ready {
                ModelPairingState::Resolved
            } else {
                ModelPairingState::Unresolved
            },
            checked_projector_index_revision: 0,
            compatible_vision_projector_types: plan.compatible_vision_projector_types.clone(),
            reason: match plan.status {
                ModelStatus::Ready => None,
                ModelStatus::NeedsProjector => Some(ModelPairingReason::NoMatch),
                ModelStatus::Broken => Some(ModelPairingReason::MissingMetadata),
            },
            updated_at_unix_ms: now_unix_ms(),
        });
        self.registry.insert_model(entry)?;
        self.registry.save()?;
        Ok(ResolvedSource { entry_id })
    }
}

fn remote_requests(
    assets: &BTreeMap<String, AssetRecord>,
    sources: impl IntoIterator<Item = (RemoteAssetRole, String)>,
) -> Result<Vec<RemoteAcquisitionRequest>, ModelError> {
    sources
        .into_iter()
        .enumerate()
        .map(|(index, (role, url))| {
            let url = crate::lifecycle::acquisition::canonical_remote_url(&url)?;
            Ok(RemoteAcquisitionRequest {
                member_id: u32::try_from(index)
                    .map_err(|_| invalid_source("remote model contains too many source members"))?,
                role,
                candidates: remote_candidates(assets, role, &url),
                url,
            })
        })
        .collect()
}

fn remote_candidates(
    assets: &BTreeMap<String, AssetRecord>,
    role: RemoteAssetRole,
    url: &str,
) -> Vec<RemoteCacheCandidate> {
    assets
        .values()
        .filter(|record| record.kind == role.asset_kind())
        .filter_map(|record| {
            let AssetSource::Remote {
                url: record_url,
                etag,
                last_modified,
            } = &record.source
            else {
                return None;
            };
            (record_url == url).then(|| RemoteCacheCandidate {
                candidate_id: record.id.clone(),
                asset_ids: vec![record.id.clone()],
                metadata: RemoteMetadata {
                    url: record_url.clone(),
                    name: record.name.clone(),
                    bytes: record.bytes,
                    etag: etag.clone(),
                    last_modified: last_modified.clone(),
                },
            })
        })
        .collect()
}

fn resolved_records(
    resolved: &[crate::lifecycle::acquisition::RemoteResolvedMember],
    downloaded: &BTreeMap<String, AssetRecord>,
    manifest: &crate::lifecycle::RegistryManifest,
) -> Result<Vec<AssetRecord>, ModelError> {
    resolved
        .iter()
        .flat_map(|member| member.asset_ids.iter())
        .map(|asset_id| {
            downloaded
                .get(asset_id)
                .or_else(|| manifest.assets.get(asset_id))
                .cloned()
                .ok_or_else(|| ModelError::AssetMissing(asset_id.clone()))
        })
        .collect()
}

fn cached_local_record_matches(
    record: &AssetRecord,
    kind: Option<ModelAssetKind>,
    source_bytes: u64,
    source_path: &Path,
    source_modified_unix_ms: Option<u64>,
) -> bool {
    if kind.is_some_and(|expected| record.kind != expected) || record.bytes != source_bytes {
        return false;
    }
    let AssetSource::Local {
        path: record_source_path,
        modified_unix_ms: record_modified_unix_ms,
    } = &record.source
    else {
        return false;
    };
    same_path(record_source_path, source_path)
        && match (*record_modified_unix_ms, source_modified_unix_ms) {
            (Some(record), Some(source)) => record == source,
            _ => true,
        }
}
