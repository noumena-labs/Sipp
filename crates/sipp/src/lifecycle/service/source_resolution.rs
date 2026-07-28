use std::collections::BTreeMap;
use std::path::PathBuf;
use url::Url;

#[cfg(not(target_family = "wasm"))]
use crate::lifecycle::acquisition::native::NativeRemoteExecutor;
use crate::lifecycle::acquisition::{
    RemoteAcquisition, RemoteAcquisitionProgress, RemoteAcquisitionRequest, RemoteCacheCandidate,
    RemoteMetadata,
};
use crate::lifecycle::registry::model_entry_from_assets;
use crate::lifecycle::storage::{now_unix_ms, StorageBackend};
use crate::lifecycle::util::classified_asset;
use crate::lifecycle::{
    AssetRecord, AssetSource, LocalPathAnchor, ModelError, ModelPairing, ModelPairingReason,
    ModelPairingState, ModelStatus, PairingResolver,
};

use super::helpers::model_id_from_plan;
use super::{invalid_source, ModelStoreState};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/service/source_resolution_tests.rs"]
mod source_resolution_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const MODEL_SOURCES_REQUIRED: &str = "model sources must not be empty";

enum ResolvedSources {
    Local(Vec<PathBuf>),
    Remote(Vec<String>),
}

#[derive(Debug)]
struct AcquiredAsset {
    record: AssetRecord,
    created: bool,
}

pub(super) struct ModelAddOutcome {
    pub(super) model_id: String,
    pub(super) created: bool,
}

fn resolve_sources(sources: Vec<PathBuf>) -> Result<ResolvedSources, ModelError> {
    if sources.is_empty() {
        return Err(invalid_source(MODEL_SOURCES_REQUIRED));
    }
    let mut local = Vec::new();
    let mut remote = Vec::new();
    for source in sources {
        let Some(value) = source.to_str() else {
            local.push(source);
            continue;
        };
        match Url::parse(value) {
            Ok(url) if matches!(url.scheme(), "http" | "https") => {
                remote.push(crate::lifecycle::acquisition::canonical_remote_url(value)?)
            }
            Ok(url) if value.contains("://") => {
                return Err(invalid_source(format!(
                    "model URL scheme must be http or https, not {}",
                    url.scheme()
                )));
            }
            Err(error) if has_http_scheme(value) => {
                return Err(invalid_source(format!("model URL is invalid: {error}")));
            }
            _ => local.push(source),
        }
    }
    match (local.is_empty(), remote.is_empty()) {
        (false, true) => Ok(ResolvedSources::Local(local)),
        (true, false) => Ok(ResolvedSources::Remote(remote)),
        _ => Err(invalid_source(
            "local files and remote URLs cannot be added together",
        )),
    }
}

fn has_http_scheme(value: &str) -> bool {
    value.split_once(':').is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    })
}

impl<B: StorageBackend> ModelStoreState<B> {
    pub(super) async fn add(
        &mut self,
        sources: Vec<PathBuf>,
    ) -> Result<ModelAddOutcome, ModelError> {
        match resolve_sources(sources)? {
            ResolvedSources::Local(paths) => self.add_local(paths),
            ResolvedSources::Remote(urls) => self.add_remote(urls).await,
        }
    }

    fn add_local(&mut self, paths: Vec<PathBuf>) -> Result<ModelAddOutcome, ModelError> {
        let records = paths
            .into_iter()
            .map(|path| self.assets.register_local_path(path))
            .collect::<Result<Vec<_>, _>>()?;
        self.register_assets(&records)
    }

    #[cfg(not(target_family = "wasm"))]
    async fn add_remote(&mut self, urls: Vec<String>) -> Result<ModelAddOutcome, ModelError> {
        let acquisition_id = self.acquisition_ids.issue()?;
        let journal = self.assets.acquisition_journal(acquisition_id.clone());
        let requests = remote_requests(&self.registry.manifest.assets, urls)?;

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
                    let acquired: Vec<_> = records
                        .into_iter()
                        .map(|record| AcquiredAsset {
                            created: resolved.iter().any(|member| {
                                member.created_asset_ids.iter().any(|id| id == &record.id)
                            }),
                            record,
                        })
                        .collect();
                    return match self.commit_acquired(acquired) {
                        Ok(model_id) => {
                            journal.clear()?;
                            Ok(model_id)
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
    async fn add_remote(&mut self, _urls: Vec<String>) -> Result<ModelAddOutcome, ModelError> {
        Err(ModelError::UnsupportedOperation {
            operation: "native model service remote acquisition",
            reason: "browser acquisition is driven through BrowserLifecycleService".to_string(),
        })
    }

    fn commit_acquired(
        &mut self,
        acquired: Vec<AcquiredAsset>,
    ) -> Result<ModelAddOutcome, ModelError> {
        let previous = self.registry.manifest.clone();
        let result = self.register_assets(
            &acquired
                .iter()
                .map(|asset| asset.record.clone())
                .collect::<Vec<_>>(),
        );
        if result.is_err() {
            self.registry.manifest = previous;
            for asset in acquired.iter().filter(|asset| asset.created) {
                self.assets.delete_managed_asset(&asset.record)?;
            }
        }
        result
    }

    fn register_assets(&mut self, records: &[AssetRecord]) -> Result<ModelAddOutcome, ModelError> {
        let displaced_managed_assets: Vec<_> = records
            .iter()
            .filter(|record| matches!(&record.source, AssetSource::Local { .. }))
            .filter_map(|record| self.registry.manifest.assets.get(&record.id))
            .filter(|record| matches!(&record.source, AssetSource::Remote { .. }))
            .cloned()
            .collect();
        let classified: Vec<_> = records
            .iter()
            .map(|record| {
                classified_asset(
                    record.id.clone(),
                    record.name.clone(),
                    record.inspection.clone(),
                )
            })
            .collect();
        let plan = PairingResolver::resolve(&classified)?;
        let entry_id = model_id_from_plan(&plan);
        let created = !self.registry.manifest.models.contains_key(&entry_id);
        let source_key = source_key(records);
        let replaced: Vec<_> = self
            .registry
            .manifest
            .models
            .values()
            .filter(|entry| entry.id != entry_id)
            .map(|entry| {
                Ok(
                    (entry_source_key(&self.registry.manifest, entry)? == source_key)
                        .then_some(entry.id.clone()),
                )
            })
            .collect::<Result<Vec<_>, ModelError>>()?
            .into_iter()
            .flatten()
            .collect();
        let mut orphaned = Vec::new();
        for model_id in replaced {
            orphaned.extend(self.registry.remove_model(&model_id)?.orphaned_assets);
        }
        for record in records {
            self.registry.upsert_asset(record.clone())?;
        }
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
        for asset in orphaned {
            self.assets.delete_managed_asset(&asset)?;
        }
        for asset in displaced_managed_assets {
            self.assets.delete_managed_asset(&asset)?;
        }
        Ok(ModelAddOutcome {
            model_id: entry_id,
            created,
        })
    }
}

fn source_key(records: &[AssetRecord]) -> Vec<String> {
    let mut sources: Vec<_> = records.iter().map(asset_source_key).collect();
    sources.sort();
    sources
}

fn entry_source_key(
    manifest: &crate::lifecycle::RegistryManifest,
    entry: &crate::lifecycle::ModelEntry,
) -> Result<Vec<String>, ModelError> {
    let records = entry
        .model_asset_ids
        .iter()
        .chain(entry.projector_asset_id.iter())
        .map(|asset_id| {
            manifest
                .assets
                .get(asset_id)
                .cloned()
                .ok_or_else(|| ModelError::AssetMissing(asset_id.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(source_key(&records))
}

fn asset_source_key(record: &AssetRecord) -> String {
    match &record.source {
        AssetSource::Local { path, anchor, .. } => local_source_key(path, *anchor),
        AssetSource::Remote { url, .. } => format!("remote:{url}"),
    }
}

fn local_source_key(path: &std::path::Path, anchor: LocalPathAnchor) -> String {
    let anchor = match anchor {
        LocalPathAnchor::Absolute => "absolute",
        LocalPathAnchor::SourceRoot => "source-root",
    };
    #[cfg(windows)]
    {
        format!(
            "local:{anchor}:{}",
            path.to_string_lossy().to_ascii_lowercase()
        )
    }
    #[cfg(not(windows))]
    {
        format!("local:{anchor}:{}", path.display())
    }
}

fn remote_requests(
    assets: &BTreeMap<String, AssetRecord>,
    urls: impl IntoIterator<Item = String>,
) -> Result<Vec<RemoteAcquisitionRequest>, ModelError> {
    urls.into_iter()
        .enumerate()
        .map(|(index, url)| {
            let url = crate::lifecycle::acquisition::canonical_remote_url(&url)?;
            Ok(RemoteAcquisitionRequest {
                member_id: u32::try_from(index)
                    .map_err(|_| invalid_source("remote model contains too many source members"))?,
                candidates: remote_candidates(assets, &url),
                url,
            })
        })
        .collect()
}

fn remote_candidates(
    assets: &BTreeMap<String, AssetRecord>,
    url: &str,
) -> Vec<RemoteCacheCandidate> {
    assets
        .values()
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
