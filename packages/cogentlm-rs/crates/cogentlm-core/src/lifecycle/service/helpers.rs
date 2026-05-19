//! Small helpers used by `ModelService`: path comparison, asset classification,
//! pairing-state projection, fingerprinting, and hashing.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::engine::protocol::EngineState;
use crate::lifecycle::backend_policy::BackendPlan;
use crate::lifecycle::util::hex_lower;
use crate::lifecycle::{
    AssetInspection, AssetRecord, ClassifiedAsset, ModelEntry, ModelError, ModelInfo, ModelPairing,
    ModelPairingReason, ModelPairingState, ModelServiceState, ModelStatus, PairingPlan,
};

use crate::lifecycle::storage::now_unix_ms;

pub(super) fn same_path(left: &Path, right: &Path) -> bool {
    #[cfg(windows)]
    {
        left.to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

pub(super) fn classified_asset_from_record(record: &AssetRecord) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: record.id.clone(),
        name: record.name.clone(),
        inspection: record
            .inspection
            .clone()
            .unwrap_or_else(AssetInspection::unknown),
    }
}

pub(super) fn pairing_state_from_plan(plan: &PairingPlan) -> ModelPairing {
    let reason = match plan.status {
        ModelStatus::Ready => None,
        ModelStatus::NeedsProjector => Some(ModelPairingReason::NoMatch),
        ModelStatus::Broken => Some(ModelPairingReason::MissingMetadata),
    };
    ModelPairing {
        state: if plan.status == ModelStatus::Ready {
            ModelPairingState::Resolved
        } else {
            ModelPairingState::Unresolved
        },
        checked_projector_index_revision: 0,
        compatible_vision_projector_types: plan.compatible_vision_projector_types.clone(),
        reason,
        updated_at_unix_ms: now_unix_ms(),
    }
}

pub(super) fn model_id_from_plan(plan: &PairingPlan) -> String {
    let mut ids = plan.model_asset_ids.clone();
    if let Some(projector) = &plan.projector_asset_id {
        ids.push(projector.clone());
    }
    ids.sort();
    ids.dedup();
    format!("model-{}", stable_hash(ids.join("\n").as_bytes()))
}

pub(super) fn runtime_fingerprint(
    entry: &ModelEntry,
    backend_plan: &BackendPlan,
) -> Result<String, ModelError> {
    let mut model_asset_ids = entry.model_asset_ids.clone();
    model_asset_ids.sort();
    let runtime = serde_json::to_value(&backend_plan.config)?;
    let value = json!({
        "modelAssetIds": model_asset_ids,
        "projectorAssetId": entry.projector_asset_id,
        "backend": backend_plan.selection.selected,
        "runtime": runtime,
    });
    Ok(stable_hash(value.to_string().as_bytes()))
}

pub(super) fn stable_hash(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

pub(super) fn hash_file(path: &Path) -> Result<String, ModelError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

pub(super) fn service_state_from_engine_state(
    state: EngineState,
    model: ModelInfo,
) -> ModelServiceState {
    ModelServiceState {
        status: state.status,
        model: Some(model),
        backend: state.backend,
        runtime: state.runtime,
        requests: state.requests,
        stats: state.stats,
        updated_at_unix_ms: state.updated_at_unix_ms,
    }
}
