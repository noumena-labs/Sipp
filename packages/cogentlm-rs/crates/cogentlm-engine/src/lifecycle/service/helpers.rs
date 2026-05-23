//! Small helpers used by `ModelService`: path comparison, asset classification,
//! pairing-state projection, and fingerprinting.

use std::path::Path;

use serde_json::json;

use crate::collection::sorted_values;
use crate::engine::protocol::EngineState;
use crate::lifecycle::backend_policy::BackendPlan;
use crate::lifecycle::util::{
    asset_summary, classified_asset, model_id_from_fingerprint, sha256_hex, sorted_model_asset_ids,
    AssetSummary,
};
use crate::lifecycle::{
    AssetRecord, AssetSource, ClassifiedAsset, ModelEntry, ModelError, ModelInfo, ModelPairing,
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
    classified_asset(
        record.id.clone(),
        record.name.clone(),
        record.inspection.clone(),
    )
}

pub(super) fn model_asset_summary<'asset>(
    assets: impl Iterator<Item = &'asset AssetRecord>,
) -> AssetSummary {
    asset_summary(assets.map(|asset| {
        (
            asset.bytes,
            matches!(asset.source, AssetSource::Remote { .. }),
        )
    }))
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
    let ids = sorted_model_asset_ids(&plan.model_asset_ids, plan.projector_asset_id.as_ref());
    model_id_from_fingerprint(&sha256_hex(ids.join("\n").as_bytes()))
}

pub(super) fn runtime_fingerprint(
    entry: &ModelEntry,
    backend_plan: &BackendPlan,
) -> Result<String, ModelError> {
    let runtime = serde_json::to_value(&backend_plan.config)?;
    let value = json!({
        "modelAssetIds": sorted_values(entry.model_asset_ids.clone()),
        "projectorAssetId": entry.projector_asset_id,
        "backend": backend_plan.selection.selected,
        "runtime": runtime,
    });
    Ok(sha256_hex(value.to_string().as_bytes()))
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
