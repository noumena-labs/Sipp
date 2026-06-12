//! Small helpers used by `ModelService`: path comparison, asset classification,
//! pairing-state projection, and fingerprinting.

use std::path::Path;

use serde_json::json;

use crate::collection::sorted_values;
use crate::lifecycle::backend_policy::BackendPlan;
use crate::lifecycle::util::{sha256_hex, sorted_model_asset_ids};
use crate::lifecycle::{ModelEntry, ModelError, PairingPlan};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/service/helpers_tests.rs"]
mod helpers_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

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

pub(super) fn model_id_from_plan(plan: &PairingPlan) -> String {
    let ids = sorted_model_asset_ids(&plan.model_asset_ids, plan.projector_asset_id.as_ref());
    format!("model-{}", sha256_hex(ids.join("\n").as_bytes()))
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
