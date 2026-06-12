//! Pairing of model assets (weights + projector) into a runnable ModelEntry.

use super::types::{
    AssetRole, ClassifiedAsset, ModelError, ModelModality, ModelStatus, PairingPlan,
};
use crate::collection::sorted_unique_strings;
use crate::lifecycle::util::{invalid_pairing, invalid_source};

#[derive(Debug, Clone, Copy, Default)]
pub struct PairingResolver;

#[derive(Debug)]
struct AssetSelection<'a> {
    model_files: Vec<&'a ClassifiedAsset>,
    projector: Option<&'a ClassifiedAsset>,
}

#[derive(Debug)]
struct BaseModelResolution {
    compatible_vision_projector_types: Vec<String>,
    name: String,
    vision_capable: bool,
}

const EXPLICIT_PROJECTOR_NOT_INSTALLED: &str = "explicit projector asset was not installed";
const PROJECTOR_NOT_RUNNABLE_MODEL: &str = "projector assets are not runnable models";
const NO_MODEL_ASSETS_PROVIDED: &str = "no model assets were provided";

impl PairingResolver {
    pub fn resolve(files: &[ClassifiedAsset]) -> Result<PairingPlan, ModelError> {
        let selection = select_assets(files, None)?;
        let base = resolve_base_model(&selection.model_files)?;
        if let Some(projector) = selection.projector {
            validate_implicit_projector(&base, projector)?;
            return Ok(pairing_plan(&selection.model_files, Some(projector), base));
        }

        Ok(pairing_plan(&selection.model_files, None, base))
    }

    pub fn resolve_explicit(
        files: &[ClassifiedAsset],
        explicit_projector_asset_id: &str,
    ) -> Result<PairingPlan, ModelError> {
        let selection = select_assets(files, Some(explicit_projector_asset_id))?;
        let projector = selection
            .projector
            .ok_or_else(|| invalid_pairing(EXPLICIT_PROJECTOR_NOT_INSTALLED))?;
        let base = resolve_base_model(&selection.model_files)?;
        validate_projector_compatibility(&base, projector)?;
        Ok(pairing_plan(&selection.model_files, Some(projector), base))
    }
}

fn select_assets<'a>(
    files: &'a [ClassifiedAsset],
    explicit_projector_asset_id: Option<&str>,
) -> Result<AssetSelection<'a>, ModelError> {
    if files.is_empty() {
        return Err(invalid_source(NO_MODEL_ASSETS_PROVIDED));
    }

    let projectors: Vec<_> = files.iter().filter(|file| is_projector(file)).collect();
    if explicit_projector_asset_id.is_none() && projectors.len() > 1 {
        return Err(invalid_pairing(format!(
            "multiple projector assets were provided: {}",
            join_asset_names(&projectors)
        )));
    }

    let projector = if let Some(asset_id) = explicit_projector_asset_id {
        Some(select_explicit_projector(files, asset_id)?)
    } else {
        projectors.first().copied()
    };

    let projector_asset_id = projector.map(|asset| asset.asset_id.as_str());
    let mut model_files: Vec<_> = files
        .iter()
        .filter(|file| !is_selected_asset(file, projector_asset_id))
        .collect();
    model_files.sort_by(|left, right| left.name.cmp(&right.name));
    if model_files.is_empty() {
        return Err(invalid_pairing(PROJECTOR_NOT_RUNNABLE_MODEL));
    }

    Ok(AssetSelection {
        model_files,
        projector,
    })
}

fn select_explicit_projector<'a>(
    files: &'a [ClassifiedAsset],
    asset_id: &str,
) -> Result<&'a ClassifiedAsset, ModelError> {
    let projector = files
        .iter()
        .find(|file| file.asset_id == asset_id)
        .ok_or_else(|| invalid_pairing(EXPLICIT_PROJECTOR_NOT_INSTALLED))?;
    if !is_projector(projector) {
        return Err(invalid_pairing(format!(
            "\"{}\" is not a projector asset",
            projector.name
        )));
    }
    Ok(projector)
}

fn model_asset_ids(files: &[&ClassifiedAsset]) -> Vec<String> {
    files.iter().map(|file| file.asset_id.clone()).collect()
}

fn pairing_plan(
    model_files: &[&ClassifiedAsset],
    projector: Option<&ClassifiedAsset>,
    base: BaseModelResolution,
) -> PairingPlan {
    let has_projector = projector.is_some();
    PairingPlan {
        model_asset_ids: model_asset_ids(model_files),
        projector_asset_id: projector.map(|asset| asset.asset_id.clone()),
        name: base.name,
        modality: pairing_modality(has_projector, base.vision_capable),
        status: pairing_status(has_projector, base.vision_capable),
        compatible_vision_projector_types: base.compatible_vision_projector_types,
    }
}

fn pairing_modality(has_projector: bool, vision_capable: bool) -> ModelModality {
    if has_projector || vision_capable {
        ModelModality::Vision
    } else {
        ModelModality::Text
    }
}

fn pairing_status(has_projector: bool, vision_capable: bool) -> ModelStatus {
    if has_projector || !vision_capable {
        ModelStatus::Ready
    } else {
        ModelStatus::NeedsProjector
    }
}

fn join_asset_names(files: &[&ClassifiedAsset]) -> String {
    files
        .iter()
        .map(|file| file.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn resolve_base_model(files: &[&ClassifiedAsset]) -> Result<BaseModelResolution, ModelError> {
    let model_candidates: Vec<_> = files
        .iter()
        .copied()
        .filter(|file| !is_projector(file))
        .collect();
    if model_candidates.is_empty() {
        return Err(invalid_pairing(PROJECTOR_NOT_RUNNABLE_MODEL));
    }

    let vision_candidates: Vec<_> = model_candidates
        .iter()
        .copied()
        .filter(|file| file.inspection.vision_capable)
        .collect();
    let compatibility_sources: Vec<_> = vision_candidates
        .iter()
        .copied()
        .filter(|file| !file.inspection.compatible_vision_projector_types.is_empty())
        .collect();
    if !compatible_vision_types_agree(&compatibility_sources) {
        return Err(invalid_source(
            "model assets disagree on compatible vision projector types",
        ));
    }

    let base = vision_candidates
        .first()
        .copied()
        .unwrap_or(model_candidates[0]);
    Ok(BaseModelResolution {
        compatible_vision_projector_types: compatibility_sources
            .first()
            .map(|file| stable_type_list_vec(&file.inspection.compatible_vision_projector_types))
            .unwrap_or_default(),
        name: base.name.clone(),
        vision_capable: !vision_candidates.is_empty(),
    })
}

fn validate_implicit_projector(
    base: &BaseModelResolution,
    projector: &ClassifiedAsset,
) -> Result<(), ModelError> {
    if !base.vision_capable {
        return Err(invalid_pairing(
            "projector assets can only be paired with vision-capable models",
        ));
    }
    validate_projector_compatibility(base, projector)
}

fn validate_projector_compatibility(
    base: &BaseModelResolution,
    projector: &ClassifiedAsset,
) -> Result<(), ModelError> {
    let Some(provided_type) = projector
        .inspection
        .provided_vision_projector_type
        .as_deref()
    else {
        return Ok(());
    };
    if !base.compatible_vision_projector_types.is_empty()
        && !base
            .compatible_vision_projector_types
            .iter()
            .any(|expected| expected == provided_type)
    {
        return Err(invalid_pairing(format!(
            "projector type \"{}\" is not compatible with this model; expected one of: {}",
            provided_type,
            base.compatible_vision_projector_types.join(", ")
        )));
    }
    Ok(())
}

fn compatible_vision_types_agree(files: &[&ClassifiedAsset]) -> bool {
    if files.len() < 2 {
        return true;
    }
    let expected = stable_type_list_vec(&files[0].inspection.compatible_vision_projector_types);
    files.iter().skip(1).all(|file| {
        stable_type_list_vec(&file.inspection.compatible_vision_projector_types) == expected
    })
}

fn stable_type_list_vec(values: &[String]) -> Vec<String> {
    sorted_unique_strings(values.to_vec())
}

fn is_projector(file: &ClassifiedAsset) -> bool {
    file.inspection.role == AssetRole::Projector
}

fn is_selected_asset(file: &ClassifiedAsset, asset_id: Option<&str>) -> bool {
    asset_id == Some(file.asset_id.as_str())
}

#[cfg(test)]
#[path = "../../tests/lifecycle/pairing_tests.rs"]
mod pairing_tests;
