//! Pairing of model assets (weights + projector) into a runnable ModelEntry.

use super::types::{
    AssetInspection, AssetRole, ClassifiedAsset, ModelError, ModelModality, ModelStatus,
    PairingPlan,
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
    compatible_audio_projector_types: Vec<String>,
    compatible_audio_generation_projector_types: Vec<String>,
    name: String,
    vision_capable: bool,
    audio_capable: bool,
}

const PROJECTOR_NOT_RUNNABLE_MODEL: &str = "projector assets are not runnable models";
const NO_MODEL_ASSETS_PROVIDED: &str = "no model assets were provided";

impl PairingResolver {
    pub fn resolve(files: &[ClassifiedAsset]) -> Result<PairingPlan, ModelError> {
        let selection = select_assets(files)?;
        let base = resolve_base_model(&selection.model_files)?;
        if let Some(projector) = selection.projector {
            validate_projector_compatibility(&base, projector)?;
            return Ok(pairing_plan(&selection.model_files, Some(projector), base));
        }

        Ok(pairing_plan(&selection.model_files, None, base))
    }
}

fn select_assets(files: &[ClassifiedAsset]) -> Result<AssetSelection<'_>, ModelError> {
    if files.is_empty() {
        return Err(invalid_source(NO_MODEL_ASSETS_PROVIDED));
    }

    let projectors: Vec<_> = files.iter().filter(|file| is_projector(file)).collect();
    if projectors.len() > 1 {
        return Err(invalid_pairing(format!(
            "multiple projector assets were provided: {}",
            join_asset_names(&projectors)
        )));
    }

    let projector = projectors.first().copied();
    let mut model_files: Vec<_> = files.iter().filter(|file| !is_projector(file)).collect();
    model_files.sort_by(|left, right| left.name.cmp(&right.name));
    if model_files.is_empty() {
        return Err(invalid_pairing(PROJECTOR_NOT_RUNNABLE_MODEL));
    }

    Ok(AssetSelection {
        model_files,
        projector,
    })
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
    let vision_capable = base.vision_capable
        || projector.is_some_and(|asset| {
            asset.inspection.vision_capable
                || asset.inspection.provided_vision_projector_type.is_some()
        });
    let audio_capable = base.audio_capable
        || projector.is_some_and(|asset| {
            asset.inspection.audio_capable
                || asset.inspection.provided_audio_projector_type.is_some()
                || asset.inspection.audio_generation_capable
                || asset
                    .inspection
                    .provided_audio_generation_projector_type
                    .is_some()
        });
    PairingPlan {
        model_asset_ids: model_asset_ids(model_files),
        projector_asset_id: projector.map(|asset| asset.asset_id.clone()),
        name: base.name,
        modality: pairing_modality(vision_capable, audio_capable),
        status: pairing_status(has_projector, base.vision_capable || base.audio_capable),
        compatible_vision_projector_types: base.compatible_vision_projector_types,
        compatible_audio_projector_types: base.compatible_audio_projector_types,
        compatible_audio_generation_projector_types: base
            .compatible_audio_generation_projector_types,
    }
}

fn pairing_modality(vision_capable: bool, audio_capable: bool) -> ModelModality {
    match (vision_capable, audio_capable) {
        (false, false) => ModelModality::Text,
        (true, false) => ModelModality::Vision,
        (false, true) => ModelModality::Audio,
        (true, true) => ModelModality::Multimodal,
    }
}

fn pairing_status(has_projector: bool, media_capable: bool) -> ModelStatus {
    if has_projector || !media_capable {
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
    let media_candidates: Vec<_> = model_candidates
        .iter()
        .copied()
        .filter(|file| {
            file.inspection.vision_capable
                || file.inspection.audio_capable
                || file.inspection.audio_generation_capable
        })
        .collect();
    let compatibility_sources: Vec<_> = vision_candidates
        .iter()
        .copied()
        .filter(|file| !file.inspection.compatible_vision_projector_types.is_empty())
        .collect();
    let audio_compatibility_sources: Vec<_> = model_candidates
        .iter()
        .copied()
        .filter(|file| !file.inspection.compatible_audio_projector_types.is_empty())
        .collect();
    let audio_generation_compatibility_sources: Vec<_> = model_candidates
        .iter()
        .copied()
        .filter(|file| {
            !file
                .inspection
                .compatible_audio_generation_projector_types
                .is_empty()
        })
        .collect();
    if !compatible_types_agree(&compatibility_sources, |inspection| {
        &inspection.compatible_vision_projector_types
    }) {
        return Err(invalid_source(
            "model assets disagree on compatible vision projector types",
        ));
    }
    if !compatible_types_agree(&audio_compatibility_sources, |inspection| {
        &inspection.compatible_audio_projector_types
    }) {
        return Err(invalid_source(
            "model assets disagree on compatible audio projector types",
        ));
    }
    if !compatible_types_agree(&audio_generation_compatibility_sources, |inspection| {
        &inspection.compatible_audio_generation_projector_types
    }) {
        return Err(invalid_source(
            "model assets disagree on compatible audio-generation projector types",
        ));
    }

    let base = media_candidates
        .first()
        .copied()
        .unwrap_or(model_candidates[0]);
    Ok(BaseModelResolution {
        compatible_vision_projector_types: compatibility_sources
            .first()
            .map(|file| stable_type_list_vec(&file.inspection.compatible_vision_projector_types))
            .unwrap_or_default(),
        compatible_audio_projector_types: audio_compatibility_sources
            .first()
            .map(|file| stable_type_list_vec(&file.inspection.compatible_audio_projector_types))
            .unwrap_or_default(),
        compatible_audio_generation_projector_types: audio_generation_compatibility_sources
            .first()
            .map(|file| {
                stable_type_list_vec(&file.inspection.compatible_audio_generation_projector_types)
            })
            .unwrap_or_default(),
        name: base.name.clone(),
        vision_capable: !vision_candidates.is_empty(),
        audio_capable: model_candidates
            .iter()
            .any(|file| file.inspection.audio_capable || file.inspection.audio_generation_capable),
    })
}

fn validate_projector_compatibility(
    base: &BaseModelResolution,
    projector: &ClassifiedAsset,
) -> Result<(), ModelError> {
    validate_projector_type(
        "vision",
        &base.compatible_vision_projector_types,
        projector
            .inspection
            .provided_vision_projector_type
            .as_deref(),
    )?;
    validate_projector_type(
        "audio",
        &base.compatible_audio_projector_types,
        projector
            .inspection
            .provided_audio_projector_type
            .as_deref(),
    )?;
    validate_projector_type(
        "audio-generation",
        &base.compatible_audio_generation_projector_types,
        projector
            .inspection
            .provided_audio_generation_projector_type
            .as_deref(),
    )
}

fn validate_projector_type(
    modality: &str,
    compatible_types: &[String],
    provided_type: Option<&str>,
) -> Result<(), ModelError> {
    if compatible_types.is_empty() {
        return Ok(());
    }
    let Some(provided_type) = provided_type else {
        return Err(invalid_pairing(format!(
            "projector does not declare its {modality} projector type; expected one of: {}",
            compatible_types.join(", ")
        )));
    };
    if compatible_types
        .iter()
        .any(|expected| expected == provided_type)
    {
        return Ok(());
    }
    Err(invalid_pairing(format!(
        concat!(
            "{} projector type \"{}\" is not compatible with this model; ",
            "expected one of: {}"
        ),
        modality,
        provided_type,
        compatible_types.join(", ")
    )))
}

fn compatible_types_agree<F>(files: &[&ClassifiedAsset], field: F) -> bool
where
    F: for<'a> Fn(&'a AssetInspection) -> &'a [String],
{
    if files.len() < 2 {
        return true;
    }
    let expected = stable_type_list_vec(field(&files[0].inspection));
    files
        .iter()
        .skip(1)
        .all(|file| stable_type_list_vec(field(&file.inspection)) == expected)
}

fn stable_type_list_vec(values: &[String]) -> Vec<String> {
    sorted_unique_strings(values.to_vec())
}

fn is_projector(file: &ClassifiedAsset) -> bool {
    file.inspection.role == AssetRole::Projector
}

#[cfg(test)]
#[path = "../../tests/lifecycle/pairing_tests.rs"]
mod pairing_tests;
