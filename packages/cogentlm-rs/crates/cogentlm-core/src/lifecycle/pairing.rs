use super::types::{
    AssetRole, ClassifiedAsset, ModelError, ModelModality, ModelStatus, PairingPlan,
};

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

impl PairingResolver {
    pub fn resolve(files: &[ClassifiedAsset]) -> Result<PairingPlan, ModelError> {
        let selection = select_assets(files, None)?;
        let base = resolve_base_model(&selection.model_files)?;
        Ok(PairingPlan {
            model_asset_ids: selection
                .model_files
                .iter()
                .map(|file| file.asset_id.clone())
                .collect(),
            projector_asset_id: None,
            name: base.name,
            modality: if base.vision_capable {
                ModelModality::Vision
            } else {
                ModelModality::Text
            },
            status: if base.vision_capable {
                ModelStatus::NeedsProjector
            } else {
                ModelStatus::Ready
            },
            compatible_vision_projector_types: base.compatible_vision_projector_types,
        })
    }

    pub fn resolve_explicit(
        files: &[ClassifiedAsset],
        explicit_projector_asset_id: &str,
    ) -> Result<PairingPlan, ModelError> {
        let selection = select_assets(files, Some(explicit_projector_asset_id))?;
        let projector = selection.projector.ok_or_else(|| {
            ModelError::InvalidModelPairing(
                "explicit projector asset was not installed".to_string(),
            )
        })?;
        let base = resolve_base_model(&selection.model_files)?;
        validate_explicit_projector(&base, projector)?;
        Ok(PairingPlan {
            model_asset_ids: selection
                .model_files
                .iter()
                .map(|file| file.asset_id.clone())
                .collect(),
            projector_asset_id: Some(projector.asset_id.clone()),
            name: base.name,
            modality: ModelModality::Vision,
            status: ModelStatus::Ready,
            compatible_vision_projector_types: base.compatible_vision_projector_types,
        })
    }
}

fn select_assets<'a>(
    files: &'a [ClassifiedAsset],
    explicit_projector_asset_id: Option<&str>,
) -> Result<AssetSelection<'a>, ModelError> {
    if files.is_empty() {
        return Err(ModelError::InvalidModelSource(
            "no model assets were provided".to_string(),
        ));
    }

    let projectors: Vec<_> = files
        .iter()
        .filter(|file| file.inspection.role == AssetRole::Projector)
        .collect();
    if explicit_projector_asset_id.is_none() && projectors.len() > 1 {
        return Err(ModelError::InvalidModelPairing(format!(
            "multiple projector assets were provided: {}",
            projectors
                .iter()
                .map(|file| file.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    let projector = if let Some(asset_id) = explicit_projector_asset_id {
        let projector = files
            .iter()
            .find(|file| file.asset_id == asset_id)
            .ok_or_else(|| {
                ModelError::InvalidModelPairing(
                    "explicit projector asset was not installed".to_string(),
                )
            })?;
        if projector.inspection.role != AssetRole::Projector {
            return Err(ModelError::InvalidModelPairing(format!(
                "\"{}\" is not a projector asset",
                projector.name
            )));
        }
        Some(projector)
    } else {
        projectors.first().copied()
    };

    let mut model_files: Vec<_> = files
        .iter()
        .filter(|file| Some(file.asset_id.as_str()) != projector.map(|item| item.asset_id.as_str()))
        .collect();
    model_files.sort_by(|left, right| left.name.cmp(&right.name));
    if model_files.is_empty() {
        return Err(ModelError::InvalidModelPairing(
            "projector assets are not runnable models".to_string(),
        ));
    }

    Ok(AssetSelection {
        model_files,
        projector,
    })
}

fn resolve_base_model(files: &[&ClassifiedAsset]) -> Result<BaseModelResolution, ModelError> {
    let model_candidates: Vec<_> = files
        .iter()
        .copied()
        .filter(|file| file.inspection.role != AssetRole::Projector)
        .collect();
    if model_candidates.is_empty() {
        return Err(ModelError::InvalidModelPairing(
            "projector assets are not runnable models".to_string(),
        ));
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
        return Err(ModelError::InvalidModelSource(
            "model assets disagree on compatible vision projector types".to_string(),
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

fn validate_explicit_projector(
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
        return Err(ModelError::InvalidModelPairing(format!(
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
    let expected = stable_type_list(&files[0].inspection.compatible_vision_projector_types);
    files.iter().skip(1).all(|file| {
        stable_type_list(&file.inspection.compatible_vision_projector_types) == expected
    })
}

fn stable_type_list(values: &[String]) -> String {
    stable_type_list_vec(values).join("\0")
}

fn stable_type_list_vec(values: &[String]) -> Vec<String> {
    let mut values = values.to_vec();
    values.sort();
    values.dedup();
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::AssetInspection;

    fn model(id: &str, name: &str, vision_types: &[&str]) -> ClassifiedAsset {
        ClassifiedAsset {
            asset_id: id.to_string(),
            name: name.to_string(),
            inspection: AssetInspection {
                version: 1,
                role: AssetRole::Model,
                architecture: Some("test".to_string()),
                vision_capable: !vision_types.is_empty(),
                compatible_vision_projector_types: vision_types
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
                provided_vision_projector_type: None,
            },
        }
    }

    fn projector(id: &str, name: &str, projector_type: Option<&str>) -> ClassifiedAsset {
        ClassifiedAsset {
            asset_id: id.to_string(),
            name: name.to_string(),
            inspection: AssetInspection {
                version: 1,
                role: AssetRole::Projector,
                architecture: Some("clip".to_string()),
                vision_capable: false,
                compatible_vision_projector_types: Vec::new(),
                provided_vision_projector_type: projector_type.map(str::to_string),
            },
        }
    }

    #[test]
    fn resolves_text_model_as_ready() {
        let plan =
            PairingResolver::resolve(&[model("asset-model", "base.gguf", &[])]).expect("plan");

        assert_eq!(plan.modality, ModelModality::Text);
        assert_eq!(plan.status, ModelStatus::Ready);
        assert_eq!(plan.projector_asset_id, None);
    }

    #[test]
    fn resolves_vision_base_as_needing_projector() {
        let plan = PairingResolver::resolve(&[model("asset-model", "base.gguf", &["lfm2"])])
            .expect("plan");

        assert_eq!(plan.modality, ModelModality::Vision);
        assert_eq!(plan.status, ModelStatus::NeedsProjector);
        assert_eq!(plan.compatible_vision_projector_types, vec!["lfm2"]);
    }

    #[test]
    fn accepts_explicit_compatible_projector() {
        let base = model("asset-model", "base.gguf", &["lfm2"]);
        let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

        let plan =
            PairingResolver::resolve_explicit(&[base, mmproj], "asset-projector").expect("plan");

        assert_eq!(plan.modality, ModelModality::Vision);
        assert_eq!(plan.status, ModelStatus::Ready);
        assert_eq!(plan.projector_asset_id, Some("asset-projector".to_string()));
    }

    #[test]
    fn rejects_explicit_incompatible_projector() {
        let base = model("asset-model", "base.gguf", &["lfm2"]);
        let mmproj = projector("asset-projector", "bad-mmproj.gguf", Some("other"));

        let error = PairingResolver::resolve_explicit(&[base, mmproj], "asset-projector")
            .expect_err("pairing error");

        assert!(matches!(error, ModelError::InvalidModelPairing(_)));
    }

    #[test]
    fn rejects_multiple_implicit_projectors() {
        let base = model("asset-model", "base.gguf", &["lfm2"]);
        let first = projector("asset-projector-a", "a.gguf", Some("lfm2"));
        let second = projector("asset-projector-b", "b.gguf", Some("lfm2"));

        let error = PairingResolver::resolve(&[base, first, second]).expect_err("pairing error");

        assert!(matches!(error, ModelError::InvalidModelPairing(_)));
    }

    #[test]
    fn rejects_shards_with_conflicting_projector_types() {
        let first = model("asset-a", "a.gguf", &["lfm2"]);
        let second = model("asset-b", "b.gguf", &["qwen3vl_merger"]);

        let error = PairingResolver::resolve(&[first, second]).expect_err("source error");

        assert!(matches!(error, ModelError::InvalidModelSource(_)));
    }
}
