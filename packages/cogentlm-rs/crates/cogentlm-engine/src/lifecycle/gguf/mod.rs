//! GGUF metadata inspection used for model detection and pairing.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::types::{
    AssetInspection, AssetRole, GgufMetadataInspection, ModelDetection, ModelDetectionMethod,
    ModelError,
};

mod reader;

use reader::{
    advance_offset, read_string, read_u32, read_u64_as_usize, read_value, skip_value,
    GgufValueType, MetadataValue,
};

const GGUF_MAGIC: u32 = 0x4655_4747;
const SUPPORTED_GGUF_VERSIONS: &[u32] = &[2, 3];
const DEFAULT_MAX_PREFIX_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_MAX_PREFIX_BYTES_U64: u64 = 8 * 1024 * 1024;
const DEFAULT_INITIAL_READ_BYTES: usize = 64 * 1024;

const EARLY_STOP_KEYS: &[&str] = &[
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
    "tokenizer.huggingface.json",
];

const TARGET_KEYS: &[&str] = &[
    "general.type",
    "general.architecture",
    "clip.projector_type",
    "clip.vision.projector_type",
    "clip.has_vision_encoder",
];

pub fn inspect_gguf_metadata_path(
    path: impl AsRef<Path>,
) -> Result<Option<GgufMetadataInspection>, ModelError> {
    let mut file = File::open(path)?;
    let mut prefix = Vec::with_capacity(DEFAULT_INITIAL_READ_BYTES);
    file.by_ref()
        .take(DEFAULT_MAX_PREFIX_BYTES_U64)
        .read_to_end(&mut prefix)?;
    inspect_gguf_metadata(&prefix)
}

pub fn inspect_gguf_metadata(bytes: &[u8]) -> Result<Option<GgufMetadataInspection>, ModelError> {
    if bytes.len() < 24 {
        return Ok(None);
    }

    let magic = read_u32(bytes, 0)?;
    if magic != GGUF_MAGIC {
        return Ok(None);
    }

    let version = read_u32(bytes, 4)?;
    if !SUPPORTED_GGUF_VERSIONS.contains(&version) {
        return Err(ModelError::UnsupportedGgufVersion(version));
    }

    let kv_count = read_u64_as_usize(bytes, 16)?;
    let mut offset = 24usize;
    let mut scanned_key_count = 0usize;
    let mut stopped_early_at_key = None;
    let mut general_type = None;
    let mut general_architecture = None;
    let mut clip_projector_type = None;
    let mut clip_vision_projector_type = None;
    let mut clip_has_vision_encoder = None;

    for _ in 0..kv_count {
        let key = read_string(bytes, &mut offset)?;
        let value_type = GgufValueType::from_u32(read_u32(bytes, offset)?)?;
        advance_offset(&mut offset, 4)?;
        scanned_key_count += 1;

        if EARLY_STOP_KEYS.contains(&key.as_str())
            && has_useful_metadata(
                general_type.as_ref(),
                general_architecture.as_ref(),
                clip_projector_type.as_ref(),
                clip_vision_projector_type.as_ref(),
                clip_has_vision_encoder,
            )
        {
            stopped_early_at_key = Some(key);
            break;
        }

        if TARGET_KEYS.contains(&key.as_str()) {
            let value = read_value(bytes, &mut offset, value_type)?;
            match (key.as_str(), value) {
                ("general.type", MetadataValue::String(value)) => {
                    general_type = normalize_optional_string(&value);
                }
                ("general.architecture", MetadataValue::String(value)) => {
                    general_architecture = normalize_optional_string(&value);
                }
                ("clip.projector_type", MetadataValue::String(value)) => {
                    clip_projector_type = normalize_optional_string(&value);
                }
                ("clip.vision.projector_type", MetadataValue::String(value)) => {
                    clip_vision_projector_type = normalize_optional_string(&value);
                }
                ("clip.has_vision_encoder", MetadataValue::Bool(value)) => {
                    clip_has_vision_encoder = Some(value);
                }
                _ => {}
            }
        } else {
            skip_value(bytes, &mut offset, value_type)?;
        }
    }

    Ok(Some(GgufMetadataInspection {
        general_type,
        general_architecture,
        clip_projector_type,
        clip_vision_projector_type,
        clip_has_vision_encoder,
        scanned_key_count,
        stopped_early_at_key,
    }))
}

pub fn detect_model_from_gguf_bytes(
    name: impl Into<String>,
    bytes: &[u8],
) -> Result<ModelDetection, ModelError> {
    let name: String = name.into();
    let model_name = normalize_file_name(&name);
    let Some(metadata) = inspect_gguf_metadata(bytes)? else {
        return Ok(ModelDetection {
            inspection: AssetInspection::unknown(),
            detection_method: ModelDetectionMethod::None,
            model_name,
            model_type: None,
            model_architecture: None,
        });
    };

    let model_type = metadata.general_type;
    let model_architecture = metadata.general_architecture;
    let provided_vision_projector_type = metadata
        .clip_vision_projector_type
        .or(metadata.clip_projector_type);
    let clip_has_vision_encoder = metadata.clip_has_vision_encoder == Some(true);
    let inspection = build_inspection(
        model_type.as_deref(),
        model_architecture.as_deref(),
        clip_has_vision_encoder,
        provided_vision_projector_type,
    );
    let detection_method = if inspection.role == AssetRole::Unknown {
        ModelDetectionMethod::None
    } else {
        ModelDetectionMethod::GgufMetadata
    };

    Ok(ModelDetection {
        inspection,
        detection_method,
        model_name,
        model_type,
        model_architecture,
    })
}

fn build_inspection(
    model_type: Option<&str>,
    architecture: Option<&str>,
    clip_has_vision_encoder: bool,
    provided_vision_projector_type: Option<String>,
) -> AssetInspection {
    let is_projector = model_type == Some("mmproj")
        || architecture == Some("clip")
        || provided_vision_projector_type.is_some();
    let compatible_vision_projector_types = if is_projector {
        Vec::new()
    } else {
        resolve_compatible_vision_projector_types(architecture, clip_has_vision_encoder)
    };
    let vision_capable =
        !is_projector && (clip_has_vision_encoder || !compatible_vision_projector_types.is_empty());
    let role = if is_projector {
        AssetRole::Projector
    } else if model_type.is_some() || architecture.is_some() || clip_has_vision_encoder {
        AssetRole::Model
    } else {
        AssetRole::Unknown
    };

    AssetInspection {
        version: 1,
        role,
        architecture: architecture.map(str::to_string),
        vision_capable,
        compatible_vision_projector_types,
        provided_vision_projector_type,
    }
}

fn resolve_compatible_vision_projector_types(
    architecture: Option<&str>,
    clip_has_vision_encoder: bool,
) -> Vec<String> {
    let Some(architecture) = architecture else {
        return Vec::new();
    };
    let (types, requires_vision_encoder): (&[&str], bool) = match architecture {
        "cogvlm" => (&["cogvlm"], false),
        "gemma3" => (&["gemma3"], true),
        "gemma3n" => (&["gemma3nv"], true),
        "gemma4" => (&["gemma4v"], true),
        "hunyuan_vl" => (&["hunyuanvl"], false),
        "lfm2" => (&["lfm2"], true),
        "llama4" => (&["llama4"], true),
        "minicpm" | "minicpm3" => (&["resampler", "minicpmv4_6"], true),
        "paddleocr" => (&["paddleocr"], false),
        "qwen2vl" => (&["qwen2vl_merger", "qwen2.5vl_merger"], false),
        "qwen3vl" | "qwen3vlmoe" => (&["qwen3vl_merger"], false),
        _ => (&[], false),
    };
    if requires_vision_encoder && !clip_has_vision_encoder {
        return Vec::new();
    }
    let mut compatible_types = Vec::with_capacity(types.len());
    compatible_types.extend(types.iter().map(|value| (*value).to_string()));
    compatible_types
}

fn has_useful_metadata(
    general_type: Option<&String>,
    general_architecture: Option<&String>,
    clip_projector_type: Option<&String>,
    clip_vision_projector_type: Option<&String>,
    clip_has_vision_encoder: Option<bool>,
) -> bool {
    general_type.is_some()
        || general_architecture.is_some()
        || clip_projector_type.is_some()
        || clip_vision_projector_type.is_some()
        || clip_has_vision_encoder.is_some()
}

fn normalize_file_name(file_name: &str) -> String {
    let trimmed = file_name.trim();
    if trimmed.is_empty() {
        "model.gguf".to_string()
    } else {
        trimmed.to_string()
    }
}

fn normalize_optional_string(value: &str) -> Option<String> {
    let normalized = value.trim().to_lowercase();
    (!normalized.is_empty()).then_some(normalized)
}

#[cfg(test)]
mod tests {
    mod gguf_tests;
}
