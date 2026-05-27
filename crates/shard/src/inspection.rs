//! GGUF metadata inspection used for model detection and pairing.

use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::bytes::{u64_from_usize, usize_from_u64, CountingReader};
use crate::{GgufError, GgufValueType, BYTES_PER_MIB_USIZE, GGUF_MAGIC, SUPPORTED_GGUF_VERSIONS};

const DEFAULT_MAX_PREFIX_BYTES: usize = 8 * BYTES_PER_MIB_USIZE;
const DEFAULT_MAX_PREFIX_BYTES_U64: u64 = DEFAULT_MAX_PREFIX_BYTES as u64;
const DEFAULT_INITIAL_READ_BYTES: usize = BYTES_PER_MIB_USIZE / 16;

const EARLY_STOP_KEYS: &[&str] = &[
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
    "tokenizer.huggingface.json",
];

const TARGET_KEYS: &[&str] = &[
    "general.type",
    "general.architecture",
    "general.pooling_type",
    "clip.projector_type",
    "clip.vision.projector_type",
    "clip.has_vision_encoder",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetRole {
    Model,
    Projector,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetInspection {
    pub version: u32,
    pub role: AssetRole,
    pub architecture: Option<String>,
    pub vision_capable: bool,
    pub compatible_vision_projector_types: Vec<String>,
    pub provided_vision_projector_type: Option<String>,
}

impl AssetInspection {
    pub fn unknown() -> Self {
        Self {
            version: 1,
            role: AssetRole::Unknown,
            architecture: None,
            vision_capable: false,
            compatible_vision_projector_types: Vec::new(),
            provided_vision_projector_type: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufMetadataInspection {
    pub general_type: Option<String>,
    pub general_architecture: Option<String>,
    pub pooling_type: Option<u32>,
    pub clip_projector_type: Option<String>,
    pub clip_vision_projector_type: Option<String>,
    pub clip_has_vision_encoder: Option<bool>,
    pub scanned_key_count: usize,
    pub stopped_early_at_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDetection {
    pub inspection: AssetInspection,
    pub detection_method: ModelDetectionMethod,
    pub model_name: String,
    pub model_type: Option<String>,
    pub model_architecture: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDetectionMethod {
    GgufMetadata,
    None,
}

#[derive(Debug, Clone, PartialEq)]
enum MetadataValue {
    String(String),
    Bool(bool),
    U32(u32),
    Skipped,
}

pub fn inspect_gguf_metadata_path(
    path: impl AsRef<Path>,
) -> Result<Option<GgufMetadataInspection>, GgufError> {
    let mut file = File::open(path)?;
    let mut prefix = Vec::with_capacity(DEFAULT_INITIAL_READ_BYTES);
    file.by_ref()
        .take(DEFAULT_MAX_PREFIX_BYTES_U64)
        .read_to_end(&mut prefix)?;
    inspect_gguf_metadata(&prefix)
}

pub fn inspect_gguf_metadata(bytes: &[u8]) -> Result<Option<GgufMetadataInspection>, GgufError> {
    if bytes.len() < 24 {
        return Ok(None);
    }

    let mut cursor = Cursor::new(bytes);
    let mut reader = CountingReader::new(&mut cursor);

    let magic = read_metadata_u32(&mut reader, bytes.len())?;
    if magic != GGUF_MAGIC {
        return Ok(None);
    }

    let version = read_metadata_u32(&mut reader, bytes.len())?;
    if !SUPPORTED_GGUF_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let _tensor_count = read_metadata_u64(&mut reader, bytes.len())?;
    let kv_count = usize_from_u64(read_metadata_u64(&mut reader, bytes.len())?, "kv count")?;
    let mut scanned_key_count = 0usize;
    let mut stopped_early_at_key = None;
    let mut general_type = None;
    let mut general_architecture = None;
    let mut pooling_type = None;
    let mut clip_projector_type = None;
    let mut clip_vision_projector_type = None;
    let mut clip_has_vision_encoder = None;

    for _ in 0..kv_count {
        let key = read_metadata_string(&mut reader, bytes.len())?;
        let value_type = GgufValueType::from_u32(read_metadata_u32(&mut reader, bytes.len())?)?;
        scanned_key_count += 1;

        if EARLY_STOP_KEYS.contains(&key.as_str())
            && has_useful_metadata(
                general_type.as_ref(),
                general_architecture.as_ref(),
                pooling_type,
                clip_projector_type.as_ref(),
                clip_vision_projector_type.as_ref(),
                clip_has_vision_encoder,
            )
        {
            stopped_early_at_key = Some(key);
            break;
        }

        if is_target_key(&key) {
            let value = read_metadata_value(&mut reader, value_type, bytes.len())?;
            match (key.as_str(), value) {
                ("general.type", MetadataValue::String(value)) => {
                    general_type = normalize_optional_string(&value);
                }
                ("general.architecture", MetadataValue::String(value)) => {
                    general_architecture = normalize_optional_string(&value);
                }
                (key, MetadataValue::U32(value)) if is_pooling_key(key) => {
                    pooling_type = Some(value);
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
            skip_metadata_value(&mut reader, value_type, bytes.len())?;
        }
    }

    Ok(Some(GgufMetadataInspection {
        general_type,
        general_architecture,
        pooling_type,
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
) -> Result<ModelDetection, GgufError> {
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

fn read_metadata_value<R: Read>(
    reader: &mut CountingReader<'_, R>,
    value_type: GgufValueType,
    prefix_len: usize,
) -> Result<MetadataValue, GgufError> {
    match value_type {
        GgufValueType::String => Ok(MetadataValue::String(read_metadata_string(
            reader, prefix_len,
        )?)),
        GgufValueType::Bool => Ok(MetadataValue::Bool(
            read_metadata_u8(reader, prefix_len)? != 0,
        )),
        GgufValueType::Uint32 => Ok(MetadataValue::U32(read_metadata_u32(reader, prefix_len)?)),
        GgufValueType::Array => {
            skip_array(reader, prefix_len)?;
            Ok(MetadataValue::Skipped)
        }
        _ => {
            skip_metadata_value(reader, value_type, prefix_len)?;
            Ok(MetadataValue::Skipped)
        }
    }
}

fn skip_metadata_value<R: Read>(
    reader: &mut CountingReader<'_, R>,
    value_type: GgufValueType,
    prefix_len: usize,
) -> Result<(), GgufError> {
    match value_type {
        GgufValueType::String => skip_metadata_string(reader, prefix_len),
        GgufValueType::Array => skip_array(reader, prefix_len),
        _ => {
            let size = value_type
                .scalar_size()
                .ok_or_else(|| GgufError::Invalid("unsupported scalar type".to_string()))?;
            skip_metadata_bytes(reader, size, prefix_len)
        }
    }
}

fn skip_array<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<(), GgufError> {
    let item_type = GgufValueType::from_u32(read_metadata_u32(reader, prefix_len)?)?;
    let len = usize_from_u64(read_metadata_u64(reader, prefix_len)?, "array length")?;
    if item_type == GgufValueType::String {
        for _ in 0..len {
            skip_metadata_string(reader, prefix_len)?;
        }
        return Ok(());
    }
    let Some(item_size) = item_type.scalar_size() else {
        return Err(GgufError::Invalid(
            "nested GGUF arrays are not supported".to_string(),
        ));
    };
    let byte_len = len
        .checked_mul(item_size)
        .ok_or_else(|| GgufError::Invalid("array length overflow".to_string()))?;
    skip_metadata_bytes(reader, byte_len, prefix_len)
}

fn read_metadata_string<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<String, GgufError> {
    let len = usize_from_u64(read_metadata_u64(reader, prefix_len)?, "string length")?;
    require_available(reader, len, prefix_len)?;
    let bytes = reader
        .read_vec(len)
        .map_err(|error| map_metadata_error(error, prefix_len))?;
    String::from_utf8(bytes).map_err(|_| GgufError::Invalid("string is not UTF-8".to_string()))
}

fn skip_metadata_string<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<(), GgufError> {
    let len = usize_from_u64(read_metadata_u64(reader, prefix_len)?, "string length")?;
    skip_metadata_bytes(reader, len, prefix_len)
}

fn read_metadata_u8<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<u8, GgufError> {
    reader
        .read_u8()
        .map_err(|error| map_metadata_error(error, prefix_len))
}

fn read_metadata_u32<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<u32, GgufError> {
    reader
        .read_u32()
        .map_err(|error| map_metadata_error(error, prefix_len))
}

fn read_metadata_u64<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<u64, GgufError> {
    reader
        .read_u64()
        .map_err(|error| map_metadata_error(error, prefix_len))
}

fn skip_metadata_bytes<R: Read>(
    reader: &mut CountingReader<'_, R>,
    len: usize,
    prefix_len: usize,
) -> Result<(), GgufError> {
    require_available(reader, len, prefix_len)?;
    reader
        .skip_bytes(len)
        .map_err(|error| map_metadata_error(error, prefix_len))
}

fn require_available<R: Read>(
    reader: &CountingReader<'_, R>,
    len: usize,
    prefix_len: usize,
) -> Result<(), GgufError> {
    let end = reader
        .position()
        .checked_add(u64_from_usize(len, "metadata length")?)
        .ok_or_else(|| GgufError::Invalid("metadata offset overflow".to_string()))?;
    if end <= u64_from_usize(prefix_len, "metadata prefix length")? {
        return Ok(());
    }
    Err(metadata_prefix_error(prefix_len))
}

fn map_metadata_error(error: GgufError, prefix_len: usize) -> GgufError {
    match error {
        GgufError::Io(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
            metadata_prefix_error(prefix_len)
        }
        other => other,
    }
}

fn metadata_prefix_error(prefix_len: usize) -> GgufError {
    if prefix_len >= DEFAULT_MAX_PREFIX_BYTES {
        GgufError::MetadataTooLarge {
            max_bytes: DEFAULT_MAX_PREFIX_BYTES,
        }
    } else {
        GgufError::Invalid("metadata is truncated".to_string())
    }
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
    pooling_type: Option<u32>,
    clip_projector_type: Option<&String>,
    clip_vision_projector_type: Option<&String>,
    clip_has_vision_encoder: Option<bool>,
) -> bool {
    general_type.is_some()
        || general_architecture.is_some()
        || pooling_type.is_some()
        || clip_projector_type.is_some()
        || clip_vision_projector_type.is_some()
        || clip_has_vision_encoder.is_some()
}

fn is_target_key(key: &str) -> bool {
    TARGET_KEYS.contains(&key) || is_pooling_key(key)
}

fn is_pooling_key(key: &str) -> bool {
    key == "general.pooling_type" || key.ends_with(".pooling_type")
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
    mod inspection_tests;
}
