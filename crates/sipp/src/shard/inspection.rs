//! GGUF metadata inspection used for model detection and pairing.

use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::shard::bytes::{u64_from_usize, usize_from_u64, CountingReader};
use crate::shard::{
    GgufError, GgufValueType, BYTES_PER_MIB_USIZE, GGUF_MAGIC, SUPPORTED_GGUF_VERSIONS,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../tests/shard/inspection_tests.rs"]
mod inspection_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
const DEFAULT_MAX_PREFIX_BYTES: usize = 8 * BYTES_PER_MIB_USIZE;
const DEFAULT_MAX_PREFIX_BYTES_U64: u64 = DEFAULT_MAX_PREFIX_BYTES as u64;
const DEFAULT_INITIAL_READ_BYTES: usize = BYTES_PER_MIB_USIZE / 16;
const QWEN3TTS_ARCHITECTURE: &str = "qwen3tts";
const QWEN3TTS_AUDIO_PROJECTOR: &str = "qwen3tts_spkenc";
const QWEN3TTS_AUDIO_GENERATION_PROJECTOR: &str = "qwen3tts_gen";

const EARLY_STOP_KEYS: &[&str] = &[
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
    "tokenizer.huggingface.json",
];

const TARGET_KEYS: &[&str] = &[
    "general.type",
    "general.architecture",
    "general.tags",
    "general.pooling_type",
    "clip.projector_type",
    "clip.vision.projector_type",
    "clip.audio.projector_type",
    "clip.gen.audio.projector_type",
    "clip.has_vision_encoder",
    "clip.has_audio_encoder",
    "clip.has_gen_audio_encoder",
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
    /// Context capacity declared by the model's GGUF metadata.
    pub trained_context_size: Option<u32>,
    pub vision_capable: bool,
    /// Whether the asset accepts audio input or requires an audio projector.
    pub audio_capable: bool,
    /// Whether the asset generates audio or requires an audio-generation projector.
    pub audio_generation_capable: bool,
    pub compatible_vision_projector_types: Vec<String>,
    /// Audio-input projector types compatible with this model asset.
    pub compatible_audio_projector_types: Vec<String>,
    /// Audio-generation projector types compatible with this model asset.
    pub compatible_audio_generation_projector_types: Vec<String>,
    pub provided_vision_projector_type: Option<String>,
    /// Audio-input projector type provided by this projector asset.
    pub provided_audio_projector_type: Option<String>,
    /// Audio-generation projector type provided by this projector asset.
    pub provided_audio_generation_projector_type: Option<String>,
}

impl AssetInspection {
    /// Current persisted asset-inspection schema version.
    pub const VERSION: u32 = 4;

    pub fn unknown() -> Self {
        Self {
            version: Self::VERSION,
            role: AssetRole::Unknown,
            architecture: None,
            trained_context_size: None,
            vision_capable: false,
            audio_capable: false,
            audio_generation_capable: false,
            compatible_vision_projector_types: Vec::new(),
            compatible_audio_projector_types: Vec::new(),
            compatible_audio_generation_projector_types: Vec::new(),
            provided_vision_projector_type: None,
            provided_audio_projector_type: None,
            provided_audio_generation_projector_type: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufMetadataInspection {
    pub general_type: Option<String>,
    pub general_architecture: Option<String>,
    /// Value stored in `<architecture>.context_length`.
    pub trained_context_size: Option<u32>,
    /// Values stored in `general.tags`.
    pub general_tags: Vec<String>,
    pub pooling_type: Option<u32>,
    pub clip_projector_type: Option<String>,
    pub clip_vision_projector_type: Option<String>,
    /// Value stored in `clip.audio.projector_type`.
    pub clip_audio_projector_type: Option<String>,
    /// Value stored in `clip.gen.audio.projector_type`.
    pub clip_audio_generation_projector_type: Option<String>,
    pub clip_has_vision_encoder: Option<bool>,
    /// Value stored in `clip.has_audio_encoder`.
    pub clip_has_audio_encoder: Option<bool>,
    /// Value stored in `clip.has_gen_audio_encoder`.
    pub clip_has_audio_generation_encoder: Option<bool>,
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
    Strings(Vec<String>),
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
    let mut metadata = GgufMetadataInspection::default();
    let mut context_lengths = Vec::new();

    for _ in 0..kv_count {
        let key = read_metadata_string(&mut reader, bytes.len())?;
        let value_type = GgufValueType::from_u32(read_metadata_u32(&mut reader, bytes.len())?)?;
        metadata.scanned_key_count += 1;

        if EARLY_STOP_KEYS.contains(&key.as_str()) && has_useful_metadata(&metadata) {
            metadata.stopped_early_at_key = Some(key);
            break;
        }

        if let Some(architecture) = context_length_architecture(&key) {
            let value = read_metadata_value(&mut reader, value_type, bytes.len())?;
            if let MetadataValue::U32(value) = value {
                context_lengths.push((architecture.to_string(), value));
            }
        } else if is_target_key(&key) {
            let value = read_metadata_value(&mut reader, value_type, bytes.len())?;
            match (key.as_str(), value) {
                ("general.type", MetadataValue::String(value)) => {
                    metadata.general_type = normalize_optional_string(&value);
                }
                ("general.architecture", MetadataValue::String(value)) => {
                    metadata.general_architecture = normalize_optional_string(&value);
                }
                ("general.tags", MetadataValue::Strings(values)) => {
                    metadata.general_tags = normalize_strings(values);
                }
                (key, MetadataValue::U32(value)) if is_pooling_key(key) => {
                    metadata.pooling_type = Some(value);
                }
                ("clip.projector_type", MetadataValue::String(value)) => {
                    metadata.clip_projector_type = normalize_optional_string(&value);
                }
                ("clip.vision.projector_type", MetadataValue::String(value)) => {
                    metadata.clip_vision_projector_type = normalize_optional_string(&value);
                }
                ("clip.audio.projector_type", MetadataValue::String(value)) => {
                    metadata.clip_audio_projector_type = normalize_optional_string(&value);
                }
                ("clip.gen.audio.projector_type", MetadataValue::String(value)) => {
                    metadata.clip_audio_generation_projector_type =
                        normalize_optional_string(&value);
                }
                ("clip.has_vision_encoder", MetadataValue::Bool(value)) => {
                    metadata.clip_has_vision_encoder = Some(value);
                }
                ("clip.has_audio_encoder", MetadataValue::Bool(value)) => {
                    metadata.clip_has_audio_encoder = Some(value);
                }
                ("clip.has_gen_audio_encoder", MetadataValue::Bool(value)) => {
                    metadata.clip_has_audio_generation_encoder = Some(value);
                }
                _ => {}
            }
        } else {
            skip_metadata_value(&mut reader, value_type, bytes.len())?;
        }
    }

    metadata.trained_context_size = metadata
        .general_architecture
        .as_deref()
        .and_then(|expected| {
            context_lengths
                .into_iter()
                .find_map(|(architecture, size)| (architecture == expected).then_some(size))
        });

    Ok(Some(metadata))
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

    let inspection = build_inspection(&metadata);
    let model_type = metadata.general_type;
    let model_architecture = metadata.general_architecture;
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
        GgufValueType::Array => read_metadata_array(reader, prefix_len),
        _ => {
            skip_metadata_value(reader, value_type, prefix_len)?;
            Ok(MetadataValue::Skipped)
        }
    }
}

fn read_metadata_array<R: Read>(
    reader: &mut CountingReader<'_, R>,
    prefix_len: usize,
) -> Result<MetadataValue, GgufError> {
    let item_type = GgufValueType::from_u32(read_metadata_u32(reader, prefix_len)?)?;
    let len = usize_from_u64(read_metadata_u64(reader, prefix_len)?, "array length")?;
    if item_type == GgufValueType::String {
        let mut values = Vec::new();
        for _ in 0..len {
            values.push(read_metadata_string(reader, prefix_len)?);
        }
        return Ok(MetadataValue::Strings(values));
    }
    let Some(item_size) = item_type.scalar_size() else {
        return Err(GgufError::Invalid(
            "nested GGUF arrays are not supported".to_string(),
        ));
    };
    let byte_len = len
        .checked_mul(item_size)
        .ok_or_else(|| GgufError::Invalid("array length overflow".to_string()))?;
    skip_metadata_bytes(reader, byte_len, prefix_len)?;
    Ok(MetadataValue::Skipped)
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

fn build_inspection(metadata: &GgufMetadataInspection) -> AssetInspection {
    let model_type = metadata.general_type.as_deref();
    let architecture = metadata.general_architecture.as_deref();
    let is_asr_model = metadata
        .general_tags
        .iter()
        .any(|tag| tag == "automatic-speech-recognition");
    let clip_has_vision_encoder = metadata.clip_has_vision_encoder == Some(true);
    let clip_has_audio_encoder = metadata.clip_has_audio_encoder == Some(true);
    let clip_has_audio_generation_encoder =
        metadata.clip_has_audio_generation_encoder == Some(true);
    let projector_type = |supported: bool, specific: &Option<String>| {
        supported
            .then(|| {
                metadata
                    .clip_projector_type
                    .clone()
                    .or_else(|| specific.clone())
            })
            .flatten()
    };
    let provided_vision_projector_type = projector_type(
        clip_has_vision_encoder,
        &metadata.clip_vision_projector_type,
    );
    let provided_audio_projector_type =
        projector_type(clip_has_audio_encoder, &metadata.clip_audio_projector_type);
    let provided_audio_generation_projector_type = projector_type(
        clip_has_audio_generation_encoder,
        &metadata.clip_audio_generation_projector_type,
    );
    let is_projector = model_type == Some("mmproj")
        || architecture == Some("clip")
        || metadata.clip_projector_type.is_some()
        || provided_vision_projector_type.is_some()
        || provided_audio_projector_type.is_some()
        || provided_audio_generation_projector_type.is_some();
    let compatible_vision_projector_types = if is_projector {
        Vec::new()
    } else {
        resolve_compatible_vision_projector_types(
            architecture,
            clip_has_vision_encoder,
            is_asr_model,
        )
    };
    let compatible_audio_projector_types = if is_projector {
        Vec::new()
    } else {
        resolve_compatible_audio_projector_types(architecture, is_asr_model)
    };
    let compatible_audio_generation_projector_types = if is_projector {
        Vec::new()
    } else {
        resolve_compatible_audio_generation_projector_types(architecture)
    };
    let vision_capable = if is_projector {
        clip_has_vision_encoder
    } else {
        clip_has_vision_encoder || !compatible_vision_projector_types.is_empty()
    };
    let role = if is_projector {
        AssetRole::Projector
    } else if model_type.is_some()
        || architecture.is_some()
        || clip_has_vision_encoder
        || clip_has_audio_encoder
        || clip_has_audio_generation_encoder
        || is_asr_model
    {
        AssetRole::Model
    } else {
        AssetRole::Unknown
    };

    AssetInspection {
        version: AssetInspection::VERSION,
        role,
        architecture: architecture.map(str::to_string),
        trained_context_size: metadata.trained_context_size.filter(|size| *size > 0),
        vision_capable,
        audio_capable: clip_has_audio_encoder || !compatible_audio_projector_types.is_empty(),
        audio_generation_capable: clip_has_audio_generation_encoder
            || !compatible_audio_generation_projector_types.is_empty(),
        compatible_vision_projector_types,
        compatible_audio_projector_types,
        compatible_audio_generation_projector_types,
        provided_vision_projector_type,
        provided_audio_projector_type,
        provided_audio_generation_projector_type,
    }
}

fn resolve_compatible_vision_projector_types(
    architecture: Option<&str>,
    clip_has_vision_encoder: bool,
    is_asr_model: bool,
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
        "qwen3vl" | "qwen3vlmoe" if !is_asr_model => (&["qwen3vl_merger"], false),
        _ => (&[], false),
    };
    if requires_vision_encoder && !clip_has_vision_encoder {
        return Vec::new();
    }
    let mut compatible_types = Vec::with_capacity(types.len());
    compatible_types.extend(types.iter().map(|value| (*value).to_string()));
    compatible_types
}

fn resolve_compatible_audio_projector_types(
    architecture: Option<&str>,
    is_asr_model: bool,
) -> Vec<String> {
    // Backbones do not currently declare compatible projector types in GGUF.
    // Keep that backend compatibility knowledge at the inspection boundary so
    // lifecycle pairing can remain a generic required/provided-role match.
    if architecture == Some(QWEN3TTS_ARCHITECTURE) {
        return vec![QWEN3TTS_AUDIO_PROJECTOR.to_string()];
    }
    if !is_asr_model {
        return Vec::new();
    }
    match architecture {
        Some("qwen3vl" | "qwen3vlmoe") => vec!["qwen3a".to_string()],
        Some("lfm2") => vec!["lfm2a".to_string()],
        _ => Vec::new(),
    }
}

fn resolve_compatible_audio_generation_projector_types(architecture: Option<&str>) -> Vec<String> {
    match architecture {
        Some(QWEN3TTS_ARCHITECTURE) => {
            vec![QWEN3TTS_AUDIO_GENERATION_PROJECTOR.to_string()]
        }
        _ => Vec::new(),
    }
}

fn has_useful_metadata(metadata: &GgufMetadataInspection) -> bool {
    metadata.general_type.is_some()
        || metadata.general_architecture.is_some()
        || metadata.trained_context_size.is_some()
        || !metadata.general_tags.is_empty()
        || metadata.pooling_type.is_some()
        || metadata.clip_projector_type.is_some()
        || metadata.clip_vision_projector_type.is_some()
        || metadata.clip_audio_projector_type.is_some()
        || metadata.clip_audio_generation_projector_type.is_some()
        || metadata.clip_has_vision_encoder.is_some()
        || metadata.clip_has_audio_encoder.is_some()
        || metadata.clip_has_audio_generation_encoder.is_some()
}

fn is_target_key(key: &str) -> bool {
    TARGET_KEYS.contains(&key) || is_pooling_key(key)
}

fn is_pooling_key(key: &str) -> bool {
    key == "general.pooling_type" || key.ends_with(".pooling_type")
}

fn context_length_architecture(key: &str) -> Option<&str> {
    key.strip_suffix(".context_length")
        .filter(|architecture| !architecture.is_empty())
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

fn normalize_strings(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| normalize_optional_string(&value))
        .collect()
}
