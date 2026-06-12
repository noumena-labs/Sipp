//! GGUF cache planning and split-file writing.
//!
//! This crate is intentionally independent from llama.cpp and `cogentlm-sys`.
//! The browser can use it on the wasm32 Emscripten path for model acquisition
//! without moving inference ownership away from the existing WebGPU runtime.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

mod bytes;
mod inspection;

use bytes::{
    align_to, copy_exact_from, read_raw_value, u32_from_usize, u64_from_usize, usize_from_u32,
    usize_from_u64, write_string, write_u32, write_u64, write_zeros, CountingReader,
    CountingWriter, ReadAtCursor,
};
pub use inspection::{
    detect_model_from_gguf_bytes, inspect_gguf_metadata, inspect_gguf_metadata_path,
    AssetInspection, AssetRole, GgufMetadataInspection, ModelDetection, ModelDetectionMethod,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../tests/shard/support.rs"]
mod support;

#[cfg(test)]
#[path = "../tests/shard/root_tests.rs"]
mod root_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
const GGUF_MAGIC: u32 = 0x4655_4747;
const SUPPORTED_GGUF_VERSIONS: &[u32] = &[2, 3];
const DEFAULT_ALIGNMENT: u64 = 32;
pub(crate) const BYTES_PER_MIB_USIZE: usize = 1024 * 1024;
const BYTES_PER_MIB_U64: u64 = 1024 * 1024;
const BYTES_PER_GIB_U64: u64 = 1024 * BYTES_PER_MIB_U64;
const DEFAULT_DIRECT_LOAD_MAX_BYTES: u64 = 2 * BYTES_PER_GIB_U64;
const DEFAULT_SHARD_MAX_BYTES: u64 = 512 * BYTES_PER_MIB_U64;
const COPY_BUFFER_BYTES: usize = 8 * BYTES_PER_MIB_USIZE;

const SPLIT_NO_KEY: &str = "split.no";
const SPLIT_COUNT_KEY: &str = "split.count";
const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";
const GENERAL_ALIGNMENT_KEY: &str = "general.alignment";

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid GGUF file: {0}")]
    Invalid(String),
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("GGUF metadata prefix exceeded {max_bytes} bytes")]
    MetadataTooLarge { max_bytes: usize },
    #[error("source GGUF is already split into {0} files")]
    AlreadySplit(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrowserCacheLayout {
    SingleFile,
    SplitGguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrowserCachePolicy {
    pub direct_load_max_bytes: u64,
    pub shard_max_bytes: u64,
}

impl Default for BrowserCachePolicy {
    fn default() -> Self {
        Self {
            direct_load_max_bytes: DEFAULT_DIRECT_LOAD_MAX_BYTES,
            shard_max_bytes: DEFAULT_SHARD_MAX_BYTES,
        }
    }
}

impl BrowserCachePolicy {
    pub fn resolve_layout(&self, source_bytes: Option<u64>) -> BrowserCacheLayout {
        match source_bytes {
            Some(bytes) if bytes <= self.direct_load_max_bytes => BrowserCacheLayout::SingleFile,
            _ => BrowserCacheLayout::SplitGguf,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufSplitOptions {
    pub shard_max_bytes: u64,
}

impl Default for GgufSplitOptions {
    fn default() -> Self {
        Self {
            shard_max_bytes: BrowserCachePolicy::default().shard_max_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufShard {
    pub index: u16,
    pub count: u16,
    pub path: PathBuf,
    pub tensor_count: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSplitManifest {
    pub source_bytes: u64,
    pub total_tensors: usize,
    pub shards: Vec<GgufShard>,
}

pub trait GgufReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError>;
}

pub trait GgufShardSink {
    type Writer: Write;

    fn create_shard(
        &mut self,
        path: &Path,
        index: u16,
        count: u16,
    ) -> Result<Self::Writer, GgufError>;

    fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub(crate) fn from_u32(value: u32) -> Result<Self, GgufError> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::Invalid(format!("unknown value type {value}"))),
        }
    }

    pub(crate) fn scalar_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

#[derive(Debug, Clone)]
struct KvEntry {
    key: String,
    value_type: GgufValueType,
    raw_value: Vec<u8>,
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: u32,
    source_offset: u64,
    source_span: u64,
}

#[derive(Debug, Clone)]
struct GgufMetadata {
    version: u32,
    kvs: Vec<KvEntry>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
    alignment: u64,
}

#[derive(Debug, Clone)]
struct ShardPlan {
    tensors: Vec<usize>,
    source_spans: u64,
}

struct FileReadAt {
    file: File,
}

impl FileReadAt {
    fn new(file: File) -> Self {
        Self { file }
    }
}

impl GgufReadAt for FileReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(dst)?;
        Ok(())
    }
}

struct FileShardSink;

impl GgufShardSink for FileShardSink {
    type Writer = File;

    fn create_shard(
        &mut self,
        path: &Path,
        _index: u16,
        _count: u16,
    ) -> Result<Self::Writer, GgufError> {
        if let Some(parent) = shard_parent_dir(path) {
            fs::create_dir_all(parent)?;
        }
        Ok(File::create(path)?)
    }

    fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
        Ok(writer.metadata()?.len())
    }
}

fn shard_parent_dir(path: &Path) -> Option<&Path> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
}

pub fn split_path(prefix: impl AsRef<Path>, index: u16, count: u16) -> PathBuf {
    let prefix = prefix.as_ref();
    let name = format!(
        "{}-{:05}-of-{:05}.gguf",
        prefix.to_string_lossy(),
        u32::from(index) + 1,
        u32::from(count)
    );
    PathBuf::from(name)
}

pub fn split_gguf_file(
    input_path: impl AsRef<Path>,
    output_prefix: impl AsRef<Path>,
    options: GgufSplitOptions,
) -> Result<GgufSplitManifest, GgufError> {
    let input_path = input_path.as_ref();
    let source_bytes = fs::metadata(input_path)?.len();
    let mut source = FileReadAt::new(File::open(input_path)?);
    let mut sink = FileShardSink;
    split_gguf(source_bytes, &mut source, output_prefix, options, &mut sink)
}

pub fn split_gguf<S, K>(
    source_bytes: u64,
    source: &mut S,
    output_prefix: impl AsRef<Path>,
    options: GgufSplitOptions,
    sink: &mut K,
) -> Result<GgufSplitManifest, GgufError>
where
    S: GgufReadAt,
    K: GgufShardSink,
{
    let output_prefix = output_prefix.as_ref();
    let (metadata, plans, shard_count) =
        prepare_split(source_bytes, source, options.shard_max_bytes)?;

    let mut shards = Vec::with_capacity(plans.len());
    for (plan_index, plan) in plans.iter().enumerate() {
        let index = u16::try_from(plan_index)
            .map_err(|_| GgufError::Invalid("too many GGUF shards".to_string()))?;
        let path = split_path(output_prefix, index, shard_count);

        let mut writer = CountingWriter::new(sink.create_shard(&path, index, shard_count)?);
        write_shard(source, &metadata, plan, index, shard_count, &mut writer)?;
        let bytes = sink.finish_shard(writer.into_inner()?)?;
        shards.push(GgufShard {
            index,
            count: shard_count,
            path,
            tensor_count: plan.tensors.len(),
            bytes,
        });
    }

    Ok(GgufSplitManifest {
        source_bytes,
        total_tensors: metadata.tensors.len(),
        shards,
    })
}

pub fn plan_gguf_split<S>(
    source_bytes: u64,
    source: &mut S,
    output_prefix: impl AsRef<Path>,
    options: GgufSplitOptions,
) -> Result<GgufSplitManifest, GgufError>
where
    S: GgufReadAt,
{
    let output_prefix = output_prefix.as_ref();
    let (metadata, plans, shard_count) =
        prepare_split(source_bytes, source, options.shard_max_bytes)?;
    let shards = plans
        .iter()
        .enumerate()
        .map(|(plan_index, plan)| {
            let index = u16::try_from(plan_index)
                .map_err(|_| GgufError::Invalid("too many GGUF shards".to_string()))?;
            Ok(GgufShard {
                index,
                count: shard_count,
                path: split_path(output_prefix, index, shard_count),
                tensor_count: plan.tensors.len(),
                bytes: plan.source_spans,
            })
        })
        .collect::<Result<Vec<_>, GgufError>>()?;

    Ok(GgufSplitManifest {
        source_bytes,
        total_tensors: metadata.tensors.len(),
        shards,
    })
}

fn prepare_split<S: GgufReadAt>(
    source_bytes: u64,
    source: &mut S,
    shard_max_bytes: u64,
) -> Result<(GgufMetadata, Vec<ShardPlan>, u16), GgufError> {
    let mut metadata = parse_metadata_from_source(source_bytes, source)?;
    assign_source_spans(&mut metadata, source_bytes)?;
    ensure_monolithic(&metadata)?;
    let plans = plan_shards(&metadata.tensors, shard_max_bytes)?;
    let shard_count = u16::try_from(plans.len())
        .map_err(|_| GgufError::Invalid("too many GGUF shards".to_string()))?;
    Ok((metadata, plans, shard_count))
}

fn ensure_monolithic(metadata: &GgufMetadata) -> Result<(), GgufError> {
    if let Some(count) = metadata
        .kvs
        .iter()
        .find(|kv| kv.key == SPLIT_COUNT_KEY)
        .and_then(read_u16_kv)
    {
        if count > 1 {
            return Err(GgufError::AlreadySplit(count));
        }
    }
    Ok(())
}

fn parse_metadata_from_source<S: GgufReadAt>(
    source_bytes: u64,
    source: &mut S,
) -> Result<GgufMetadata, GgufError> {
    let mut cursor = ReadAtCursor::new(source, source_bytes);
    parse_metadata(&mut cursor)
}

fn parse_metadata<R: Read>(reader: &mut R) -> Result<GgufMetadata, GgufError> {
    let mut reader = CountingReader::new(reader);
    let magic = reader.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::Invalid("missing GGUF magic".to_string()));
    }
    let version = reader.read_u32()?;
    if !SUPPORTED_GGUF_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = reader.read_u64()?;
    let kv_count = reader.read_u64()?;
    let kv_count_usize = usize_from_u64(kv_count, "kv count")?;
    let tensor_count_usize = usize_from_u64(tensor_count, "tensor count")?;

    let mut kvs = Vec::with_capacity(kv_count_usize);
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..kv_count_usize {
        let key = reader.read_string()?;
        let value_type = GgufValueType::from_u32(reader.read_u32()?)?;
        let raw_value = read_raw_value(&mut reader, value_type)?;
        let kv = KvEntry {
            key,
            value_type,
            raw_value,
        };
        if kv.key == GENERAL_ALIGNMENT_KEY {
            alignment = read_alignment(&kv).unwrap_or(DEFAULT_ALIGNMENT);
        }
        kvs.push(kv);
    }

    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::Invalid(format!(
            "invalid GGUF alignment {alignment}"
        )));
    }

    let mut tensors = Vec::with_capacity(tensor_count_usize);
    for _ in 0..tensor_count_usize {
        let name = reader.read_string()?;
        let n_dimensions = usize_from_u32(reader.read_u32()?, "tensor dimension count")?;
        if n_dimensions == 0 || n_dimensions > 16 {
            return Err(GgufError::Invalid(format!(
                "invalid tensor dimension count {n_dimensions}"
            )));
        }

        let mut dimensions = Vec::with_capacity(n_dimensions);
        for _ in 0..n_dimensions {
            dimensions.push(reader.read_u64()?);
        }
        let tensor_type = reader.read_u32()?;
        let source_offset = reader.read_u64()?;
        tensors.push(TensorInfo {
            name,
            dimensions,
            tensor_type,
            source_offset,
            source_span: 0,
        });
    }

    let data_offset = align_to(reader.position(), alignment)?;
    Ok(GgufMetadata {
        version,
        kvs,
        tensors,
        data_offset,
        alignment,
    })
}

fn assign_source_spans(metadata: &mut GgufMetadata, file_size: u64) -> Result<(), GgufError> {
    if file_size < metadata.data_offset {
        return Err(GgufError::Invalid(
            "file ends before tensor data".to_string(),
        ));
    }

    let mut order: Vec<usize> = (0..metadata.tensors.len()).collect();
    order.sort_by_key(|&idx| metadata.tensors[idx].source_offset);
    for window in order.windows(2) {
        let current = window[0];
        let next = window[1];
        let start = metadata.tensors[current].source_offset;
        let end = metadata.tensors[next].source_offset;
        if end <= start {
            return Err(GgufError::Invalid(
                "tensor offsets are not strictly increasing".to_string(),
            ));
        }
        metadata.tensors[current].source_span = end - start;
    }
    if let Some(&last) = order.last() {
        let source_start = metadata
            .data_offset
            .checked_add(metadata.tensors[last].source_offset)
            .ok_or_else(|| GgufError::Invalid("tensor offset overflow".to_string()))?;
        if source_start > file_size {
            return Err(GgufError::Invalid(
                "last tensor starts after end of file".to_string(),
            ));
        }
        metadata.tensors[last].source_span = file_size - source_start;
    }
    Ok(())
}

fn plan_shards(tensors: &[TensorInfo], shard_max_bytes: u64) -> Result<Vec<ShardPlan>, GgufError> {
    if tensors.is_empty() {
        return Err(GgufError::Invalid("GGUF contains no tensors".to_string()));
    }
    if shard_max_bytes == 0 {
        return Err(GgufError::Invalid(
            "shard_max_bytes must be positive".to_string(),
        ));
    }

    let mut plans: Vec<ShardPlan> = Vec::new();
    let mut current = ShardPlan {
        tensors: Vec::new(),
        source_spans: 0,
    };
    for (idx, tensor) in tensors.iter().enumerate() {
        if tensor.source_span == 0 {
            return Err(GgufError::Invalid(format!(
                "tensor '{}' has no source bytes",
                tensor.name
            )));
        }
        if !current.tensors.is_empty()
            && current
                .source_spans
                .checked_add(tensor.source_span)
                .is_some_and(|next| next > shard_max_bytes)
        {
            plans.push(current);
            current = ShardPlan {
                tensors: Vec::new(),
                source_spans: 0,
            };
        }
        current.tensors.push(idx);
        current.source_spans = current
            .source_spans
            .checked_add(tensor.source_span)
            .ok_or_else(|| GgufError::Invalid("shard size overflow".to_string()))?;
    }
    if !current.tensors.is_empty() {
        plans.push(current);
    }
    Ok(plans)
}

fn write_shard<S: GgufReadAt, W: Write>(
    source: &mut S,
    metadata: &GgufMetadata,
    plan: &ShardPlan,
    index: u16,
    count: u16,
    mut output: &mut CountingWriter<W>,
) -> Result<(), GgufError> {
    let shard_alignment = if index == 0 {
        metadata.alignment
    } else {
        DEFAULT_ALIGNMENT
    };
    let shard_kvs = build_shard_kvs(metadata, index, count)?;

    write_u32(&mut output, GGUF_MAGIC)?;
    write_u32(&mut output, metadata.version)?;
    write_u64(&mut output, shard_tensor_count_value(plan.tensors.len())?)?;
    write_u64(&mut output, shard_kv_count_value(shard_kvs.len())?)?;

    for kv in &shard_kvs {
        write_string(&mut output, &kv.key)?;
        write_u32(&mut output, kv.value_type as u32)?;
        output.write_all(&kv.raw_value)?;
    }

    let mut shard_offset = 0u64;
    let mut shard_tensor_offsets = Vec::with_capacity(plan.tensors.len());
    for &tensor_idx in &plan.tensors {
        let tensor = &metadata.tensors[tensor_idx];
        shard_tensor_offsets.push(shard_offset);
        write_string(&mut output, &tensor.name)?;
        let dimension_count = tensor_dimension_count_value(tensor.dimensions.len())?;
        write_u32(&mut output, dimension_count)?;
        for &dimension in &tensor.dimensions {
            write_u64(&mut output, dimension)?;
        }
        write_u32(&mut output, tensor.tensor_type)?;
        write_u64(&mut output, shard_offset)?;
        shard_offset = shard_offset
            .checked_add(tensor.source_span)
            .ok_or_else(|| GgufError::Invalid("shard tensor offset overflow".to_string()))?;
    }

    let metadata_end = output.position();
    let data_offset = align_to(metadata_end, shard_alignment)?;
    write_zeros(&mut output, data_offset - metadata_end)?;

    let mut copy_buffer = vec![0u8; COPY_BUFFER_BYTES];
    for (local_idx, &tensor_idx) in plan.tensors.iter().enumerate() {
        let tensor = &metadata.tensors[tensor_idx];
        let expected_position = shard_write_position(data_offset, shard_tensor_offsets[local_idx])?;
        ensure_shard_position(output.position(), expected_position)?;

        let source_position = metadata
            .data_offset
            .checked_add(tensor.source_offset)
            .ok_or_else(|| GgufError::Invalid("source tensor offset overflow".to_string()))?;
        copy_exact_from(
            source,
            source_position,
            &mut output,
            tensor.source_span,
            &mut copy_buffer,
        )?;
    }

    output.flush()?;
    Ok(())
}

fn shard_tensor_count_value(len: usize) -> Result<u64, GgufError> {
    u64_from_usize(len, "shard tensor count")
}

fn shard_kv_count_value(len: usize) -> Result<u64, GgufError> {
    u64_from_usize(len, "shard kv count")
}

fn tensor_dimension_count_value(len: usize) -> Result<u32, GgufError> {
    u32_from_usize(len, "tensor dimension count")
}

fn shard_write_position(data_offset: u64, shard_tensor_offset: u64) -> Result<u64, GgufError> {
    data_offset
        .checked_add(shard_tensor_offset)
        .ok_or_else(|| GgufError::Invalid("shard write offset overflow".to_string()))
}

fn ensure_shard_position(actual: u64, expected: u64) -> Result<(), GgufError> {
    if actual != expected {
        return Err(GgufError::Invalid(
            "internal shard offset mismatch".to_string(),
        ));
    }
    Ok(())
}

fn build_shard_kvs(
    metadata: &GgufMetadata,
    index: u16,
    count: u16,
) -> Result<Vec<KvEntry>, GgufError> {
    let mut kvs = Vec::new();
    if index == 0 {
        kvs.extend(
            metadata
                .kvs
                .iter()
                .filter(|kv| {
                    kv.key != SPLIT_NO_KEY
                        && kv.key != SPLIT_COUNT_KEY
                        && kv.key != SPLIT_TENSORS_COUNT_KEY
                })
                .cloned(),
        );
    }
    kvs.push(u16_kv(SPLIT_NO_KEY, index));
    kvs.push(u16_kv(SPLIT_COUNT_KEY, count));
    let total_tensors = i32::try_from(metadata.tensors.len())
        .map_err(|_| GgufError::Invalid("split.tensors.count does not fit i32".to_string()))?;
    kvs.push(i32_kv(SPLIT_TENSORS_COUNT_KEY, total_tensors));
    Ok(kvs)
}

fn u16_kv(key: &str, value: u16) -> KvEntry {
    KvEntry {
        key: key.to_string(),
        value_type: GgufValueType::Uint16,
        raw_value: value.to_le_bytes().to_vec(),
    }
}

fn i32_kv(key: &str, value: i32) -> KvEntry {
    KvEntry {
        key: key.to_string(),
        value_type: GgufValueType::Int32,
        raw_value: value.to_le_bytes().to_vec(),
    }
}

fn read_u16_kv(kv: &KvEntry) -> Option<u16> {
    if kv.value_type != GgufValueType::Uint16 || kv.raw_value.len() != 2 {
        return None;
    }
    Some(u16::from_le_bytes(kv.raw_value.as_slice().try_into().ok()?))
}

fn read_alignment(kv: &KvEntry) -> Option<u64> {
    match kv.value_type {
        GgufValueType::Uint32 if kv.raw_value.len() == 4 => {
            Some(u32::from_le_bytes(kv.raw_value.as_slice().try_into().ok()?) as u64)
        }
        GgufValueType::Uint64 if kv.raw_value.len() == 8 => {
            Some(u64::from_le_bytes(kv.raw_value.as_slice().try_into().ok()?))
        }
        _ => None,
    }
}
