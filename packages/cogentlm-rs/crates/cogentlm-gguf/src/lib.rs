//! GGUF cache planning and split-file writing.
//!
//! This crate is intentionally independent from llama.cpp and `cogentlm-sys`.
//! The browser can use it on the wasm32 Emscripten path for model acquisition
//! without moving inference ownership away from the existing WebGPU runtime.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

const GGUF_MAGIC: u32 = 0x4655_4747;
const SUPPORTED_GGUF_VERSIONS: &[u32] = &[2, 3];
const DEFAULT_ALIGNMENT: u64 = 32;

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
            direct_load_max_bytes: 2 * 1024 * 1024 * 1024,
            shard_max_bytes: 512 * 1024 * 1024,
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
enum GgufValueType {
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
    fn from_u32(value: u32) -> Result<Self, GgufError> {
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

    fn scalar_size(self) -> Option<usize> {
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
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        Ok(File::create(path)?)
    }

    fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
        Ok(writer.metadata()?.len())
    }
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
    let mut metadata = parse_metadata_from_source(source)?;
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

fn parse_metadata_from_source<S: GgufReadAt>(source: &mut S) -> Result<GgufMetadata, GgufError> {
    let mut cursor = ReadAtCursor::new(source);
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
    write_u64(&mut output, plan.tensors.len() as u64)?;
    write_u64(&mut output, shard_kvs.len() as u64)?;

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
        write_u32(&mut output, tensor.dimensions.len() as u32)?;
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

    let mut copy_buffer = vec![0u8; 1024 * 1024];
    for (local_idx, &tensor_idx) in plan.tensors.iter().enumerate() {
        let tensor = &metadata.tensors[tensor_idx];
        let expected_position = data_offset
            .checked_add(shard_tensor_offsets[local_idx])
            .ok_or_else(|| GgufError::Invalid("shard write offset overflow".to_string()))?;
        if output.position() != expected_position {
            return Err(GgufError::Invalid(
                "internal shard offset mismatch".to_string(),
            ));
        }

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

fn read_raw_value<R: Read>(
    reader: &mut CountingReader<'_, R>,
    value_type: GgufValueType,
) -> Result<Vec<u8>, GgufError> {
    let mut raw = Vec::new();
    read_raw_value_into(reader, value_type, &mut raw)?;
    Ok(raw)
}

fn read_raw_value_into<R: Read>(
    reader: &mut CountingReader<'_, R>,
    value_type: GgufValueType,
    raw: &mut Vec<u8>,
) -> Result<(), GgufError> {
    match value_type {
        GgufValueType::String => {
            let len = reader.read_u64()?;
            raw.extend_from_slice(&len.to_le_bytes());
            let len = usize_from_u64(len, "string length")?;
            let bytes = reader.read_vec(len)?;
            raw.extend_from_slice(&bytes);
        }
        GgufValueType::Array => {
            let item_type_raw = reader.read_u32()?;
            raw.extend_from_slice(&item_type_raw.to_le_bytes());
            let item_type = GgufValueType::from_u32(item_type_raw)?;
            let len = reader.read_u64()?;
            raw.extend_from_slice(&len.to_le_bytes());
            let len = usize_from_u64(len, "array length")?;
            for _ in 0..len {
                read_raw_value_into(reader, item_type, raw)?;
            }
        }
        _ => {
            let size = value_type
                .scalar_size()
                .ok_or_else(|| GgufError::Invalid("invalid scalar value type".to_string()))?;
            raw.extend_from_slice(&reader.read_vec(size)?);
        }
    }
    Ok(())
}

fn copy_exact_from<S: GgufReadAt, W: Write>(
    source: &mut S,
    mut source_offset: u64,
    output: &mut W,
    mut bytes: u64,
    buffer: &mut [u8],
) -> Result<(), GgufError> {
    while bytes > 0 {
        let chunk = usize::try_from(bytes.min(buffer.len() as u64))
            .map_err(|_| GgufError::Invalid("copy chunk too large".to_string()))?;
        source.read_at(source_offset, &mut buffer[..chunk])?;
        output.write_all(&buffer[..chunk])?;
        source_offset = source_offset
            .checked_add(chunk as u64)
            .ok_or_else(|| GgufError::Invalid("copy offset overflow".to_string()))?;
        bytes -= chunk as u64;
    }
    Ok(())
}

fn write_zeros<W: Write>(writer: &mut W, bytes: u64) -> Result<(), GgufError> {
    const ZEROES: [u8; 64] = [0; 64];
    let mut remaining = bytes;
    while remaining > 0 {
        let chunk = usize::try_from(remaining.min(ZEROES.len() as u64))
            .map_err(|_| GgufError::Invalid("padding too large".to_string()))?;
        writer.write_all(&ZEROES[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<(), GgufError> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<(), GgufError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<(), GgufError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn align_to(value: u64, alignment: u64) -> Result<u64, GgufError> {
    if alignment == 0 {
        return Err(GgufError::Invalid("zero alignment".to_string()));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| GgufError::Invalid("alignment overflow".to_string()))
    }
}

fn usize_from_u64(value: u64, name: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Invalid(format!("{name} does not fit usize")))
}

fn usize_from_u32(value: u32, name: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Invalid(format!("{name} does not fit usize")))
}

struct ReadAtCursor<'a, S> {
    source: &'a mut S,
    position: u64,
}

impl<'a, S> ReadAtCursor<'a, S> {
    fn new(source: &'a mut S) -> Self {
        Self {
            source,
            position: 0,
        }
    }
}

impl<S: GgufReadAt> Read for ReadAtCursor<'_, S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        self.source
            .read_at(self.position, buf)
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
        self.position = self
            .position
            .checked_add(buf.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "read position overflow"))?;
        Ok(buf.len())
    }
}

struct CountingWriter<W> {
    inner: W,
    position: u64,
}

impl<W> CountingWriter<W> {
    fn new(inner: W) -> Self {
        Self { inner, position: 0 }
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn into_inner(mut self) -> Result<W, GgufError>
    where
        W: Write,
    {
        self.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.position = self
            .position
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "write position overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.write_all(buf)?;
        self.position = self
            .position
            .checked_add(buf.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "write position overflow"))?;
        Ok(())
    }
}

struct CountingReader<'a, R> {
    inner: &'a mut R,
    position: u64,
}

impl<'a, R: Read> CountingReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self { inner, position: 0 }
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn read_exact_counted(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        self.inner.read_exact(buf)?;
        self.position = self
            .position
            .checked_add(buf.len() as u64)
            .ok_or_else(|| GgufError::Invalid("reader position overflow".to_string()))?;
        Ok(())
    }

    fn read_vec(&mut self, len: usize) -> Result<Vec<u8>, GgufError> {
        let mut buf = vec![0u8; len];
        self.read_exact_counted(&mut buf)?;
        Ok(buf)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let mut buf = [0u8; 4];
        self.read_exact_counted(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut buf = [0u8; 8];
        self.read_exact_counted(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = usize_from_u64(self.read_u64()?, "string length")?;
        let bytes = self.read_vec(len)?;
        String::from_utf8(bytes)
            .map_err(|_| GgufError::Invalid("GGUF string is not UTF-8".to_string()))
    }
}

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn chooses_single_file_under_direct_threshold() {
        let policy = BrowserCachePolicy {
            direct_load_max_bytes: 1024,
            shard_max_bytes: 128,
        };
        assert_eq!(
            policy.resolve_layout(Some(1024)),
            BrowserCacheLayout::SingleFile
        );
        assert_eq!(
            policy.resolve_layout(Some(1025)),
            BrowserCacheLayout::SplitGguf
        );
        assert_eq!(policy.resolve_layout(None), BrowserCacheLayout::SplitGguf);
    }

    #[test]
    fn writes_llama_compatible_split_shards() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("temp dir");
        let source = root.join("model.gguf");
        let prefix = root.join("model");
        let original = fake_gguf();
        fs::write(&source, &original).expect("write source");

        let manifest = split_gguf_file(
            &source,
            &prefix,
            GgufSplitOptions {
                shard_max_bytes: 128,
            },
        )
        .expect("split");

        assert_eq!(manifest.total_tensors, 3);
        assert_eq!(manifest.shards.len(), 2);
        assert_eq!(manifest.shards[0].tensor_count, 2);
        assert_eq!(manifest.shards[1].tensor_count, 1);
        assert_eq!(
            manifest.shards[0]
                .path
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "model-00001-of-00002.gguf"
        );

        let first = parse_file(&manifest.shards[0].path);
        let second = parse_file(&manifest.shards[1].path);
        assert_eq!(read_split_no(&first), Some(0));
        assert_eq!(read_split_count(&first), Some(2));
        assert_eq!(read_split_no(&second), Some(1));
        assert_eq!(read_split_count(&second), Some(2));
        assert!(first.kvs.iter().any(|kv| kv.key == "general.architecture"));
        assert!(!second.kvs.iter().any(|kv| kv.key == "general.architecture"));

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn splits_through_read_at_and_custom_sink() {
        let original = fake_gguf();
        let mut source = MemoryReadAt {
            bytes: original.clone(),
        };
        let mut sink = MemoryShardSink { shards: Vec::new() };

        let manifest = split_gguf(
            original.len() as u64,
            &mut source,
            "model",
            GgufSplitOptions {
                shard_max_bytes: 128,
            },
            &mut sink,
        )
        .expect("split");

        assert_eq!(manifest.shards.len(), 2);
        assert_eq!(sink.shards.len(), 2);
        assert_eq!(
            sink.shards[0].path,
            PathBuf::from("model-00001-of-00002.gguf")
        );
        let first = parse_bytes(&sink.shards[0].bytes);
        let second = parse_bytes(&sink.shards[1].bytes);
        assert_eq!(read_split_no(&first), Some(0));
        assert_eq!(read_split_no(&second), Some(1));
    }

    #[test]
    fn plans_split_count_through_read_at() {
        let original = fake_gguf();
        let mut source = MemoryReadAt {
            bytes: original.clone(),
        };
        let manifest = plan_gguf_split(
            original.len() as u64,
            &mut source,
            "model",
            GgufSplitOptions {
                shard_max_bytes: 128,
            },
        )
        .expect("plan");

        assert_eq!(manifest.shards.len(), 2);
        assert_eq!(
            manifest.shards[0].path,
            PathBuf::from("model-00001-of-00002.gguf")
        );
        assert_eq!(
            manifest.shards[1].path,
            PathBuf::from("model-00002-of-00002.gguf")
        );
    }

    fn parse_file(path: &Path) -> GgufMetadata {
        let mut file = File::open(path).expect("open shard");
        let bytes = file.metadata().expect("metadata").len();
        let mut parsed = parse_metadata(&mut file).expect("parse shard");
        assign_source_spans(&mut parsed, bytes).expect("spans");
        parsed
    }

    fn parse_bytes(bytes: &[u8]) -> GgufMetadata {
        let mut cursor = Cursor::new(bytes);
        let mut parsed = parse_metadata(&mut cursor).expect("parse shard");
        assign_source_spans(&mut parsed, bytes.len() as u64).expect("spans");
        parsed
    }

    fn read_split_no(metadata: &GgufMetadata) -> Option<u16> {
        metadata
            .kvs
            .iter()
            .find(|kv| kv.key == SPLIT_NO_KEY)
            .and_then(read_u16_kv)
    }

    fn read_split_count(metadata: &GgufMetadata) -> Option<u16> {
        metadata
            .kvs
            .iter()
            .find(|kv| kv.key == SPLIT_COUNT_KEY)
            .and_then(read_u16_kv)
    }

    fn fake_gguf() -> Vec<u8> {
        let tensors = vec![
            ("blk.0.weight", vec![1u8; 64]),
            ("blk.1.weight", vec![2u8; 64]),
            ("output.weight", vec![3u8; 64]),
        ];
        let mut metadata = Vec::new();
        write_header_and_metadata(&mut metadata, tensors.len() as u64);

        let mut tensor_data = Vec::new();
        let mut tensor_offsets = Vec::new();
        for (_, data) in &tensors {
            let next_offset = align_to(tensor_data.len() as u64, DEFAULT_ALIGNMENT).unwrap();
            tensor_data.resize(next_offset as usize, 0);
            tensor_offsets.push(next_offset);
            tensor_data.extend_from_slice(data);
        }

        for ((name, _), offset) in tensors.iter().zip(tensor_offsets) {
            write_string(&mut metadata, name).unwrap();
            write_u32(&mut metadata, 1).unwrap();
            write_u64(&mut metadata, 16).unwrap();
            write_u32(&mut metadata, 0).unwrap();
            write_u64(&mut metadata, offset).unwrap();
        }

        let data_offset = align_to(metadata.len() as u64, DEFAULT_ALIGNMENT).unwrap();
        metadata.resize(data_offset as usize, 0);
        metadata.extend_from_slice(&tensor_data);
        metadata
    }

    fn write_header_and_metadata(bytes: &mut Vec<u8>, tensor_count: u64) {
        let mut cursor = Cursor::new(bytes);
        write_u32(&mut cursor, GGUF_MAGIC).unwrap();
        write_u32(&mut cursor, 3).unwrap();
        write_u64(&mut cursor, tensor_count).unwrap();
        write_u64(&mut cursor, 1).unwrap();
        write_string(&mut cursor, "general.architecture").unwrap();
        write_u32(&mut cursor, GgufValueType::String as u32).unwrap();
        write_string(&mut cursor, "llama").unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "cogentlm-gguf-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    struct MemoryReadAt {
        bytes: Vec<u8>,
    }

    impl GgufReadAt for MemoryReadAt {
        fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
            let offset = offset as usize;
            let end = offset
                .checked_add(dst.len())
                .ok_or_else(|| GgufError::Invalid("read offset overflow".to_string()))?;
            dst.copy_from_slice(&self.bytes[offset..end]);
            Ok(())
        }
    }

    struct MemoryShard {
        path: PathBuf,
        bytes: Vec<u8>,
    }

    struct MemoryShardSink {
        shards: Vec<MemoryShard>,
    }

    impl GgufShardSink for MemoryShardSink {
        type Writer = MemoryShardWriter;

        fn create_shard(
            &mut self,
            path: &Path,
            _index: u16,
            _count: u16,
        ) -> Result<Self::Writer, GgufError> {
            Ok(MemoryShardWriter {
                path: path.to_path_buf(),
                bytes: Vec::new(),
            })
        }

        fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
            let bytes = writer.bytes.len() as u64;
            self.shards.push(MemoryShard {
                path: writer.path,
                bytes: writer.bytes,
            });
            Ok(bytes)
        }
    }

    struct MemoryShardWriter {
        path: PathBuf,
        bytes: Vec<u8>,
    }

    impl Write for MemoryShardWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
