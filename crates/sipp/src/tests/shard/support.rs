//! Shared deterministic fixtures for `lib`, `inspection`, and `bytes` tests.
//!
//! Provides model-free GGUF byte builders, in-memory read/sink handles, and
//! counter-based temporary paths so tests avoid native execution, timing,
//! network access, and machine-local model files.

use std::fs;
use std::io::{self, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::shard::bytes::{align_to, usize_from_u64, write_string, write_u32, write_u64};
use crate::shard::{
    assign_source_spans, parse_metadata, GgufError, GgufMetadata, GgufReadAt, GgufShardSink,
    GgufValueType, KvEntry, TensorInfo, DEFAULT_ALIGNMENT, GENERAL_ALIGNMENT_KEY, GGUF_MAGIC,
    SPLIT_COUNT_KEY, SPLIT_NO_KEY,
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(super) enum MetadataValue<'a> {
    String(&'a str),
    Bool(bool),
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    ArrayU32(&'a [u32]),
    ArrayString(&'a [&'a str]),
    ArrayHeader { item_type: GgufValueType, len: u64 },
}

pub(super) struct FakeTensor {
    pub(super) name: String,
    pub(super) dimensions: Vec<u64>,
    pub(super) tensor_type: u32,
    pub(super) source_offset: Option<u64>,
    pub(super) data: Vec<u8>,
}

impl FakeTensor {
    pub(super) fn new(name: &str, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            dimensions: vec![1],
            tensor_type: 0,
            source_offset: None,
            data,
        }
    }

    pub(super) fn with_dimensions(mut self, dimensions: Vec<u64>) -> Self {
        self.dimensions = dimensions;
        self
    }
}

pub(super) fn metadata_gguf(entries: &[(&str, MetadataValue<'_>)]) -> Vec<u8> {
    metadata_gguf_version(3, entries)
}

pub(super) fn metadata_gguf_version(
    version: u32,
    entries: &[(&str, MetadataValue<'_>)],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, GGUF_MAGIC).expect("magic");
    write_u32(&mut bytes, version).expect("version");
    write_u64(&mut bytes, 0).expect("tensor count");
    write_u64(
        &mut bytes,
        u64::try_from(entries.len()).expect("metadata entry count"),
    )
    .expect("kv count");
    for (key, value) in entries {
        push_metadata_entry(&mut bytes, key, value);
    }
    bytes
}

pub(super) fn gguf_with_tensors(
    entries: &[(&str, MetadataValue<'_>)],
    tensors: &[FakeTensor],
    alignment: u64,
) -> Vec<u8> {
    let include_alignment = alignment != DEFAULT_ALIGNMENT;
    let mut metadata = Vec::new();
    write_u32(&mut metadata, GGUF_MAGIC).expect("magic");
    write_u32(&mut metadata, 3).expect("version");
    write_u64(
        &mut metadata,
        u64::try_from(tensors.len()).expect("tensor count"),
    )
    .expect("tensor count");
    write_u64(
        &mut metadata,
        u64::try_from(entries.len() + usize::from(include_alignment)).expect("kv count"),
    )
    .expect("kv count");

    if include_alignment {
        push_metadata_entry(
            &mut metadata,
            GENERAL_ALIGNMENT_KEY,
            &MetadataValue::Uint32(u32::try_from(alignment).expect("alignment")),
        );
    }
    for (key, value) in entries {
        push_metadata_entry(&mut metadata, key, value);
    }

    let mut tensor_data = Vec::new();
    let mut tensor_offsets = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let next_offset = if let Some(offset) = tensor.source_offset {
            offset
        } else {
            align_to(
                u64::try_from(tensor_data.len()).expect("tensor data length"),
                alignment,
            )
            .expect("aligned tensor offset")
        };
        let next_offset_usize = usize_from_u64(next_offset, "tensor offset").expect("offset");
        if tensor_data.len() < next_offset_usize {
            tensor_data.resize(next_offset_usize, 0);
        }
        tensor_offsets.push(next_offset);
        tensor_data.extend_from_slice(&tensor.data);
    }

    for (tensor, offset) in tensors.iter().zip(tensor_offsets) {
        write_string(&mut metadata, &tensor.name).expect("tensor name");
        write_u32(
            &mut metadata,
            u32::try_from(tensor.dimensions.len()).expect("dimension count"),
        )
        .expect("dimension count");
        for &dimension in &tensor.dimensions {
            write_u64(&mut metadata, dimension).expect("dimension");
        }
        write_u32(&mut metadata, tensor.tensor_type).expect("tensor type");
        write_u64(&mut metadata, offset).expect("tensor offset");
    }

    let data_offset = align_to(
        u64::try_from(metadata.len()).expect("metadata length"),
        alignment,
    )
    .expect("data offset");
    metadata.resize(
        usize_from_u64(data_offset, "data offset").expect("data offset"),
        0,
    );
    metadata.extend_from_slice(&tensor_data);
    metadata
}

pub(super) fn split_fixture_gguf() -> Vec<u8> {
    gguf_with_tensors(
        &[("general.architecture", MetadataValue::String("llama"))],
        &[
            FakeTensor::new("blk.0.weight", vec![1u8; 64]),
            FakeTensor::new("blk.1.weight", vec![2u8; 64]),
            FakeTensor::new("output.weight", vec![3u8; 64]),
        ],
        DEFAULT_ALIGNMENT,
    )
}

pub(super) fn push_metadata_entry(bytes: &mut Vec<u8>, key: &str, value: &MetadataValue<'_>) {
    write_string(bytes, key).expect("metadata key");
    match value {
        MetadataValue::String(value) => {
            write_u32(bytes, GgufValueType::String as u32).expect("string type");
            write_string(bytes, value).expect("string value");
        }
        MetadataValue::Bool(value) => {
            write_u32(bytes, GgufValueType::Bool as u32).expect("bool type");
            bytes.push(u8::from(*value));
        }
        MetadataValue::Uint8(value) => {
            write_u32(bytes, GgufValueType::Uint8 as u32).expect("u8 type");
            bytes.push(*value);
        }
        MetadataValue::Uint16(value) => {
            write_u32(bytes, GgufValueType::Uint16 as u32).expect("u16 type");
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataValue::Uint32(value) => {
            write_u32(bytes, GgufValueType::Uint32 as u32).expect("u32 type");
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataValue::Int32(value) => {
            write_u32(bytes, GgufValueType::Int32 as u32).expect("i32 type");
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataValue::Uint64(value) => {
            write_u32(bytes, GgufValueType::Uint64 as u32).expect("u64 type");
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataValue::ArrayU32(values) => {
            write_u32(bytes, GgufValueType::Array as u32).expect("array type");
            write_u32(bytes, GgufValueType::Uint32 as u32).expect("array item type");
            write_u64(bytes, u64::try_from(values.len()).expect("array length"))
                .expect("array length");
            for value in *values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        MetadataValue::ArrayString(values) => {
            write_u32(bytes, GgufValueType::Array as u32).expect("array type");
            write_u32(bytes, GgufValueType::String as u32).expect("array item type");
            write_u64(bytes, u64::try_from(values.len()).expect("array length"))
                .expect("array length");
            for value in *values {
                write_string(bytes, value).expect("array string");
            }
        }
        MetadataValue::ArrayHeader { item_type, len } => {
            write_u32(bytes, GgufValueType::Array as u32).expect("array type");
            write_u32(bytes, *item_type as u32).expect("array item type");
            write_u64(bytes, *len).expect("array length");
        }
    }
}

pub(super) fn parse_bytes(bytes: &[u8]) -> GgufMetadata {
    let mut cursor = Cursor::new(bytes);
    let mut parsed = parse_metadata(&mut cursor).expect("parse shard");
    assign_source_spans(
        &mut parsed,
        u64::try_from(bytes.len()).expect("bytes length"),
    )
    .expect("spans");
    parsed
}

pub(super) fn parse_file(path: &Path) -> GgufMetadata {
    let mut file = fs::File::open(path).expect("open shard");
    let bytes = file.metadata().expect("metadata").len();
    let mut parsed = parse_metadata(&mut file).expect("parse shard");
    assign_source_spans(&mut parsed, bytes).expect("spans");
    parsed
}

pub(super) fn string_kv(key: &str, value: &str) -> KvEntry {
    let mut raw_value = Vec::new();
    write_string(&mut raw_value, value).expect("string raw value");
    KvEntry {
        key: key.to_string(),
        value_type: GgufValueType::String,
        raw_value,
    }
}

pub(super) fn read_split_no(metadata: &GgufMetadata) -> Option<u16> {
    metadata
        .kvs
        .iter()
        .find(|kv| kv.key == SPLIT_NO_KEY)
        .and_then(crate::shard::read_u16_kv)
}

pub(super) fn read_split_count(metadata: &GgufMetadata) -> Option<u16> {
    metadata
        .kvs
        .iter()
        .find(|kv| kv.key == SPLIT_COUNT_KEY)
        .and_then(crate::shard::read_u16_kv)
}

pub(super) fn unique_temp_dir() -> PathBuf {
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("sipp-shard-test-{}-{id}", std::process::id()))
}

pub(super) struct MemoryReadAt {
    pub(super) bytes: Vec<u8>,
    pub(super) fail_at: Option<u64>,
}

impl MemoryReadAt {
    pub(super) fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            fail_at: None,
        }
    }

    pub(super) fn with_fail_at(mut self, fail_at: u64) -> Self {
        self.fail_at = Some(fail_at);
        self
    }
}

impl GgufReadAt for MemoryReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        if self.fail_at == Some(offset) {
            return Err(GgufError::Invalid("planned read failure".to_string()));
        }
        let offset = usize_from_u64(offset, "memory read offset")?;
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| GgufError::Invalid("read offset overflow".to_string()))?;
        let Some(src) = self.bytes.get(offset..end) else {
            return Err(GgufError::Invalid(
                "memory read exceeds source length".to_string(),
            ));
        };
        dst.copy_from_slice(src);
        Ok(())
    }
}

pub(super) struct MemoryShard {
    pub(super) path: PathBuf,
    pub(super) bytes: Vec<u8>,
}

pub(super) struct MemoryShardSink {
    pub(super) shards: Vec<MemoryShard>,
    pub(super) fail_create: bool,
    pub(super) fail_finish: bool,
}

impl MemoryShardSink {
    pub(super) fn new() -> Self {
        Self {
            shards: Vec::new(),
            fail_create: false,
            fail_finish: false,
        }
    }

    pub(super) fn fail_create() -> Self {
        Self {
            shards: Vec::new(),
            fail_create: true,
            fail_finish: false,
        }
    }

    pub(super) fn fail_finish() -> Self {
        Self {
            shards: Vec::new(),
            fail_create: false,
            fail_finish: true,
        }
    }
}

impl GgufShardSink for MemoryShardSink {
    type Writer = MemoryShardWriter;

    fn create_shard(
        &mut self,
        path: &Path,
        _index: u16,
        _count: u16,
    ) -> Result<Self::Writer, GgufError> {
        if self.fail_create {
            return Err(GgufError::Invalid("planned create failure".to_string()));
        }
        Ok(MemoryShardWriter {
            path: path.to_path_buf(),
            bytes: Vec::new(),
        })
    }

    fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
        if self.fail_finish {
            return Err(GgufError::Invalid("planned finish failure".to_string()));
        }
        let bytes = u64::try_from(writer.bytes.len())
            .map_err(|_| GgufError::Invalid("memory shard length exceeds u64".to_string()))?;
        self.shards.push(MemoryShard {
            path: writer.path,
            bytes: writer.bytes,
        });
        Ok(bytes)
    }
}

pub(super) struct MemoryShardWriter {
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

pub(super) fn tensor(name: &str, offset: u64, span: u64) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        dimensions: vec![1],
        tensor_type: 0,
        source_offset: offset,
        source_span: span,
    }
}
