//! Unit tests for the parent module.

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
fn gguf_value_type_round_trips_wire_codes() {
    for (value_type, raw) in [
        (GgufValueType::Uint8, 0),
        (GgufValueType::Uint16, 2),
        (GgufValueType::Int32, 5),
        (GgufValueType::String, 8),
        (GgufValueType::Uint64, 10),
    ] {
        assert_eq!(value_type as u32, raw);
        assert_eq!(
            GgufValueType::from_u32(raw).expect("known type"),
            value_type
        );
    }
}

#[test]
fn read_raw_value_preserves_scalar_and_string_bytes() {
    let mut scalar_cursor = Cursor::new(7_u64.to_le_bytes());
    let mut scalar_reader = CountingReader::new(&mut scalar_cursor);
    let scalar = read_raw_value(&mut scalar_reader, GgufValueType::Uint64).expect("scalar");

    assert_eq!(scalar, 7_u64.to_le_bytes());
    assert_eq!(scalar.capacity(), scalar.len());

    let mut string_bytes = Vec::with_capacity(11);
    string_bytes.extend_from_slice(&3_u64.to_le_bytes());
    string_bytes.extend_from_slice(b"abc");
    let mut string_cursor = Cursor::new(string_bytes);
    let mut string_reader = CountingReader::new(&mut string_cursor);
    let string = read_raw_value(&mut string_reader, GgufValueType::String).expect("string");

    assert_eq!(&string[..8], &3_u64.to_le_bytes());
    assert_eq!(&string[8..], b"abc");
    assert!(string.capacity() >= string.len());
}

#[test]
fn shard_kvs_filter_existing_split_metadata_and_append_current_values() {
    let metadata = GgufMetadata {
        version: 3,
        kvs: vec![
            string_kv("general.architecture", "llama"),
            u16_kv(SPLIT_NO_KEY, 7),
            u16_kv(SPLIT_COUNT_KEY, 8),
            i32_kv(SPLIT_TENSORS_COUNT_KEY, 9),
        ],
        tensors: vec![
            TensorInfo {
                name: "a".to_string(),
                dimensions: vec![1],
                tensor_type: 0,
                source_offset: 0,
                source_span: 16,
            },
            TensorInfo {
                name: "b".to_string(),
                dimensions: vec![1],
                tensor_type: 0,
                source_offset: 16,
                source_span: 16,
            },
        ],
        data_offset: 0,
        alignment: DEFAULT_ALIGNMENT,
    };

    let kvs = build_shard_kvs(&metadata, 0, 2).expect("kvs");

    assert_eq!(
        kvs.iter()
            .filter(|kv| kv.key == "general.architecture")
            .count(),
        1
    );
    assert_eq!(kvs.iter().filter(|kv| kv.key == SPLIT_NO_KEY).count(), 1);
    assert_eq!(
        read_u16_kv(kvs.iter().find(|kv| kv.key == SPLIT_NO_KEY).unwrap()),
        Some(0)
    );
    assert_eq!(
        read_u16_kv(kvs.iter().find(|kv| kv.key == SPLIT_COUNT_KEY).unwrap()),
        Some(2)
    );
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
        test_u64_from_usize(original.len(), "original length"),
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
        test_u64_from_usize(original.len(), "original length"),
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
    assign_source_spans(
        &mut parsed,
        test_u64_from_usize(bytes.len(), "bytes length"),
    )
    .expect("spans");
    parsed
}

fn string_kv(key: &str, value: &str) -> KvEntry {
    let mut raw_value = Vec::new();
    write_string(&mut raw_value, value).expect("string raw value");
    KvEntry {
        key: key.to_string(),
        value_type: GgufValueType::String,
        raw_value,
    }
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
    write_header_and_metadata(
        &mut metadata,
        test_u64_from_usize(tensors.len(), "tensor count"),
    );

    let mut tensor_data = Vec::new();
    let mut tensor_offsets = Vec::new();
    for (_, data) in &tensors {
        let next_offset = align_to(
            test_u64_from_usize(tensor_data.len(), "tensor data length"),
            DEFAULT_ALIGNMENT,
        )
        .unwrap();
        tensor_data.resize(test_usize_from_u64(next_offset, "next tensor offset"), 0);
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

    let data_offset = align_to(
        test_u64_from_usize(metadata.len(), "metadata length"),
        DEFAULT_ALIGNMENT,
    )
    .unwrap();
    metadata.resize(test_usize_from_u64(data_offset, "data offset"), 0);
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

fn test_u64_from_usize(value: usize, name: &str) -> u64 {
    u64::try_from(value).expect(name)
}

fn test_usize_from_u64(value: u64, name: &str) -> usize {
    usize_from_u64(value, name).expect(name)
}

struct MemoryReadAt {
    bytes: Vec<u8>,
}

impl GgufReadAt for MemoryReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
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
        let bytes = u64::try_from(writer.bytes.len())
            .map_err(|_| GgufError::Invalid("memory shard length exceeds u64".to_string()))?;
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
