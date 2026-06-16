//! Tests the `crate root` module in `sipp::shard`.
//!
//! Covers cache policy defaults, GGUF metadata parsing, split planning/writing,
//! and deterministic in-memory/file sink error paths without native model
//! execution.

use super::*;
use crate::shard::support::*;

use std::fs;
use std::io::{self, Cursor, Write};
use std::path::{Path, PathBuf};

#[test]
fn default_options_use_expected_cache_thresholds() {
    let policy = BrowserCachePolicy::default();
    let options = GgufSplitOptions::default();

    assert_eq!(policy.direct_load_max_bytes, DEFAULT_DIRECT_LOAD_MAX_BYTES);
    assert_eq!(policy.shard_max_bytes, DEFAULT_SHARD_MAX_BYTES);
    assert_eq!(options.shard_max_bytes, DEFAULT_SHARD_MAX_BYTES);
}

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
fn split_path_formats_one_based_padded_indices() {
    assert_eq!(
        split_path("cache/model", 0, 12),
        PathBuf::from("cache/model-00001-of-00012.gguf")
    );
    assert_eq!(
        split_path(PathBuf::from("model"), 9, 12),
        PathBuf::from("model-00010-of-00012.gguf")
    );
}

#[test]
fn gguf_value_type_round_trips_wire_codes_and_scalar_sizes() {
    for (value_type, raw, size) in [
        (GgufValueType::Uint8, 0, Some(1)),
        (GgufValueType::Int8, 1, Some(1)),
        (GgufValueType::Uint16, 2, Some(2)),
        (GgufValueType::Int16, 3, Some(2)),
        (GgufValueType::Uint32, 4, Some(4)),
        (GgufValueType::Int32, 5, Some(4)),
        (GgufValueType::Float32, 6, Some(4)),
        (GgufValueType::Bool, 7, Some(1)),
        (GgufValueType::String, 8, None),
        (GgufValueType::Array, 9, None),
        (GgufValueType::Uint64, 10, Some(8)),
        (GgufValueType::Int64, 11, Some(8)),
        (GgufValueType::Float64, 12, Some(8)),
    ] {
        assert_eq!(value_type as u32, raw);
        assert_eq!(
            GgufValueType::from_u32(raw).expect("known type"),
            value_type
        );
        assert_eq!(value_type.scalar_size(), size);
    }

    assert!(matches!(
        GgufValueType::from_u32(99),
        Err(GgufError::Invalid(message)) if message == "unknown value type 99"
    ));
}

#[test]
fn read_raw_value_preserves_scalar_string_and_array_bytes() {
    let mut scalar_cursor = Cursor::new(7_u64.to_le_bytes());
    let mut scalar_reader = CountingReader::new(&mut scalar_cursor);
    let scalar = read_raw_value(&mut scalar_reader, GgufValueType::Uint64).expect("scalar");
    assert_eq!(scalar, 7_u64.to_le_bytes());
    assert_eq!(scalar.capacity(), scalar.len());

    let mut string_bytes = Vec::new();
    write_string(&mut string_bytes, "abc").expect("string input");
    let mut string_cursor = Cursor::new(string_bytes);
    let mut string_reader = CountingReader::new(&mut string_cursor);
    let string = read_raw_value(&mut string_reader, GgufValueType::String).expect("string");
    assert_eq!(&string[..8], &3_u64.to_le_bytes());
    assert_eq!(&string[8..], b"abc");
    assert!(string.capacity() >= string.len());

    let mut array_bytes = Vec::new();
    write_u32(&mut array_bytes, GgufValueType::Uint16 as u32).expect("array type");
    write_u64(&mut array_bytes, 2).expect("array length");
    array_bytes.extend_from_slice(&10_u16.to_le_bytes());
    array_bytes.extend_from_slice(&11_u16.to_le_bytes());
    let mut array_cursor = Cursor::new(array_bytes);
    let mut array_reader = CountingReader::new(&mut array_cursor);
    let array = read_raw_value(&mut array_reader, GgufValueType::Array).expect("array");
    assert_eq!(&array[..4], &(GgufValueType::Uint16 as u32).to_le_bytes());
    assert_eq!(&array[4..12], &2_u64.to_le_bytes());
    assert_eq!(&array[12..14], &10_u16.to_le_bytes());
    assert_eq!(&array[14..16], &11_u16.to_le_bytes());
}

#[test]
fn parse_metadata_rejects_invalid_magic_version_alignment_and_dimensions() {
    let mut invalid_magic = Cursor::new(0_u32.to_le_bytes());
    assert!(matches!(
        parse_metadata(&mut invalid_magic),
        Err(GgufError::Invalid(message)) if message == "missing GGUF magic"
    ));

    let mut unsupported = Cursor::new(metadata_gguf_version(99, &[]));
    assert!(matches!(
        parse_metadata(&mut unsupported),
        Err(GgufError::UnsupportedVersion(99))
    ));

    let invalid_alignment = metadata_gguf(&[(GENERAL_ALIGNMENT_KEY, MetadataValue::Uint32(3))]);
    let mut invalid_alignment = Cursor::new(invalid_alignment);
    assert!(matches!(
        parse_metadata(&mut invalid_alignment),
        Err(GgufError::Invalid(message)) if message == "invalid GGUF alignment 3"
    ));

    for dimension_count in [0_u32, 17] {
        let mut bytes = Vec::new();
        write_u32(&mut bytes, GGUF_MAGIC).expect("magic");
        write_u32(&mut bytes, 3).expect("version");
        write_u64(&mut bytes, 1).expect("tensor count");
        write_u64(&mut bytes, 0).expect("kv count");
        write_string(&mut bytes, "bad.weight").expect("tensor name");
        write_u32(&mut bytes, dimension_count).expect("dimension count");
        let mut cursor = Cursor::new(bytes);
        assert!(matches!(
            parse_metadata(&mut cursor),
            Err(GgufError::Invalid(message))
                if message == format!("invalid tensor dimension count {dimension_count}")
        ));
    }
}

#[test]
fn parse_metadata_reads_custom_alignment_and_tensor_fields() {
    let bytes = gguf_with_tensors(
        &[("general.architecture", MetadataValue::String("llama"))],
        &[FakeTensor::new("tok_embeddings.weight", vec![1u8; 8]).with_dimensions(vec![2, 4])],
        64,
    );
    let mut cursor = Cursor::new(&bytes);
    let parsed = parse_metadata(&mut cursor).expect("metadata");

    assert_eq!(parsed.version, 3);
    assert_eq!(parsed.alignment, 64);
    assert_eq!(parsed.tensors[0].name, "tok_embeddings.weight");
    assert_eq!(parsed.tensors[0].dimensions, vec![2, 4]);
    assert_eq!(parsed.data_offset % 64, 0);
}

#[test]
fn monolithic_check_rejects_already_split_sources() {
    let metadata = GgufMetadata {
        version: 3,
        kvs: vec![u16_kv(SPLIT_COUNT_KEY, 2)],
        tensors: Vec::new(),
        data_offset: 0,
        alignment: DEFAULT_ALIGNMENT,
    };

    assert!(matches!(
        ensure_monolithic(&metadata),
        Err(GgufError::AlreadySplit(2))
    ));

    let single = GgufMetadata {
        kvs: vec![u16_kv(SPLIT_COUNT_KEY, 1)],
        ..metadata
    };
    ensure_monolithic(&single).expect("single split marker is monolithic");
}

#[test]
fn assign_source_spans_rejects_invalid_file_and_tensor_offsets() {
    let mut before_data = GgufMetadata {
        version: 3,
        kvs: Vec::new(),
        tensors: Vec::new(),
        data_offset: 32,
        alignment: DEFAULT_ALIGNMENT,
    };
    assert!(matches!(
        assign_source_spans(&mut before_data, 31),
        Err(GgufError::Invalid(message)) if message == "file ends before tensor data"
    ));

    let mut duplicate_offsets = GgufMetadata {
        tensors: vec![tensor("a", 0, 0), tensor("b", 0, 0)],
        data_offset: 0,
        ..before_data.clone()
    };
    assert!(matches!(
        assign_source_spans(&mut duplicate_offsets, 2),
        Err(GgufError::Invalid(message))
            if message == "tensor offsets are not strictly increasing"
    ));

    let mut offset_overflow = GgufMetadata {
        tensors: vec![tensor("last", 1, 0)],
        data_offset: u64::MAX,
        ..before_data.clone()
    };
    assert!(matches!(
        assign_source_spans(&mut offset_overflow, u64::MAX),
        Err(GgufError::Invalid(message)) if message == "tensor offset overflow"
    ));

    let mut after_file = GgufMetadata {
        tensors: vec![tensor("last", 8, 0)],
        data_offset: 16,
        ..before_data
    };
    assert!(matches!(
        assign_source_spans(&mut after_file, 23),
        Err(GgufError::Invalid(message)) if message == "last tensor starts after end of file"
    ));

    let mut no_tensors = GgufMetadata {
        version: 3,
        kvs: Vec::new(),
        tensors: Vec::new(),
        data_offset: 16,
        alignment: DEFAULT_ALIGNMENT,
    };
    assign_source_spans(&mut no_tensors, 16).expect("empty tensor metadata");
}

#[test]
fn plan_shards_covers_empty_zero_span_split_and_overflow_cases() {
    assert!(matches!(
        plan_shards(&[], 1),
        Err(GgufError::Invalid(message)) if message == "GGUF contains no tensors"
    ));

    assert!(matches!(
        plan_shards(&[tensor("a", 0, 1)], 0),
        Err(GgufError::Invalid(message)) if message == "shard_max_bytes must be positive"
    ));

    assert!(matches!(
        plan_shards(&[tensor("empty", 0, 0)], 1),
        Err(GgufError::Invalid(message)) if message == "tensor 'empty' has no source bytes"
    ));

    let plans = plan_shards(
        &[tensor("a", 0, 4), tensor("b", 4, 4), tensor("c", 8, 8)],
        8,
    )
    .expect("plans");
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].tensors, vec![0, 1]);
    assert_eq!(plans[1].source_spans, 8);

    assert!(matches!(
        plan_shards(&[tensor("huge", 0, u64::MAX), tensor("next", 1, 1)], u64::MAX),
        Err(GgufError::Invalid(message)) if message == "shard size overflow"
    ));
}

#[test]
fn split_file_writes_llama_compatible_split_shards_and_creates_parent_dirs() {
    let root = unique_temp_dir();
    let input_dir = root.join("input");
    let output_prefix = root.join("nested").join("model");
    fs::create_dir_all(&input_dir).expect("temp dir");
    let source = input_dir.join("model.gguf");
    let original = split_fixture_gguf();
    fs::write(&source, &original).expect("write source");

    let manifest = split_gguf_file(
        &source,
        &output_prefix,
        GgufSplitOptions {
            shard_max_bytes: 128,
        },
    )
    .expect("split");

    assert_eq!(
        manifest.source_bytes,
        u64::try_from(original.len()).unwrap()
    );
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
    assert!(manifest.shards[0].bytes > 0);

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
fn split_file_missing_input_is_io_error() {
    let error = split_gguf_file("missing-model.gguf", "unused", GgufSplitOptions::default())
        .expect_err("missing input");

    assert!(matches!(error, GgufError::Io(_)));
}

#[test]
fn file_shard_sink_creates_parent_directory_and_reports_written_bytes() {
    assert_eq!(shard_parent_dir(Path::new("shard.gguf")), None);

    let relative = PathBuf::from(format!("sipp-shard-rootless-{}.gguf", std::process::id()));
    fs::remove_file(&relative).ok();
    let mut sink = FileShardSink;
    let writer = sink
        .create_shard(&relative, 0, 1)
        .expect("create rootless shard");
    assert_eq!(sink.finish_shard(writer).expect("finish rootless shard"), 0);
    fs::remove_file(&relative).ok();

    let root = unique_temp_dir();
    let path = root.join("nested").join("model-00001-of-00001.gguf");
    assert_eq!(shard_parent_dir(&path), path.parent());

    let mut sink = FileShardSink;
    let mut writer = sink.create_shard(&path, 0, 1).expect("create shard");
    writer.write_all(b"abc").expect("write shard");

    let bytes = sink.finish_shard(writer).expect("finish shard");

    assert_eq!(bytes, 3);
    assert_eq!(fs::read(&path).expect("written bytes"), b"abc");
    fs::remove_dir_all(root).ok();
}

#[test]
fn splits_through_read_at_and_custom_sink() {
    let original = split_fixture_gguf();
    let mut source = MemoryReadAt::new(original.clone());
    let mut sink = MemoryShardSink::new();

    let manifest = split_gguf(
        u64::try_from(original.len()).expect("original length"),
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
    let original = split_fixture_gguf();
    let mut source = MemoryReadAt::new(original.clone());
    let manifest = plan_gguf_split(
        u64::try_from(original.len()).expect("original length"),
        &mut source,
        "model",
        GgufSplitOptions {
            shard_max_bytes: 128,
        },
    )
    .expect("plan");

    assert_eq!(manifest.shards.len(), 2);
    assert_eq!(manifest.shards[0].bytes, 128);
    assert_eq!(manifest.shards[1].bytes, 64);
    assert_eq!(
        manifest.shards[0].path,
        PathBuf::from("model-00001-of-00002.gguf")
    );
    assert_eq!(
        manifest.shards[1].path,
        PathBuf::from("model-00002-of-00002.gguf")
    );
}

#[test]
fn split_propagates_read_create_finish_and_writer_errors() {
    let original = split_fixture_gguf();
    let source_bytes = u64::try_from(original.len()).expect("source length");

    let mut source = MemoryReadAt::new(original.clone());
    let mut sink = MemoryShardSink::fail_create();
    assert!(matches!(
        split_gguf(
            source_bytes,
            &mut source,
            "model",
            GgufSplitOptions { shard_max_bytes: 128 },
            &mut sink,
        ),
        Err(GgufError::Invalid(message)) if message == "planned create failure"
    ));

    let mut source = MemoryReadAt::new(original.clone());
    let mut sink = MemoryShardSink::fail_finish();
    assert!(matches!(
        split_gguf(
            source_bytes,
            &mut source,
            "model",
            GgufSplitOptions { shard_max_bytes: 128 },
            &mut sink,
        ),
        Err(GgufError::Invalid(message)) if message == "planned finish failure"
    ));

    let mut metadata_cursor = Cursor::new(&original);
    let metadata = parse_metadata(&mut metadata_cursor).expect("metadata");
    let first_source_position = metadata.data_offset;
    let mut source = MemoryReadAt::new(original).with_fail_at(first_source_position);
    let mut sink = MemoryShardSink::new();
    assert!(matches!(
        split_gguf(
            source_bytes,
            &mut source,
            "model",
            GgufSplitOptions { shard_max_bytes: 128 },
            &mut sink,
        ),
        Err(GgufError::Invalid(message)) if message == "planned read failure"
    ));
}

#[test]
fn prepare_split_rejects_too_many_shards() {
    let tensors = (0..=u16::MAX)
        .map(|idx| FakeTensor::new(&format!("tensor.{idx}"), vec![1]))
        .collect::<Vec<_>>();
    let bytes = gguf_with_tensors(&[], &tensors, DEFAULT_ALIGNMENT);
    let mut source = MemoryReadAt::new(bytes.clone());

    assert!(matches!(
        prepare_split(
            u64::try_from(bytes.len()).expect("source length"),
            &mut source,
            1,
        ),
        Err(GgufError::Invalid(message)) if message == "too many GGUF shards"
    ));
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
        tensors: vec![tensor("a", 0, 16), tensor("b", 16, 16)],
        data_offset: 0,
        alignment: DEFAULT_ALIGNMENT,
    };

    let first_kvs = build_shard_kvs(&metadata, 0, 2).expect("first kvs");
    assert_eq!(
        first_kvs
            .iter()
            .filter(|kv| kv.key == "general.architecture")
            .count(),
        1
    );
    assert_eq!(
        read_u16_kv(first_kvs.iter().find(|kv| kv.key == SPLIT_NO_KEY).unwrap()),
        Some(0)
    );
    assert_eq!(
        read_u16_kv(
            first_kvs
                .iter()
                .find(|kv| kv.key == SPLIT_COUNT_KEY)
                .unwrap()
        ),
        Some(2)
    );

    let second_kvs = build_shard_kvs(&metadata, 1, 2).expect("second kvs");
    assert!(!second_kvs.iter().any(|kv| kv.key == "general.architecture"));
    assert_eq!(
        read_u16_kv(second_kvs.iter().find(|kv| kv.key == SPLIT_NO_KEY).unwrap()),
        Some(1)
    );
}

#[test]
fn kv_read_helpers_reject_mismatched_types_and_lengths() {
    assert_eq!(read_u16_kv(&i32_kv("bad", 1)), None);
    assert_eq!(
        read_u16_kv(&KvEntry {
            key: "short".to_string(),
            value_type: GgufValueType::Uint16,
            raw_value: vec![1],
        }),
        None
    );

    assert_eq!(
        read_alignment(&KvEntry {
            key: GENERAL_ALIGNMENT_KEY.to_string(),
            value_type: GgufValueType::Uint32,
            raw_value: 64_u32.to_le_bytes().to_vec(),
        }),
        Some(64)
    );
    assert_eq!(
        read_alignment(&KvEntry {
            key: GENERAL_ALIGNMENT_KEY.to_string(),
            value_type: GgufValueType::Uint64,
            raw_value: 128_u64.to_le_bytes().to_vec(),
        }),
        Some(128)
    );
    assert_eq!(
        read_alignment(&string_kv(GENERAL_ALIGNMENT_KEY, "bad")),
        None
    );
}

#[test]
fn write_shard_rejects_internal_offset_overflows() {
    let metadata = GgufMetadata {
        version: 3,
        kvs: Vec::new(),
        tensors: vec![tensor("huge", 0, u64::MAX), tensor("next", 0, 1)],
        data_offset: 0,
        alignment: DEFAULT_ALIGNMENT,
    };
    let plan = ShardPlan {
        tensors: vec![0, 1],
        source_spans: u64::MAX,
    };
    let mut source = MemoryReadAt::new(Vec::new());
    let mut writer = CountingWriter::new(Vec::new());

    assert!(matches!(
        write_shard(&mut source, &metadata, &plan, 0, 1, &mut writer),
        Err(GgufError::Invalid(message)) if message == "shard tensor offset overflow"
    ));

    let metadata = GgufMetadata {
        version: 3,
        kvs: Vec::new(),
        tensors: vec![tensor("overflow", 1, 1)],
        data_offset: u64::MAX,
        alignment: DEFAULT_ALIGNMENT,
    };
    let plan = ShardPlan {
        tensors: vec![0],
        source_spans: 1,
    };
    let mut source = MemoryReadAt::new(Vec::new());
    let mut writer = CountingWriter::new(Vec::new());

    assert!(matches!(
        write_shard(&mut source, &metadata, &plan, 0, 1, &mut writer),
        Err(GgufError::Invalid(message)) if message == "source tensor offset overflow"
    ));
}

#[test]
fn private_shard_write_guards_cover_success_and_error_paths() {
    assert_eq!(shard_tensor_count_value(2).expect("tensor count"), 2);
    assert_eq!(shard_kv_count_value(3).expect("kv count"), 3);
    assert_eq!(tensor_dimension_count_value(4).expect("dimension count"), 4);
    assert!(matches!(
        tensor_dimension_count_value(usize::MAX),
        Err(GgufError::Invalid(message)) if message == "tensor dimension count does not fit u32"
    ));

    assert_eq!(shard_write_position(32, 8).expect("position"), 40);
    assert!(matches!(
        shard_write_position(u64::MAX, 1),
        Err(GgufError::Invalid(message)) if message == "shard write offset overflow"
    ));
    ensure_shard_position(40, 40).expect("matching position");
    assert!(matches!(
        ensure_shard_position(39, 40),
        Err(GgufError::Invalid(message)) if message == "internal shard offset mismatch"
    ));
}

#[test]
fn counting_writer_into_inner_flushes_and_preserves_inner_writer_errors() {
    let writer = CountingWriter::new(FlushErrorWriter);
    assert!(matches!(
        writer.into_inner(),
        Err(GgufError::Io(error)) if error.kind() == io::ErrorKind::Other
    ));
}

struct FlushErrorWriter;

impl io::Write for FlushErrorWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("planned flush failure"))
    }
}
