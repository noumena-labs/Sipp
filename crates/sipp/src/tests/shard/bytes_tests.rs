//! Tests the `bytes` module in `sipp::shard`.
//!
//! Covers byte-level GGUF value preservation, counted reader/writer behavior,
//! alignment math, read-at cursors, and deterministic copy/padding error paths
//! using fake handles only.

use super::*;
use crate::shard::{GgufError, GgufReadAt, GgufValueType};

use std::io::{self, Cursor, Read, Write};

#[test]
fn raw_values_preserve_scalars_strings_and_nested_array_payloads() {
    let mut string_array = Vec::new();
    write_u32(&mut string_array, GgufValueType::String as u32).expect("array item type");
    write_u64(&mut string_array, 2).expect("array length");
    write_string(&mut string_array, "a").expect("first string");
    write_string(&mut string_array, "bc").expect("second string");
    let mut cursor = Cursor::new(string_array);
    let mut reader = CountingReader::new(&mut cursor);

    let raw = read_raw_value(&mut reader, GgufValueType::Array).expect("array raw value");

    assert_eq!(&raw[..4], &(GgufValueType::String as u32).to_le_bytes());
    assert_eq!(&raw[4..12], &2_u64.to_le_bytes());
    assert_eq!(&raw[12..20], &1_u64.to_le_bytes());
    assert_eq!(raw[20], b'a');
    assert_eq!(&raw[21..29], &2_u64.to_le_bytes());
    assert_eq!(&raw[29..31], b"bc");
}

#[test]
fn raw_value_errors_on_impossible_string_growth_and_unknown_array_type() {
    let mut string = Cursor::new(u64::MAX.to_le_bytes());
    let mut reader = CountingReader::new(&mut string);
    assert!(matches!(
        read_raw_value(&mut reader, GgufValueType::String),
        Err(GgufError::Invalid(message)) if message == "raw string value too large"
    ));

    let reservable_but_too_large = (usize::MAX - 8) as u64;
    let mut string = Cursor::new(reservable_but_too_large.to_le_bytes());
    let mut reader = CountingReader::new(&mut string);
    assert!(matches!(
        read_raw_value(&mut reader, GgufValueType::String),
        Err(GgufError::Invalid(message)) if message == "raw value is too large"
    ));

    let mut array = Vec::new();
    write_u32(&mut array, 99).expect("bad item type");
    write_u64(&mut array, 0).expect("array length");
    let mut cursor = Cursor::new(array);
    let mut reader = CountingReader::new(&mut cursor);
    assert!(matches!(
        read_raw_value(&mut reader, GgufValueType::Array),
        Err(GgufError::Invalid(message)) if message == "unknown value type 99"
    ));

    let mut raw = Vec::new();
    assert!(matches!(
        reserve_raw_value_bytes(&mut raw, usize::MAX),
        Err(GgufError::Invalid(message)) if message == "raw value is too large"
    ));
}

#[test]
fn alignment_handles_exact_padding_zero_alignment_and_overflow() {
    assert_eq!(align_to(64, 32).expect("aligned"), 64);
    assert_eq!(align_to(65, 32).expect("padded"), 96);
    assert!(matches!(
        align_to(1, 0),
        Err(GgufError::Invalid(message)) if message == "zero alignment"
    ));
    assert!(matches!(
        align_to(u64::MAX - 1, 4),
        Err(GgufError::Invalid(message)) if message == "alignment overflow"
    ));
}

#[test]
fn read_at_cursor_reads_sequentially_and_maps_source_errors() {
    let mut source = StaticReadAt {
        bytes: b"abcdef".to_vec(),
        fail_at: None,
    };
    let len = source.bytes.len() as u64;
    let mut cursor = ReadAtCursor::new(&mut source, len);
    let mut empty = [];
    assert_eq!(cursor.read(&mut empty).expect("empty read"), 0);

    let mut first = [0u8; 2];
    cursor.read_exact(&mut first).expect("first read");
    assert_eq!(&first, b"ab");

    let mut second = [0u8; 3];
    cursor.read_exact(&mut second).expect("second read");
    assert_eq!(&second, b"cde");
}

#[test]
fn copy_exact_from_handles_zero_chunking_source_error_and_offset_overflow() {
    let mut source = StaticReadAt {
        bytes: b"abcdefgh".to_vec(),
        fail_at: None,
    };
    let mut output = Vec::new();
    let mut buffer = [0u8; 3];
    copy_exact_from(&mut source, 1, &mut output, 7, &mut buffer).expect("copy");
    assert_eq!(output, b"bcdefgh");

    copy_exact_from(&mut source, 0, &mut output, 0, &mut buffer).expect("zero copy");

    let mut source = StaticReadAt {
        bytes: b"abc".to_vec(),
        fail_at: Some(0),
    };
    let error =
        copy_exact_from(&mut source, 0, &mut Vec::new(), 1, &mut buffer).expect_err("source error");
    assert!(matches!(
        error,
        GgufError::Invalid(message) if message == "planned read failure"
    ));

    let mut source = FillReadAt;
    let mut one_byte = [0u8; 1];
    let error = copy_exact_from(&mut source, u64::MAX, &mut Vec::new(), 1, &mut one_byte)
        .expect_err("offset overflow");
    assert!(matches!(
        error,
        GgufError::Invalid(message) if message == "copy offset overflow"
    ));
}

#[test]
fn write_zeros_writes_in_chunks() {
    let mut bytes = Vec::new();
    write_zeros(&mut bytes, 130).expect("zeroes");

    assert_eq!(bytes.len(), 130);
    assert!(bytes.iter().all(|byte| *byte == 0));
}

#[test]
fn write_helpers_emit_little_endian_values() {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, 0x0102_0304).expect("u32");
    write_u64(&mut bytes, 0x0102_0304_0506_0708).expect("u64");
    write_string(&mut bytes, "xy").expect("string");

    assert_eq!(&bytes[..4], &[4, 3, 2, 1]);
    assert_eq!(&bytes[4..12], &[8, 7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(&bytes[12..20], &2_u64.to_le_bytes());
    assert_eq!(&bytes[20..], b"xy");
}

#[test]
fn counting_writer_tracks_write_write_all_flush_and_into_inner() {
    let mut writer = CountingWriter::new(ShortWriter::default());
    assert_eq!(writer.write(b"abcd").expect("write"), 2);
    assert_eq!(writer.position(), 2);
    writer.write_all(b"ef").expect("write all");
    assert_eq!(writer.position(), 4);
    writer.flush().expect("flush");

    let inner = writer.into_inner().expect("inner");
    assert_eq!(inner.bytes, b"abef");
}

#[test]
fn counting_reader_reads_skips_and_reports_invalid_strings() {
    let mut bytes = Vec::new();
    bytes.push(7);
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&11_u64.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0xaa, 9 * 1024));
    write_string(&mut bytes, "ok").expect("valid string");
    write_u64(&mut bytes, 1).expect("invalid length");
    bytes.push(0xff);
    let mut cursor = Cursor::new(bytes);
    let mut reader = CountingReader::new(&mut cursor);

    assert_eq!(reader.read_u8().expect("u8"), 7);
    assert_eq!(reader.read_u32().expect("u32"), 9);
    assert_eq!(reader.read_u64().expect("u64"), 11);
    reader.skip_bytes(9 * 1024).expect("skip");
    assert_eq!(reader.read_string().expect("string"), "ok");
    assert!(matches!(
        reader.read_string(),
        Err(GgufError::Invalid(message)) if message == "GGUF string is not UTF-8"
    ));
}

#[test]
fn integer_width_conversions_cover_success_and_failure_paths() {
    assert_eq!(usize_from_u64(7, "value").expect("usize"), 7);
    assert_eq!(usize_from_u32(7, "value").expect("usize"), 7);
    assert_eq!(u64_from_usize(7, "value").expect("u64"), 7);
    assert_eq!(u32_from_usize(7, "value").expect("u32"), 7);

    if usize::BITS < u64::BITS {
        assert!(matches!(
            usize_from_u64(u64::MAX, "value"),
            Err(GgufError::Invalid(message)) if message == "value does not fit usize"
        ));
    }

    assert!(matches!(
        u32_from_usize(usize::MAX, "value"),
        Err(GgufError::Invalid(message)) if message == "value does not fit u32"
    ));
}

struct StaticReadAt {
    bytes: Vec<u8>,
    fail_at: Option<u64>,
}

impl GgufReadAt for StaticReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        if self.fail_at == Some(offset) {
            return Err(GgufError::Invalid("planned read failure".to_string()));
        }
        let offset = usize::try_from(offset)
            .map_err(|_| GgufError::Invalid("read offset too large".to_string()))?;
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| GgufError::Invalid("read offset overflow".to_string()))?;
        let Some(bytes) = self.bytes.get(offset..end) else {
            return Err(GgufError::Invalid("read exceeds source".to_string()));
        };
        dst.copy_from_slice(bytes);
        Ok(())
    }
}

struct FillReadAt;

impl GgufReadAt for FillReadAt {
    fn read_at(&mut self, _offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        dst.fill(0xaa);
        Ok(())
    }
}

#[derive(Default)]
struct ShortWriter {
    bytes: Vec<u8>,
}

impl Write for ShortWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = buf.len().min(2);
        self.bytes.extend_from_slice(&buf[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.bytes.extend_from_slice(buf);
        Ok(())
    }
}
