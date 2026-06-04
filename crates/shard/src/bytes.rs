//! Byte-level GGUF I/O: scalar/string/array read helpers, counted readers and
//! writers, alignment math, and integer-width conversions.

use std::io::{self, Read, Write};

use super::{GgufError, GgufReadAt, GgufValueType};

const READ_AT_CURSOR_BUFFER_BYTES: usize = 64 * 1024;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "tests/bytes_tests.rs"]
mod bytes_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
pub(super) fn read_raw_value<R: Read>(
    reader: &mut CountingReader<'_, R>,
    value_type: GgufValueType,
) -> Result<Vec<u8>, GgufError> {
    let mut raw = Vec::with_capacity(value_type.scalar_size().unwrap_or(0));
    read_raw_value_into(reader, value_type, &mut raw)?;
    Ok(raw)
}

pub(super) fn read_raw_value_into<R: Read>(
    reader: &mut CountingReader<'_, R>,
    value_type: GgufValueType,
    raw: &mut Vec<u8>,
) -> Result<(), GgufError> {
    match value_type {
        GgufValueType::String => {
            let len = reader.read_u64()?;
            let len = usize_from_u64(len, "string length")?;
            reserve_raw_value_bytes(
                raw,
                8_usize
                    .checked_add(len)
                    .ok_or_else(|| GgufError::Invalid("raw string value too large".to_string()))?,
            )?;
            raw.extend_from_slice(&(u64_from_usize(len, "string length")?).to_le_bytes());
            let bytes = reader.read_vec(len)?;
            raw.extend_from_slice(&bytes);
        }
        GgufValueType::Array => {
            let item_type_raw = reader.read_u32()?;
            let item_type = GgufValueType::from_u32(item_type_raw)?;
            let len = reader.read_u64()?;
            let len = usize_from_u64(len, "array length")?;
            reserve_raw_value_bytes(raw, 12)?;
            raw.extend_from_slice(&item_type_raw.to_le_bytes());
            raw.extend_from_slice(&(u64_from_usize(len, "array length")?).to_le_bytes());
            for _ in 0..len {
                read_raw_value_into(reader, item_type, raw)?;
            }
        }
        _ => {
            let size = value_type
                .scalar_size()
                .ok_or_else(|| GgufError::Invalid("invalid scalar value type".to_string()))?;
            reserve_raw_value_bytes(raw, size)?;
            raw.extend_from_slice(&reader.read_vec(size)?);
        }
    }
    Ok(())
}

pub(super) fn reserve_raw_value_bytes(
    raw: &mut Vec<u8>,
    additional: usize,
) -> Result<(), GgufError> {
    raw.try_reserve(additional)
        .map_err(|_| GgufError::Invalid("raw value is too large".to_string()))
}

pub(super) fn copy_exact_from<S: GgufReadAt, W: Write>(
    source: &mut S,
    mut source_offset: u64,
    output: &mut W,
    mut bytes: u64,
    buffer: &mut [u8],
) -> Result<(), GgufError> {
    while bytes > 0 {
        let chunk = usize::try_from(bytes.min(u64_from_usize(buffer.len(), "copy buffer length")?))
            .map_err(|_| GgufError::Invalid("copy chunk too large".to_string()))?;
        source.read_at(source_offset, &mut buffer[..chunk])?;
        output.write_all(&buffer[..chunk])?;
        source_offset = source_offset
            .checked_add(u64_from_usize(chunk, "copy chunk length")?)
            .ok_or_else(|| GgufError::Invalid("copy offset overflow".to_string()))?;
        bytes -= u64_from_usize(chunk, "copy chunk length")?;
    }
    Ok(())
}

pub(super) fn write_zeros<W: Write>(writer: &mut W, bytes: u64) -> Result<(), GgufError> {
    const ZEROES: [u8; 64] = [0; 64];
    let mut remaining = bytes;
    while remaining > 0 {
        let chunk =
            usize::try_from(remaining.min(u64_from_usize(ZEROES.len(), "zero buffer length")?))
                .map_err(|_| GgufError::Invalid("padding too large".to_string()))?;
        writer.write_all(&ZEROES[..chunk])?;
        remaining -= u64_from_usize(chunk, "zero chunk length")?;
    }
    Ok(())
}

pub(super) fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<(), GgufError> {
    write_u64(writer, u64_from_usize(value.len(), "string length")?)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

pub(super) fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<(), GgufError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub(super) fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<(), GgufError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub(super) fn align_to(value: u64, alignment: u64) -> Result<u64, GgufError> {
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

pub(super) fn usize_from_u64(value: u64, name: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Invalid(format!("{name} does not fit usize")))
}

pub(super) fn usize_from_u32(value: u32, name: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Invalid(format!("{name} does not fit usize")))
}

pub(super) fn u64_from_usize(value: usize, name: &str) -> Result<u64, GgufError> {
    u64::try_from(value).map_err(|_| GgufError::Invalid(format!("{name} does not fit u64")))
}

pub(super) fn u32_from_usize(value: usize, name: &str) -> Result<u32, GgufError> {
    u32::try_from(value).map_err(|_| GgufError::Invalid(format!("{name} does not fit u32")))
}

pub(super) struct ReadAtCursor<'a, S> {
    source: &'a mut S,
    source_len: u64,
    position: u64,
    buffer: Vec<u8>,
    buffer_start: u64,
    buffer_len: usize,
}

impl<'a, S> ReadAtCursor<'a, S> {
    pub(super) fn new(source: &'a mut S, source_len: u64) -> Self {
        Self {
            source,
            source_len,
            position: 0,
            buffer: vec![0; READ_AT_CURSOR_BUFFER_BYTES],
            buffer_start: 0,
            buffer_len: 0,
        }
    }
}

impl<S: GgufReadAt> Read for ReadAtCursor<'_, S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        let mut written = 0usize;
        while written < buf.len() {
            if !self.buffer_contains(self.position) {
                self.fill_buffer()?;
                if self.buffer_len == 0 {
                    break;
                }
            }

            let buffer_offset = usize::try_from(self.position - self.buffer_start)
                .map_err(|_| io::Error::other("read buffer offset overflow"))?;
            let available = self.buffer_len.saturating_sub(buffer_offset);
            if available == 0 {
                self.fill_buffer()?;
                continue;
            }

            let chunk = available.min(buf.len() - written);
            buf[written..written + chunk]
                .copy_from_slice(&self.buffer[buffer_offset..buffer_offset + chunk]);
            written += chunk;
            self.position = self
                .position
                .checked_add(
                    u64::try_from(chunk).map_err(|_| io::Error::other("read length overflow"))?,
                )
                .ok_or_else(|| io::Error::other("read position overflow"))?;
        }
        Ok(written)
    }
}

impl<S: GgufReadAt> ReadAtCursor<'_, S> {
    fn buffer_contains(&self, position: u64) -> bool {
        let Some(buffer_end) = self.buffer_start.checked_add(self.buffer_len as u64) else {
            return false;
        };
        position >= self.buffer_start && position < buffer_end
    }

    fn fill_buffer(&mut self) -> io::Result<()> {
        self.buffer_start = self.position;
        if self.position >= self.source_len {
            self.buffer_len = 0;
            return Ok(());
        }

        let remaining = self.source_len - self.position;
        let read_len = match usize::try_from(remaining) {
            Ok(value) => value.min(self.buffer.len()),
            Err(_) => self.buffer.len(),
        };
        self.source
            .read_at(self.position, &mut self.buffer[..read_len])
            .map_err(|err| io::Error::other(err.to_string()))?;
        self.buffer_len = read_len;
        Ok(())
    }
}

pub(super) struct CountingWriter<W> {
    inner: W,
    position: u64,
}

impl<W> CountingWriter<W> {
    pub(super) fn new(inner: W) -> Self {
        Self { inner, position: 0 }
    }

    pub(super) fn position(&self) -> u64 {
        self.position
    }

    pub(super) fn into_inner(mut self) -> Result<W, GgufError>
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
            .checked_add(
                u64::try_from(written).map_err(|_| io::Error::other("write length overflow"))?,
            )
            .ok_or_else(|| io::Error::other("write position overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.write_all(buf)?;
        self.position = self
            .position
            .checked_add(
                u64::try_from(buf.len()).map_err(|_| io::Error::other("write length overflow"))?,
            )
            .ok_or_else(|| io::Error::other("write position overflow"))?;
        Ok(())
    }
}

pub(super) struct CountingReader<'a, R> {
    inner: &'a mut R,
    position: u64,
}

impl<'a, R: Read> CountingReader<'a, R> {
    pub(super) fn new(inner: &'a mut R) -> Self {
        Self { inner, position: 0 }
    }

    pub(super) fn position(&self) -> u64 {
        self.position
    }

    pub(super) fn read_exact_counted(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        self.inner.read_exact(buf)?;
        self.position = self
            .position
            .checked_add(
                u64::try_from(buf.len())
                    .map_err(|_| GgufError::Invalid("read length overflow".to_string()))?,
            )
            .ok_or_else(|| GgufError::Invalid("reader position overflow".to_string()))?;
        Ok(())
    }

    pub(super) fn read_vec(&mut self, len: usize) -> Result<Vec<u8>, GgufError> {
        let mut buf = vec![0u8; len];
        self.read_exact_counted(&mut buf)?;
        Ok(buf)
    }

    pub(super) fn skip_bytes(&mut self, len: usize) -> Result<(), GgufError> {
        const BUFFER_BYTES: usize = 8 * 1024;
        let mut remaining = len;
        let mut buffer = [0_u8; BUFFER_BYTES];
        while remaining > 0 {
            let chunk = remaining.min(BUFFER_BYTES);
            self.read_exact_counted(&mut buffer[..chunk])?;
            remaining -= chunk;
        }
        Ok(())
    }

    pub(super) fn read_u8(&mut self) -> Result<u8, GgufError> {
        let mut buf = [0u8; 1];
        self.read_exact_counted(&mut buf)?;
        Ok(buf[0])
    }

    pub(super) fn read_u32(&mut self) -> Result<u32, GgufError> {
        let mut buf = [0u8; 4];
        self.read_exact_counted(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub(super) fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut buf = [0u8; 8];
        self.read_exact_counted(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    pub(super) fn read_string(&mut self) -> Result<String, GgufError> {
        let len = usize_from_u64(self.read_u64()?, "string length")?;
        let bytes = self.read_vec(len)?;
        String::from_utf8(bytes)
            .map_err(|_| GgufError::Invalid("GGUF string is not UTF-8".to_string()))
    }
}
