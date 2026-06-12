//! Browser model ingestion: stream files into asset storage from WebAssembly.

use std::ffi::CString;
use std::io::{self, Write};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;

use sipp::shard::{
    plan_gguf_split, split_gguf, BrowserCacheLayout, BrowserCachePolicy, GgufError, GgufReadAt,
    GgufShardSink, GgufSplitOptions,
};

const STATUS_OK: i32 = 0;
const STATUS_SPLIT_FAILED: i32 = -3;

pub(crate) type GgufReadAtCallback = unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> i32;
pub(crate) type GgufOpenShardCallback =
    unsafe extern "C" fn(*mut c_void, *const c_char, u16, u16) -> i32;
pub(crate) type GgufWriteShardCallback = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> i32;
pub(crate) type GgufCloseShardCallback = unsafe extern "C" fn(*mut c_void) -> i32;

pub(crate) fn browser_cache_layout(
    source_bytes: u64,
    source_bytes_known: bool,
    direct_load_max_bytes: u64,
    shard_max_bytes: u64,
) -> i32 {
    let policy = BrowserCachePolicy {
        direct_load_max_bytes,
        shard_max_bytes,
    };
    match policy.resolve_layout(source_bytes_known.then_some(source_bytes)) {
        BrowserCacheLayout::SingleFile => 0,
        BrowserCacheLayout::SplitGguf => 1,
    }
}

pub(crate) fn gguf_plan_split_count(
    source_bytes: u64,
    shard_max_bytes: u64,
    user_data: *mut c_void,
    read_at: GgufReadAtCallback,
) -> i32 {
    let mut source = RawReadAt { user_data, read_at };
    plan_gguf_split(
        source_bytes,
        &mut source,
        "model",
        GgufSplitOptions { shard_max_bytes },
    )
    .ok()
    .and_then(|manifest| i32::try_from(manifest.shards.len()).ok())
    .unwrap_or(STATUS_SPLIT_FAILED)
}

pub(crate) fn gguf_split_stream(
    source_bytes: u64,
    output_prefix: &str,
    shard_max_bytes: u64,
    user_data: *mut c_void,
    read_at: GgufReadAtCallback,
    open_shard: GgufOpenShardCallback,
    write_shard: GgufWriteShardCallback,
    close_shard: GgufCloseShardCallback,
) -> i32 {
    let mut source = RawReadAt { user_data, read_at };
    let mut sink = RawShardSink {
        user_data,
        open_shard,
        write_shard,
        close_shard,
    };
    split_gguf(
        source_bytes,
        &mut source,
        output_prefix,
        GgufSplitOptions { shard_max_bytes },
        &mut sink,
    )
    .map(|_| STATUS_OK)
    .unwrap_or(STATUS_SPLIT_FAILED)
}

struct RawReadAt {
    user_data: *mut c_void,
    read_at: GgufReadAtCallback,
}

impl GgufReadAt for RawReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        let ptr = if dst.is_empty() {
            ptr::null_mut()
        } else {
            dst.as_mut_ptr()
        };
        let status = unsafe { (self.read_at)(self.user_data, offset, ptr, dst.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(GgufError::Invalid(format!(
                "read_at callback failed with status {status}"
            )))
        }
    }
}

struct RawShardSink {
    user_data: *mut c_void,
    open_shard: GgufOpenShardCallback,
    write_shard: GgufWriteShardCallback,
    close_shard: GgufCloseShardCallback,
}

impl GgufShardSink for RawShardSink {
    type Writer = RawShardWriter;

    fn create_shard(
        &mut self,
        path: &Path,
        index: u16,
        count: u16,
    ) -> Result<Self::Writer, GgufError> {
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| GgufError::Invalid("shard path contains an interior NUL".to_string()))?;
        let status = unsafe { (self.open_shard)(self.user_data, path.as_ptr(), index, count) };
        if status != 0 {
            return Err(GgufError::Invalid(format!(
                "open_shard callback failed with status {status}"
            )));
        }
        Ok(RawShardWriter {
            user_data: self.user_data,
            write_shard: self.write_shard,
            close_shard: self.close_shard,
            bytes_written: 0,
        })
    }

    fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
        let bytes_written = writer.bytes_written;
        let status = unsafe { (writer.close_shard)(writer.user_data) };
        if status != 0 {
            return Err(GgufError::Invalid(format!(
                "close_shard callback failed with status {status}"
            )));
        }
        Ok(bytes_written)
    }
}

struct RawShardWriter {
    user_data: *mut c_void,
    write_shard: GgufWriteShardCallback,
    close_shard: GgufCloseShardCallback,
    bytes_written: u64,
}

impl Write for RawShardWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let ptr = if buf.is_empty() {
            ptr::null()
        } else {
            buf.as_ptr()
        };
        let status = unsafe { (self.write_shard)(self.user_data, ptr, buf.len()) };
        if status != 0 {
            return Err(io::Error::other(format!(
                "write_shard callback failed with status {status}"
            )));
        }
        self.bytes_written += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let ptr = if buf.is_empty() {
            ptr::null()
        } else {
            buf.as_ptr()
        };
        let status = unsafe { (self.write_shard)(self.user_data, ptr, buf.len()) };
        if status != 0 {
            return Err(io::Error::other(format!(
                "write_shard callback failed with status {status}"
            )));
        }
        self.bytes_written += buf.len() as u64;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/ingest_tests.rs"]
mod ingest_tests;
