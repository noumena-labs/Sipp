//! Browser model ingestion: stream files into asset storage via wasm-bindgen.

use std::ffi::{CStr, CString};
use std::io::{self, Write};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use cogentlm_gguf::{
    plan_gguf_split, split_gguf, split_gguf_file, BrowserCacheLayout, BrowserCachePolicy,
    GgufError, GgufReadAt, GgufShardSink, GgufSplitOptions,
};

const STATUS_OK: i32 = 0;
const STATUS_NULL_POINTER: i32 = -1;
const STATUS_INVALID_UTF8: i32 = -2;
const STATUS_SPLIT_FAILED: i32 = -3;
const STATUS_INVALID_CALLBACK: i32 = -4;

type ReadAtCallback = unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> i32;
type OpenShardCallback = unsafe extern "C" fn(*mut c_void, *const c_char, u16, u16) -> i32;
type WriteShardCallback = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> i32;
type CloseShardCallback = unsafe extern "C" fn(*mut c_void) -> i32;

#[no_mangle]
pub extern "C" fn cogentlm_browser_cache_layout(
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

#[no_mangle]
pub extern "C" fn cogentlm_gguf_split_file(
    input_path: *const c_char,
    output_prefix: *const c_char,
    shard_max_bytes: u64,
) -> i32 {
    ffi_status(|| {
        let Some(input_path) = read_c_string(input_path)? else {
            return Ok(STATUS_NULL_POINTER);
        };
        let Some(output_prefix) = read_c_string(output_prefix)? else {
            return Ok(STATUS_NULL_POINTER);
        };

        split_gguf_file(
            input_path,
            output_prefix,
            GgufSplitOptions { shard_max_bytes },
        )
        .map(|_| STATUS_OK)
        .map_err(|_| STATUS_SPLIT_FAILED)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_gguf_plan_split_count(
    source_bytes: u64,
    shard_max_bytes: u64,
    user_data: *mut c_void,
    read_at: Option<ReadAtCallback>,
) -> i32 {
    ffi_status(|| {
        let Some(read_at) = read_at else {
            return Ok(STATUS_INVALID_CALLBACK);
        };
        let mut source = CallbackReadAt { user_data, read_at };
        let manifest = plan_gguf_split(
            source_bytes,
            &mut source,
            "model",
            GgufSplitOptions { shard_max_bytes },
        )
        .map_err(|_| STATUS_SPLIT_FAILED)?;
        i32::try_from(manifest.shards.len()).map_err(|_| STATUS_SPLIT_FAILED)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_gguf_split_stream(
    source_bytes: u64,
    output_prefix: *const c_char,
    shard_max_bytes: u64,
    user_data: *mut c_void,
    read_at: Option<ReadAtCallback>,
    open_shard: Option<OpenShardCallback>,
    write_shard: Option<WriteShardCallback>,
    close_shard: Option<CloseShardCallback>,
) -> i32 {
    ffi_status(|| {
        let Some(output_prefix) = read_c_string(output_prefix)? else {
            return Ok(STATUS_NULL_POINTER);
        };
        let Some(read_at) = read_at else {
            return Ok(STATUS_INVALID_CALLBACK);
        };
        let Some(open_shard) = open_shard else {
            return Ok(STATUS_INVALID_CALLBACK);
        };
        let Some(write_shard) = write_shard else {
            return Ok(STATUS_INVALID_CALLBACK);
        };
        let Some(close_shard) = close_shard else {
            return Ok(STATUS_INVALID_CALLBACK);
        };

        let mut source = CallbackReadAt { user_data, read_at };
        let mut sink = CallbackShardSink {
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
        .map_err(|_| STATUS_SPLIT_FAILED)
    })
}

fn ffi_status(operation: impl FnOnce() -> Result<i32, i32>) -> i32 {
    catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(Ok(STATUS_SPLIT_FAILED))
        .unwrap_or_else(|status| status)
}

fn read_c_string(ptr: *const c_char) -> Result<Option<String>, i32> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: FFI callers must pass a valid, NUL-terminated C string pointer
    // for non-null path arguments. We only borrow it for the duration of this
    // conversion and immediately copy it into an owned Rust String.
    let value = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| STATUS_INVALID_UTF8)?
        .to_string();
    Ok(Some(value))
}

struct CallbackReadAt {
    user_data: *mut c_void,
    read_at: ReadAtCallback,
}

impl GgufReadAt for CallbackReadAt {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        // SAFETY: `read_at` is supplied by the embedding host with the declared
        // C ABI. `dst.as_mut_ptr()` is valid for `dst.len()` bytes for the
        // duration of this synchronous callback.
        let status = unsafe { (self.read_at)(self.user_data, offset, dst.as_mut_ptr(), dst.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(GgufError::Invalid(format!(
                "read_at callback failed with status {status}"
            )))
        }
    }
}

struct CallbackShardSink {
    user_data: *mut c_void,
    open_shard: OpenShardCallback,
    write_shard: WriteShardCallback,
    close_shard: CloseShardCallback,
}

impl GgufShardSink for CallbackShardSink {
    type Writer = CallbackShardWriter;

    fn create_shard(
        &mut self,
        path: &Path,
        index: u16,
        count: u16,
    ) -> Result<Self::Writer, GgufError> {
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| GgufError::Invalid("output shard path contains a NUL byte".to_string()))?;
        // SAFETY: `open_shard` is a host callback with the declared C ABI.
        // `path` is kept alive for the whole synchronous call, so `as_ptr()`
        // remains a valid NUL-terminated string while the callback runs.
        let status = unsafe { (self.open_shard)(self.user_data, path.as_ptr(), index, count) };
        if status != 0 {
            return Err(GgufError::Invalid(format!(
                "open_shard callback failed with status {status}"
            )));
        }
        Ok(CallbackShardWriter {
            user_data: self.user_data,
            write_shard: self.write_shard,
            bytes_written: 0,
        })
    }

    fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
        let bytes_written = writer.bytes_written;
        // SAFETY: `close_shard` is a synchronous host callback with the
        // declared C ABI and receives only the opaque host-owned user pointer.
        let status = unsafe { (self.close_shard)(self.user_data) };
        if status != 0 {
            return Err(GgufError::Invalid(format!(
                "close_shard callback failed with status {status}"
            )));
        }
        Ok(bytes_written)
    }
}

struct CallbackShardWriter {
    user_data: *mut c_void,
    write_shard: WriteShardCallback,
    bytes_written: u64,
}

impl Write for CallbackShardWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // SAFETY: `write_shard` is a synchronous host callback with the
        // declared C ABI. `buf.as_ptr()` is valid for `buf.len()` bytes until
        // the callback returns.
        let status = unsafe { (self.write_shard)(self.user_data, buf.as_ptr(), buf.len()) };
        if status != 0 {
            return Err(io::Error::other(format!(
                "write_shard callback failed with status {status}"
            )));
        }
        self.add_written_bytes(buf.len())?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        // SAFETY: Same callback contract as `write`; the host consumes the
        // borrowed buffer synchronously before this function returns.
        let status = unsafe { (self.write_shard)(self.user_data, buf.as_ptr(), buf.len()) };
        if status != 0 {
            return Err(io::Error::other(format!(
                "write_shard callback failed with status {status}"
            )));
        }
        self.add_written_bytes(buf.len())?;
        Ok(())
    }
}

impl CallbackShardWriter {
    fn add_written_bytes(&mut self, len: usize) -> io::Result<()> {
        let len = u64::try_from(len).map_err(|_| io::Error::other("shard byte count overflow"))?;
        self.bytes_written = self
            .bytes_written
            .checked_add(len)
            .ok_or_else(|| io::Error::other("shard byte count overflow"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
