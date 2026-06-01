//! Browser model ingestion: stream files into asset storage from WebAssembly.

use std::io::{self, Write};
use std::path::Path;
use std::pin::Pin;

use cogentlm_shard::{
    plan_gguf_split, split_gguf, BrowserCacheLayout, BrowserCachePolicy, GgufError, GgufReadAt,
    GgufShardSink, GgufSplitOptions,
};

use crate::bridge::ffi::{
    GgufReadAt as CxxGgufReadAt, GgufShardSink as CxxGgufShardSink,
    GgufShardWriter as CxxGgufShardWriter,
};

const STATUS_OK: i32 = 0;
const STATUS_SPLIT_FAILED: i32 = -3;

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
    source: Pin<&mut CxxGgufReadAt>,
) -> i32 {
    let mut source = CxxReadAt { inner: source };
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
    source: Pin<&mut CxxGgufReadAt>,
    sink: Pin<&mut CxxGgufShardSink>,
) -> i32 {
    let mut source = CxxReadAt { inner: source };
    let mut sink = CxxShardSink { inner: sink };
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

struct CxxReadAt<'a> {
    inner: Pin<&'a mut CxxGgufReadAt>,
}

impl GgufReadAt for CxxReadAt<'_> {
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
        let status = self.inner.as_mut().read_at(offset, dst);
        if status == 0 {
            Ok(())
        } else {
            Err(GgufError::Invalid(format!(
                "read_at callback failed with status {status}"
            )))
        }
    }
}

struct CxxShardSink<'a> {
    inner: Pin<&'a mut CxxGgufShardSink>,
}

impl GgufShardSink for CxxShardSink<'_> {
    type Writer = CxxShardWriter;

    fn create_shard(
        &mut self,
        path: &Path,
        index: u16,
        count: u16,
    ) -> Result<Self::Writer, GgufError> {
        let path = path.to_string_lossy();
        let status = self.inner.as_mut().open_shard(path.as_ref(), index, count);
        if status != 0 {
            return Err(GgufError::Invalid(format!(
                "open_shard callback failed with status {status}"
            )));
        }
        Ok(CxxShardWriter {
            inner: self.inner.as_mut().create_writer(),
            bytes_written: 0,
        })
    }

    fn finish_shard(&mut self, mut writer: Self::Writer) -> Result<u64, GgufError> {
        let bytes_written = writer.bytes_written;
        let status = writer.inner.pin_mut().close_shard();
        if status != 0 {
            return Err(GgufError::Invalid(format!(
                "close_shard callback failed with status {status}"
            )));
        }
        Ok(bytes_written)
    }
}

struct CxxShardWriter {
    inner: cxx::UniquePtr<CxxGgufShardWriter>,
    bytes_written: u64,
}

impl Write for CxxShardWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let status = self.inner.pin_mut().write_shard(buf);
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
        let status = self.inner.pin_mut().write_shard(buf);
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
mod tests {
    mod ingest_tests;
}
