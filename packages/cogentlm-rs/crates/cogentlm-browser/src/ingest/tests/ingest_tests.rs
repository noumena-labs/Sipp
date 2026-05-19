//! Unit tests for the parent module.

use super::super::*;

#[test]
fn cache_layout_uses_split_for_unknown_or_large_sources() {
    assert_eq!(cogentlm_browser_cache_layout(1024, true, 2048, 512), 0);
    assert_eq!(cogentlm_browser_cache_layout(4096, true, 2048, 512), 1);
    assert_eq!(cogentlm_browser_cache_layout(0, false, 2048, 512), 1);
}

unsafe extern "C" fn ok_write_shard(_user_data: *mut c_void, _buf: *const u8, _len: usize) -> i32 {
    0
}

#[test]
fn callback_writer_rejects_byte_count_overflow() {
    let mut writer = CallbackShardWriter {
        user_data: std::ptr::null_mut(),
        write_shard: ok_write_shard,
        bytes_written: u64::MAX,
    };

    let error = writer.write(&[1]).expect_err("overflow");

    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(writer.bytes_written, u64::MAX);
}
