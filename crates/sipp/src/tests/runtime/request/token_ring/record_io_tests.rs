//! Tests the `runtime::request::token_ring::record_io` module in `sipp`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use super::*;

#[test]
fn record_header_round_trips_without_slice_conversions() {
    let header = TokenRingRecordHeader {
        stream_id: 7,
        sequence: u32::MAX - 1,
        frame_count: 3,
        byte_len: 4096,
    };

    assert_eq!(TokenRingRecordHeader::decode(header.encode()), header);
}
