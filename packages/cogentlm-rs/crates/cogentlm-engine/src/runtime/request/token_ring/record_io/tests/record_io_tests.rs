//! Unit tests for the parent module.

use super::super::*;

#[test]
fn record_header_round_trips_without_slice_conversions() {
    let header = TokenRingRecordHeader {
        stream_id: 7,
        sequence: u32::MAX - 1,
        flags: 3,
        byte_len: 4096,
    };

    assert_eq!(TokenRingRecordHeader::decode(header.encode()), header);
}
