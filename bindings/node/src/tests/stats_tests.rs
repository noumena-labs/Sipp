//! Tests Node binding conversion for response stats exposed through N-API.

use super::*;

#[test]
fn request_stats_map_tps_fields() {
    let stats = request_stats_to_node(CoreRequestStats {
        input_tokens: 1,
        output_tokens: 2,
        e2e_tokens_per_second: Some(40.0),
        decode_tokens_per_second: Some(80.0),
        ..CoreRequestStats::default()
    });

    assert_eq!(stats.e2e_tokens_per_second, Some(40.0));
    assert_eq!(stats.decode_tokens_per_second, Some(80.0));
}
