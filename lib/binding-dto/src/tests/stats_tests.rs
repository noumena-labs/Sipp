//! Tests result-stats conversion from core types.

use super::*;

#[test]
fn request_stats_map_tps_fields() {
    let stats = RequestStats::from(CoreRequestStats {
        input_tokens: 1,
        output_tokens: 2,
        e2e_tokens_per_second: Some(40.0),
        decode_tokens_per_second: Some(80.0),
        ..CoreRequestStats::default()
    });

    assert_eq!(stats.e2e_tokens_per_second, Some(40.0));
    assert_eq!(stats.decode_tokens_per_second, Some(80.0));
}
