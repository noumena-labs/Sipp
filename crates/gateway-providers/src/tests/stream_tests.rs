//! Tests the `stream` module in `cogentlm-gateway-providers`.
//!
//! Covers provider stream batch construction and SSE payload parsing with
//! deterministic byte fixtures rather than live provider streams.

use super::*;

#[test]
fn token_batch_builder_advances_sequence_and_stats() {
    let mut builder = TokenBatchBuilder::new(Some("req-1".to_string()));

    let first = builder.push_text("he");
    let second = builder.push_text("llo");

    assert_eq!(first.request_id, "req-1");
    assert_eq!(first.sequence_start, 0);
    assert_eq!(first.text, "he");
    assert_eq!(first.byte_count, 2);
    assert_eq!(first.stats.frames_sent, 1);
    assert_eq!(first.stats.bytes_sent, 2);
    assert_eq!(first.stats.batches_sent, 1);

    assert_eq!(second.request_id, "req-1");
    assert_eq!(second.sequence_start, 1);
    assert_eq!(second.text, "llo");
    assert_eq!(second.byte_count, 3);
    assert_eq!(second.stats.frames_sent, 2);
    assert_eq!(second.stats.bytes_sent, 5);
    assert_eq!(second.stats.batches_sent, 2);
}

#[test]
fn sse_parser_extracts_partial_multiline_payloads() {
    let mut parser = SseParser::new(ProviderKind::OpenAiCompatible);

    let first = parser
        .push(b"event: token\ndata: he\ndata")
        .expect("partial event");
    assert!(first.is_empty());

    let second = parser.push(b": llo\n\n").expect("complete event");

    assert_eq!(second, vec!["he\nllo".to_string()]);
}

#[test]
fn sse_parser_handles_crlf_boundaries_and_flushes_trailing_event() {
    let mut parser = SseParser::new(ProviderKind::Anthropic);

    let payloads = parser
        .push(b"event: ping\r\ndata: {}\r\n\r\ndata: trailing")
        .expect("crlf event");
    assert_eq!(payloads, vec!["{}".to_string()]);

    let trailing = parser.finish().expect("trailing event");
    assert_eq!(trailing, vec!["trailing".to_string()]);
}

#[test]
fn sse_parser_rejects_invalid_utf8_at_event_boundary() {
    let mut parser = SseParser::new(ProviderKind::OpenAi);

    let error = parser
        .push(b"data: \xFF\n\n")
        .expect_err("invalid UTF-8 must fail");

    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(error.provider, ProviderKind::OpenAi);
}
