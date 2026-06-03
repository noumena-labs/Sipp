//! Tests the `stream` module in `cogentlm-providers`.
//!
//! Covers token batch construction and SSE parsing boundaries, including split
//! chunks, CRLF/LF delimiters, no-data events, UTF-8 errors, and bounded
//! buffering with deterministic byte fixtures and no HTTP calls.

use super::*;

#[test]
fn token_batch_builder_tracks_sequence_and_stats() {
    let mut builder = TokenBatchBuilder::new(Some("req-1".to_string()));

    let first = builder.push_text("he");
    let second = builder.push_text("llo");

    assert_eq!(first.request_id, "req-1");
    assert_eq!(first.sequence_start, 0);
    assert_eq!(first.byte_count, 2);
    assert_eq!(first.stats.frames_sent, 1);
    assert_eq!(first.stats.bytes_sent, 2);
    assert_eq!(first.stats.batches_sent, 1);

    assert_eq!(second.sequence_start, 1);
    assert_eq!(second.byte_count, 3);
    assert_eq!(second.stats.frames_sent, 2);
    assert_eq!(second.stats.bytes_sent, 5);
    assert_eq!(second.stats.batches_sent, 2);
}

#[test]
fn token_batch_builder_uses_empty_request_id_when_absent() {
    let mut builder = TokenBatchBuilder::new(None);

    let batch = builder.push_text("hi");

    assert_eq!(batch.request_id, "");
    assert_eq!(batch.stream_id, 0);
    assert_eq!(batch.sequence_start, 0);
}

#[test]
fn sse_parser_handles_partial_events() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    let first = parser
        .push(br#"data: {"choices":[{"delta":{"content":"he"}"#)
        .expect("partial push");
    assert!(first.is_empty());

    let second = parser
        .push(b"}]}\n\ndata: [DONE]\n\n")
        .expect("complete push");
    assert_eq!(
        second,
        vec![
            r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string(),
            "[DONE]".to_string()
        ]
    );
}

#[test]
fn sse_parser_flushes_trailing_event() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    let pushed = parser
        .push(br#"data: {"choices":[{"delta":{"content":"he"}}]}"#)
        .expect("partial push");
    assert!(pushed.is_empty());

    assert_eq!(
        parser.finish().expect("flush trailing event"),
        vec![r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string()]
    );
}

#[test]
fn sse_parser_finish_is_empty_when_no_bytes_are_buffered() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    assert!(parser.finish().expect("empty finish").is_empty());
}

#[test]
fn sse_parser_handles_crlf_multiline_data_and_ignores_no_data_events() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    let payloads = parser
        .push(b": keepalive\r\ndata: first\r\ndata: second\r\n\r\nevent: ping\r\n\r\n")
        .expect("crlf events");

    assert_eq!(payloads, vec!["first\nsecond".to_string()]);
}

#[test]
fn sse_parser_handles_mixed_delimiter_ordering() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    let payloads = parser
        .push(b"data: crlf\r\n\r\ndata: lf\n\ndata: crlf-again\r\n\r\n")
        .expect("mixed delimiters");

    assert_eq!(
        payloads,
        vec![
            "crlf".to_string(),
            "lf".to_string(),
            "crlf-again".to_string()
        ]
    );
}

#[test]
fn sse_parser_handles_utf8_split_across_chunks() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    // "é" is 0xC3 0xA9; split the two bytes across separate network chunks.
    let mut first = br#"data: {"t":"caf"#.to_vec();
    first.push(0xC3);
    assert!(parser.push(&first).expect("partial push").is_empty());

    let mut second = vec![0xA9];
    second.extend_from_slice(b"\"}\n\n");
    let payloads = parser.push(&second).expect("complete push");

    assert_eq!(payloads, vec![r#"{"t":"café"}"#.to_string()]);
}

#[test]
fn sse_parser_rejects_delimiterless_stream_without_large_buffer() {
    let mut parser = SseParser::new(ProviderKind::Proxy);
    let bytes = vec![b'x'; MAX_SSE_BUFFER + 8];

    let err = parser
        .push(&bytes)
        .expect_err("delimiterless event over limit should fail");

    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert!(parser.buffer.len() <= MAX_SSE_BUFFER_WITH_DELIMITER);

    let err = parser
        .push(b"x")
        .expect_err("already-full buffer should fail before extending");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
}

#[test]
fn sse_parser_reports_invalid_utf8_on_completed_or_flushed_events() {
    let mut parser = SseParser::new(ProviderKind::Proxy);
    let mut bytes = b"data: ".to_vec();
    bytes.push(0xC3);
    bytes.extend_from_slice(b"\n\n");

    let err = parser
        .push(&bytes)
        .expect_err("invalid utf8 complete event should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let mut parser = SseParser::new(ProviderKind::Proxy);
    let mut bytes = b"data: ".to_vec();
    bytes.push(0xC3);
    assert!(parser
        .push(&bytes)
        .expect("partial invalid utf8")
        .is_empty());

    let err = parser
        .finish()
        .expect_err("invalid utf8 trailing event should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
}
