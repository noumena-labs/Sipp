//! Unit tests for the parent module.

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
}
