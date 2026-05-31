use std::pin::Pin;

use cogentlm_core::{FinishReason, TokenBatch, TokenDeliveryStats};
use futures_util::Stream;

use crate::{ProviderError, ProviderErrorKind};
use crate::{ProviderKind, ProviderResult, TokenUsage};

pub type ProviderStream<T> = Pin<Box<dyn Stream<Item = ProviderResult<T>> + Send>>;

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderStreamEvent {
    TokenBatch(TokenBatch),
    Usage { usage: TokenUsage },
    Finished { finish_reason: FinishReason },
}

pub(crate) struct TokenBatchBuilder {
    request_id: Option<String>,
    stream_id: u32,
    sequence: u32,
    stats: TokenDeliveryStats,
}

impl TokenBatchBuilder {
    pub(crate) fn new(request_id: Option<String>) -> Self {
        Self {
            request_id,
            stream_id: 0,
            sequence: 0,
            stats: TokenDeliveryStats::default(),
        }
    }

    pub(crate) fn push_text(&mut self, text: &str) -> TokenBatch {
        let byte_count = text.len() as u32;
        self.stats.frames_sent += 1;
        self.stats.bytes_sent += u64::from(byte_count);
        self.stats.batches_sent += 1;

        let batch = TokenBatch {
            request_id: self.request_id.clone().unwrap_or_default(),
            stream_id: self.stream_id,
            sequence_start: self.sequence,
            text: text.to_string(),
            frame_count: 1,
            byte_count,
            stats: self.stats,
        };
        self.sequence += 1;
        batch
    }
}

/// Maximum bytes buffered while waiting for an SSE event boundary. Guards
/// against a server that streams without `\n\n` delimiters.
const MAX_SSE_BUFFER: usize = 1 << 20;
const MAX_SSE_BUFFER_WITH_DELIMITER: usize = MAX_SSE_BUFFER + 4;

pub(crate) struct SseParser {
    buffer: Vec<u8>,
    provider: ProviderKind,
}

impl SseParser {
    pub(crate) fn new(provider: ProviderKind) -> Self {
        Self {
            buffer: Vec::new(),
            provider,
        }
    }

    /// Accept a raw network chunk and return any complete SSE `data:` payloads.
    /// Bytes are buffered, so a multi-byte UTF-8 character split across chunk
    /// boundaries is only decoded once the whole event has arrived.
    pub(crate) fn push(&mut self, mut bytes: &[u8]) -> ProviderResult<Vec<String>> {
        let mut payloads = Vec::new();

        while !bytes.is_empty() {
            let available = MAX_SSE_BUFFER_WITH_DELIMITER.saturating_sub(self.buffer.len());
            if available == 0 {
                return Err(self.buffer_limit_error());
            }

            let take = bytes.len().min(available);
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];

            while let Some((index, length)) = event_boundary(&self.buffer) {
                let payload = self.decode_event(index)?;
                self.buffer.drain(..index + length);
                if let Some(payload) = payload {
                    payloads.push(payload);
                }
            }

            if self.buffer.len() > MAX_SSE_BUFFER {
                return Err(self.buffer_limit_error());
            }
        }

        Ok(payloads)
    }

    /// Flush any trailing event the stream ended without a blank-line boundary.
    /// A complete final `data:` line is still delivered; a genuinely truncated
    /// multi-byte sequence surfaces here as an error rather than silently.
    pub(crate) fn finish(&mut self) -> ProviderResult<Vec<String>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let payload = self.decode_event(self.buffer.len())?;
        self.buffer.clear();
        Ok(payload.into_iter().collect())
    }

    /// Decode `self.buffer[..end]` as one UTF-8 SSE event and extract its `data:`
    /// payload. `end` always lands on an event boundary (an ASCII `\n`), so the
    /// slice is a complete UTF-8 region.
    fn decode_event(&self, end: usize) -> ProviderResult<Option<String>> {
        let event = std::str::from_utf8(&self.buffer[..end]).map_err(|err| {
            ProviderError::new(
                ProviderErrorKind::Provider,
                self.provider,
                format!("invalid UTF-8 SSE event: {err}"),
            )
        })?;
        Ok(sse_data_payload(event))
    }

    fn buffer_limit_error(&self) -> ProviderError {
        ProviderError::new(
            ProviderErrorKind::Provider,
            self.provider,
            "SSE event exceeded buffer limit without a boundary",
        )
    }
}

fn event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    match (
        find_subslice(buffer, b"\r\n\r\n"),
        find_subslice(buffer, b"\n\n"),
    ) {
        (Some(crlf), Some(lf)) if crlf < lf => Some((crlf, 4)),
        (Some(_), Some(lf)) => Some((lf, 2)),
        (Some(crlf), None) => Some((crlf, 4)),
        (None, Some(lf)) => Some((lf, 2)),
        (None, None) => None,
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn sse_data_payload(raw_event: &str) -> Option<String> {
    let lines = raw_event
        .lines()
        .filter_map(|line| line.trim_end_matches('\r').strip_prefix("data:"))
        .map(|data| data.strip_prefix(' ').unwrap_or(data))
        .collect::<Vec<_>>();

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
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
}
