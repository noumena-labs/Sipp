use std::pin::Pin;

use cogentlm_core::{FinishReason, StreamStats, TokenBatch};
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
    stats: StreamStats,
}

impl TokenBatchBuilder {
    pub(crate) fn new(request_id: Option<String>) -> Self {
        Self {
            request_id,
            stream_id: 0,
            sequence: 0,
            stats: StreamStats::default(),
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

pub(crate) struct SseParser {
    buffer: String,
    provider: ProviderKind,
}

impl SseParser {
    pub(crate) fn new(provider: ProviderKind) -> Self {
        Self {
            buffer: String::new(),
            provider,
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) -> ProviderResult<Vec<String>> {
        let chunk = std::str::from_utf8(bytes).map_err(|err| {
            ProviderError::new(
                ProviderErrorKind::Provider,
                self.provider,
                format!("invalid UTF-8 SSE chunk: {err}"),
            )
        })?;
        self.buffer.push_str(chunk);
        let mut payloads = Vec::new();
        while let Some((index, length)) = event_boundary(&self.buffer) {
            let raw_event = self.buffer[..index].to_string();
            self.buffer.drain(..index + length);
            if let Some(payload) = sse_data_payload(&raw_event) {
                payloads.push(payload);
            }
        }
        Ok(payloads)
    }

    /// Flush any trailing event the stream ended without a blank-line boundary.
    /// A complete final `data:` line is still delivered; genuinely truncated
    /// JSON surfaces later as a parse error in the provider's payload parser.
    pub(crate) fn finish(&mut self) -> Vec<String> {
        let remainder = std::mem::take(&mut self.buffer);
        if remainder.trim().is_empty() {
            return Vec::new();
        }
        sse_data_payload(&remainder).into_iter().collect()
    }
}

fn event_boundary(buffer: &str) -> Option<(usize, usize)> {
    match (buffer.find("\r\n\r\n"), buffer.find("\n\n")) {
        (Some(crlf), Some(lf)) if crlf < lf => Some((crlf, 4)),
        (Some(_), Some(lf)) => Some((lf, 2)),
        (Some(crlf), None) => Some((crlf, 4)),
        (None, Some(lf)) => Some((lf, 2)),
        (None, None) => None,
    }
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
            parser.finish(),
            vec![r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string()]
        );
    }
}
