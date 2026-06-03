use std::collections::VecDeque;
use std::pin::Pin;

use bytes::Bytes;
use cogentlm_core::{FinishReason, TokenBatch, TokenEmissionStats, TokenUsage};
use futures_util::{stream as futures_stream, Stream, StreamExt};

use crate::response::{gateway_body_error, map_finish_reason, token_usage};
use crate::{GatewayError, GatewayErrorKind, GatewayResult};

/// Gateway stream type used by streaming query and chat operations.
pub type GatewayStream<T> = Pin<Box<dyn Stream<Item = GatewayResult<T>> + Send>>;

pub(crate) type GatewayByteStream = Pin<Box<dyn Stream<Item = GatewayResult<Bytes>> + Send>>;

/// Normalized streaming event returned by a CogentLM gateway.
#[derive(Debug, Clone, PartialEq)]
pub enum GatewayStreamEvent {
    /// Text token batch emitted by a streaming text operation.
    TokenBatch(TokenBatch),
    /// Usage reported by the gateway.
    Usage { usage: TokenUsage },
    /// Final finish event.
    Finished { finish_reason: FinishReason },
}

pub(crate) fn gateway_stream_events(
    request_id: Option<String>,
    byte_stream: GatewayByteStream,
    redaction_secret: String,
) -> GatewayStream<GatewayStreamEvent> {
    let state = GatewayStreamState {
        stream: byte_stream,
        parser: SseParser::new(),
        pending: VecDeque::new(),
        batcher: TokenBatchBuilder::new(request_id),
        redaction_secret,
        closed: false,
    };

    Box::pin(futures_stream::unfold(state, next_gateway_stream_event))
}

struct GatewayStreamState {
    stream: GatewayByteStream,
    parser: SseParser,
    pending: VecDeque<GatewayResult<GatewayStreamEvent>>,
    batcher: TokenBatchBuilder,
    redaction_secret: String,
    closed: bool,
}

async fn next_gateway_stream_event(
    mut state: GatewayStreamState,
) -> Option<(GatewayResult<GatewayStreamEvent>, GatewayStreamState)> {
    loop {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.closed {
            return None;
        }

        match state.stream.next().await {
            Some(Ok(bytes)) => {
                if let Err(error) = state.push_bytes(&bytes) {
                    state.closed = true;
                    return Some((Err(error), state));
                }
            }
            Some(Err(error)) => {
                state.closed = true;
                return Some((Err(error), state));
            }
            None => {
                state.closed = true;
                if let Err(error) = state.finish_parser() {
                    return Some((Err(error), state));
                }
            }
        }
    }
}

impl GatewayStreamState {
    fn push_bytes(&mut self, bytes: &[u8]) -> GatewayResult<()> {
        for event in self.parser.push(bytes)? {
            self.push_event(event)?;
        }
        Ok(())
    }

    fn finish_parser(&mut self) -> GatewayResult<()> {
        for event in self.parser.finish()? {
            self.push_event(event)?;
        }
        Ok(())
    }

    fn push_event(&mut self, event: SseEvent) -> GatewayResult<()> {
        if event.data.trim() == "[DONE]" {
            return Ok(());
        }

        match event.event.as_deref().unwrap_or("message") {
            "token" => {
                let value = parse_json_payload(&event.data)?;
                let text = value
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| stream_protocol_error("token event missing text"))?;
                let sequence = optional_u32_field(&value, "sequence")?;
                self.pending.push_back(Ok(GatewayStreamEvent::TokenBatch(
                    self.batcher.push_text(text, sequence),
                )));
                Ok(())
            }
            "usage" => {
                let value = parse_json_payload(&event.data)?;
                self.pending.push_back(Ok(GatewayStreamEvent::Usage {
                    usage: token_usage(&value)?,
                }));
                Ok(())
            }
            "done" => {
                let value = parse_json_payload(&event.data)?;
                let finish_reason = value
                    .get("finish_reason")
                    .and_then(serde_json::Value::as_str);
                self.pending.push_back(Ok(GatewayStreamEvent::Finished {
                    finish_reason: map_finish_reason(finish_reason),
                }));
                Ok(())
            }
            "error" => {
                let value = parse_json_payload(&event.data)?;
                let mut error = gateway_body_error(value, "gateway stream error");
                error.request_id = self.batcher.request_id.clone();
                error.redact_secret(&self.redaction_secret);
                Err(error)
            }
            name => Err(stream_protocol_error(format!(
                "unsupported gateway stream event: {name}"
            ))),
        }
    }
}

fn parse_json_payload(payload: &str) -> GatewayResult<serde_json::Value> {
    serde_json::from_str(payload).map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::Gateway,
            format!("invalid gateway stream JSON payload: {error}"),
        )
    })
}

fn optional_u32_field(
    value: &serde_json::Value,
    field: &'static str,
) -> GatewayResult<Option<u32>> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    let Some(number) = raw.as_u64() else {
        return Err(stream_protocol_error(format!(
            "gateway stream field is not an unsigned integer: {field}"
        )));
    };
    u32::try_from(number)
        .map(Some)
        .map_err(|_| stream_protocol_error(format!("gateway stream field exceeds u32: {field}")))
}

fn stream_protocol_error(message: impl Into<String>) -> GatewayError {
    GatewayError::new(GatewayErrorKind::Gateway, message.into())
}

struct TokenBatchBuilder {
    request_id: Option<String>,
    stream_id: u32,
    sequence: u32,
    stats: TokenEmissionStats,
}

impl TokenBatchBuilder {
    fn new(request_id: Option<String>) -> Self {
        Self {
            request_id,
            stream_id: 0,
            sequence: 0,
            stats: TokenEmissionStats::default(),
        }
    }

    fn push_text(&mut self, text: &str, sequence_start: Option<u32>) -> TokenBatch {
        let byte_count = text.len() as u32;
        self.stats.frames_sent += 1;
        self.stats.bytes_sent += u64::from(byte_count);
        self.stats.batches_sent += 1;
        let sequence_start = sequence_start.unwrap_or(self.sequence);

        let batch = TokenBatch {
            request_id: self.request_id.clone().unwrap_or_default(),
            stream_id: self.stream_id,
            sequence_start,
            text: text.to_string(),
            frame_count: 1,
            byte_count,
            stats: self.stats,
        };
        self.sequence = sequence_start.wrapping_add(1);
        batch
    }
}

struct SseEvent {
    event: Option<String>,
    data: String,
}

struct SseParser {
    buffer: Vec<u8>,
}

const MAX_SSE_BUFFER: usize = 1 << 20;
const MAX_SSE_BUFFER_WITH_DELIMITER: usize = MAX_SSE_BUFFER + 4;

impl SseParser {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn push(&mut self, mut bytes: &[u8]) -> GatewayResult<Vec<SseEvent>> {
        let mut events = Vec::new();

        while !bytes.is_empty() {
            let available = MAX_SSE_BUFFER_WITH_DELIMITER.saturating_sub(self.buffer.len());
            if available == 0 {
                return Err(self.buffer_limit_error());
            }

            let take = bytes.len().min(available);
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];

            while let Some((index, length)) = event_boundary(&self.buffer) {
                let event = self.decode_event(index)?;
                self.buffer.drain(..index + length);
                if let Some(event) = event {
                    events.push(event);
                }
            }

            if self.buffer.len() > MAX_SSE_BUFFER {
                return Err(self.buffer_limit_error());
            }
        }

        Ok(events)
    }

    fn finish(&mut self) -> GatewayResult<Vec<SseEvent>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let event = self.decode_event(self.buffer.len())?;
        self.buffer.clear();
        Ok(event.into_iter().collect())
    }

    fn decode_event(&self, end: usize) -> GatewayResult<Option<SseEvent>> {
        let event = std::str::from_utf8(&self.buffer[..end]).map_err(|error| {
            GatewayError::new(
                GatewayErrorKind::Gateway,
                format!("invalid UTF-8 SSE event: {error}"),
            )
        })?;
        Ok(sse_event(event))
    }

    fn buffer_limit_error(&self) -> GatewayError {
        GatewayError::new(
            GatewayErrorKind::Gateway,
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

fn sse_event(raw_event: &str) -> Option<SseEvent> {
    let mut event = None;
    let mut data = Vec::new();

    for line in raw_event.lines().map(|line| line.trim_end_matches('\r')) {
        if let Some(name) = line.strip_prefix("event:") {
            event = Some(name.strip_prefix(' ').unwrap_or(name).to_string());
        } else if let Some(payload) = line.strip_prefix("data:") {
            data.push(payload.strip_prefix(' ').unwrap_or(payload));
        }
    }

    if data.is_empty() {
        None
    } else {
        Some(SseEvent {
            event,
            data: data.join("\n"),
        })
    }
}
