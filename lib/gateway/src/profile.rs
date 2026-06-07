use bytes::Bytes;
use cogentlm_client::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingResponse, CogentQueryRequest,
    CogentTextOptions, CogentTextResponse,
};
use cogentlm_core::{ChatMessage, ChatRole, TokenUsage};
use serde::{Deserialize, Serialize};

use crate::toolkit::{DecodedRequest, GatewayHttpError, ProtocolCodec, ToolkitResult};

/// First-party Cogent query JSON body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryBody {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub options: serde_json::Map<String, serde_json::Value>,
}

/// First-party Cogent chat JSON body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatBody {
    pub model: String,
    pub messages: Vec<ChatMessageBody>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub options: serde_json::Map<String, serde_json::Value>,
}

/// First-party Cogent chat message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageBody {
    pub role: ChatRole,
    pub content: String,
}

/// First-party Cogent embedding JSON body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbedBody {
    pub model: String,
    pub input: String,
    #[serde(flatten)]
    pub options: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
struct TextBody<'a> {
    id: &'a str,
    model: &'a str,
    text: &'a str,
    finish_reason: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageBody>,
}

#[derive(Serialize)]
struct EmbeddingBody<'a> {
    id: &'a str,
    model: &'a str,
    embedding: &'a [f32],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageBody>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct UsageBody {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl From<TokenUsage> for UsageBody {
    fn from(usage: TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl From<UsageBody> for TokenUsage {
    fn from(usage: UsageBody) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

/// First-party gateway JSON and SSE protocol codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct GatewayCodec;

impl ProtocolCodec for GatewayCodec {
    fn decode_query(&self, body: &[u8]) -> ToolkitResult<DecodedRequest<CogentQueryRequest>> {
        let body: QueryBody = decode(body)?;
        Ok(DecodedRequest {
            target: body.model,
            stream: body.stream,
            request: CogentQueryRequest {
                prompt: body.prompt,
                options: text_options(body.max_tokens, body.temperature, body.top_p, body.stop),
                endpoint_options: body.options,
                ..CogentQueryRequest::default()
            },
        })
    }

    fn decode_chat(&self, body: &[u8]) -> ToolkitResult<DecodedRequest<CogentChatRequest>> {
        let body: ChatBody = decode(body)?;
        Ok(DecodedRequest {
            target: body.model,
            stream: body.stream,
            request: CogentChatRequest {
                messages: body
                    .messages
                    .into_iter()
                    .map(|message| ChatMessage::new(message.role, message.content))
                    .collect(),
                options: text_options(body.max_tokens, body.temperature, body.top_p, body.stop),
                endpoint_options: body.options,
                ..CogentChatRequest::default()
            },
        })
    }

    fn decode_embed(&self, body: &[u8]) -> ToolkitResult<DecodedRequest<CogentEmbedRequest>> {
        let body: EmbedBody = decode(body)?;
        Ok(DecodedRequest {
            target: body.model,
            stream: false,
            request: CogentEmbedRequest {
                input: body.input,
                endpoint_options: body.options,
                ..CogentEmbedRequest::default()
            },
        })
    }

    fn encode_text(&self, target: &str, response: &CogentTextResponse) -> ToolkitResult<Bytes> {
        encode(&TextBody {
            id: response
                .metadata
                .upstream_response_id
                .as_deref()
                .unwrap_or("response"),
            model: target,
            text: &response.text,
            finish_reason: response.finish_reason.as_str(),
            usage: response.usage.map(UsageBody::from),
        })
    }

    fn encode_embedding(
        &self,
        target: &str,
        response: &CogentEmbeddingResponse,
    ) -> ToolkitResult<Bytes> {
        encode(&EmbeddingBody {
            id: response
                .metadata
                .upstream_response_id
                .as_deref()
                .unwrap_or("response"),
            model: target,
            embedding: &response.values,
            usage: response.usage.map(UsageBody::from),
        })
    }

    fn encode_stream_event(
        &self,
        event: &cogentlm_gateway_core::GatewayStreamEvent,
    ) -> ToolkitResult<Bytes> {
        let (name, value) = match event {
            cogentlm_gateway_core::GatewayStreamEvent::TokenBatch(batch) => (
                "token",
                serde_json::json!({
                    "text": batch.text,
                    "sequence": batch.sequence_start,
                }),
            ),
            cogentlm_gateway_core::GatewayStreamEvent::Usage(usage) => (
                "usage",
                serde_json::to_value(UsageBody::from(*usage)).map_err(encode_error)?,
            ),
            cogentlm_gateway_core::GatewayStreamEvent::Finished { finish_reason, .. } => (
                "done",
                serde_json::json!({
                    "finish_reason": finish_reason.as_str(),
                }),
            ),
        };
        Ok(Bytes::from(format!(
            "event: {name}\ndata: {}\n\n",
            serde_json::to_string(&value).map_err(encode_error)?
        )))
    }

    fn encode_error(&self, error: &GatewayHttpError) -> Bytes {
        Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "error": {
                    "code": error.code,
                    "message": error.message,
                }
            }))
            .unwrap_or_else(|_| {
                b"{\"error\":{\"code\":\"internal\",\"message\":\"encoding failed\"}}".to_vec()
            }),
        )
    }

    fn encode_stream_error(&self, error: &GatewayHttpError) -> Bytes {
        let value = serde_json::json!({
            "error": {
                "code": error.code,
                "message": error.message,
            }
        });
        Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            serde_json::to_string(&value).unwrap_or_else(|_| {
                "{\"error\":{\"code\":\"internal\",\"message\":\"encoding failed\"}}".to_string()
            })
        ))
    }

    fn content_type(&self, streaming: bool) -> &'static str {
        if streaming {
            "text/event-stream"
        } else {
            "application/json"
        }
    }
}

fn text_options(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Vec<String>,
) -> CogentTextOptions {
    CogentTextOptions {
        max_tokens,
        temperature,
        top_p,
        stop,
    }
}

fn decode<T: serde::de::DeserializeOwned>(body: &[u8]) -> ToolkitResult<T> {
    serde_json::from_slice(body)
        .map_err(|error| GatewayHttpError::bad_request("invalid_json", error.to_string()))
}

fn encode(value: &impl Serialize) -> ToolkitResult<Bytes> {
    serde_json::to_vec(value)
        .map(Bytes::from)
        .map_err(encode_error)
}

fn encode_error(error: serde_json::Error) -> GatewayHttpError {
    GatewayHttpError::internal("encoding_error", error.to_string())
}
