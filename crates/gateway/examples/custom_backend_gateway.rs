use std::{net::SocketAddr, sync::Arc};

use async_trait::async_trait;
use cogentlm_core::{ChatRole, FinishReason, TokenBatch, TokenEmissionStats};
use cogentlm_gateway::{
    BackendChatRequest, BackendEmbedRequest, BackendEmbeddingOutput, BackendQueryRequest,
    BackendTextOutput, GatewayAlias, GatewayAliasLimits, GatewayBackend, GatewayError,
    GatewayErrorKind, GatewayResult, GatewayService, GatewayServiceLimits, GatewayState,
    GatewayStream, GatewayStreamEvent, OperationSet,
};
use futures_util::stream;

struct CustomBackend;

#[async_trait]
impl GatewayBackend for CustomBackend {
    async fn query(&self, req: BackendQueryRequest) -> GatewayResult<BackendTextOutput> {
        Ok(text_output(format!("custom query: {}", req.prompt)))
    }

    async fn stream_query(
        &self,
        req: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        Ok(one_batch_stream(format!("custom query: {}", req.prompt)))
    }

    async fn chat(&self, req: BackendChatRequest) -> GatewayResult<BackendTextOutput> {
        Ok(text_output(format!(
            "custom chat: {}",
            last_user_message(req)?
        )))
    }

    async fn stream_chat(
        &self,
        req: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        Ok(one_batch_stream(format!(
            "custom chat: {}",
            last_user_message(req)?
        )))
    }

    async fn embed(&self, req: BackendEmbedRequest) -> GatewayResult<BackendEmbeddingOutput> {
        Ok(BackendEmbeddingOutput {
            values: vec![req.input.len() as f32],
            usage: None,
            response_id: None,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("COGENTLM_GATEWAY_TOKEN")
        .map_err(|_| "set COGENTLM_GATEWAY_TOKEN before running the custom gateway example")?;
    let bind = std::env::var("COGENTLM_GATEWAY_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8787".to_string())
        .parse::<SocketAddr>()?;
    let allowed_origins = std::env::var("COGENTLM_GATEWAY_ORIGIN")
        .ok()
        .into_iter()
        .collect::<Vec<_>>();

    let mut state = GatewayState::new(token);
    state.add_alias(GatewayAlias::new(
        "custom",
        OperationSet::all(),
        Arc::new(CustomBackend),
        GatewayAliasLimits::default(),
    ))?;

    let router = GatewayService::new(state, allowed_origins, GatewayServiceLimits::default())
        .router()?
        .into_make_service();
    let listener = tokio::net::TcpListener::bind(bind).await?;
    println!("custom gateway listening on {bind}");
    axum::serve(listener, router).await?;
    Ok(())
}

fn last_user_message(req: BackendChatRequest) -> GatewayResult<String> {
    req.messages
        .into_iter()
        .rev()
        .find(|message| message.role == ChatRole::User)
        .map(|message| message.content)
        .ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "custom backend requires at least one user message",
            )
        })
}

fn text_output(text: String) -> BackendTextOutput {
    BackendTextOutput {
        text,
        finish_reason: FinishReason::Stop,
        usage: None,
        response_id: None,
    }
}

fn one_batch_stream(text: String) -> GatewayStream<GatewayStreamEvent> {
    let byte_count = text.len() as u32;
    Box::pin(stream::iter([
        Ok(GatewayStreamEvent::TokenBatch(TokenBatch {
            request_id: "custom".to_string(),
            stream_id: 0,
            sequence_start: 0,
            text,
            frame_count: 1,
            byte_count,
            stats: TokenEmissionStats {
                frames_sent: 1,
                bytes_sent: u64::from(byte_count),
                batches_sent: 1,
            },
        })),
        Ok(GatewayStreamEvent::Finished {
            finish_reason: FinishReason::Stop,
        }),
    ]))
}
