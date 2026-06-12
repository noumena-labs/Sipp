use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::core::{FinishReason, TokenBatch, TokenUsage};
use crate::providers::{
    ProviderChatRequest, ProviderEmbedRequest, ProviderGenerateRequest, ProviderRequestContext,
    ProviderStreamEvent, ProviderTransport,
};
use futures::StreamExt;
use futures_channel::mpsc;

use crate::client::dispatch::InferenceEndpoint;
use crate::client::io_executor::IoExecutor;
use crate::client::{
    map, validate, SippChatRequest, SippEmbedRequest, SippEmbeddingRun, SippError,
    SippQueryRequest, SippRequestContext, SippResponseMetadata, SippResult,
    SippTextResponse, SippTextRun, SippTokenBatches, EndpointCapabilities, EndpointRef,
    ProviderEndpointError,
};

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) struct ProviderEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    model: String,
    transport: ProviderTransport,
    executor: IoExecutor,
    secrets: Vec<String>,
}

impl ProviderEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        model: String,
        capabilities: EndpointCapabilities,
        transport: ProviderTransport,
        executor: IoExecutor,
        secrets: Vec<String>,
    ) -> Self {
        Self {
            endpoint,
            capabilities,
            model,
            transport,
            executor,
            secrets,
        }
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn secrets(&self) -> Vec<String> {
        self.secrets.clone()
    }
}

impl InferenceEndpoint for ProviderEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn query_with_context(
        &self,
        context: SippRequestContext,
        request: SippQueryRequest,
    ) -> SippTextRun {
        if let Err(error) = validate::provider_query(&request) {
            return SippTextRun::ready_err(error);
        }
        let provider_request = ProviderGenerateRequest {
            model: self.model(),
            prompt: request.prompt,
            options: map::provider_generation_options(request.options),
            provider_options: request.provider_options,
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let secrets = self.secrets();
        let request_id = context.request_id;
        let provider_context = ProviderRequestContext {
            request_id: request_id.clone(),
        };

        if request.emit_tokens {
            let (batch_tx, batch_rx) = mpsc::unbounded();
            let join = executor.spawn(async move {
                run_provider_query_stream(
                    transport,
                    endpoint,
                    request_id,
                    provider_context,
                    provider_request,
                    batch_tx,
                    secrets,
                )
                .await
            });
            SippTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                SippTokenBatches::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                transport
                    .generate_with_context(provider_context, provider_request)
                    .await
                    .map(|response| map::provider_text_response(endpoint, request_id, response))
                    .map_err(|error| provider_error(error, &secrets))
            });
            SippTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                SippTokenBatches::closed(),
            )
        }
    }

    fn chat_with_context(
        &self,
        context: SippRequestContext,
        request: SippChatRequest,
    ) -> SippTextRun {
        if let Err(error) = validate::provider_chat(&request) {
            return SippTextRun::ready_err(error);
        }
        let provider_request = ProviderChatRequest {
            model: self.model(),
            messages: request.messages,
            options: map::provider_generation_options(request.options),
            provider_options: request.provider_options,
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let secrets = self.secrets();
        let request_id = context.request_id;
        let provider_context = ProviderRequestContext {
            request_id: request_id.clone(),
        };

        if request.emit_tokens {
            let (batch_tx, batch_rx) = mpsc::unbounded();
            let join = executor.spawn(async move {
                run_provider_chat_stream(
                    transport,
                    endpoint,
                    request_id,
                    provider_context,
                    provider_request,
                    batch_tx,
                    secrets,
                )
                .await
            });
            SippTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                SippTokenBatches::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                transport
                    .chat_with_context(provider_context, provider_request)
                    .await
                    .map(|response| map::provider_chat_response(endpoint, request_id, response))
                    .map_err(|error| provider_error(error, &secrets))
            });
            SippTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                SippTokenBatches::closed(),
            )
        }
    }

    fn embed_with_context(
        &self,
        context: SippRequestContext,
        request: SippEmbedRequest,
    ) -> SippEmbeddingRun {
        if let Err(error) = validate::provider_embed(&request) {
            return SippEmbeddingRun::ready_err(error);
        }
        let provider_request = ProviderEmbedRequest {
            model: self.model(),
            input: request.input,
            provider_options: request.provider_options,
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let secrets = self.secrets();
        let request_id = context.request_id;
        let provider_context = ProviderRequestContext {
            request_id: request_id.clone(),
        };
        let join = executor.spawn(async move {
            transport
                .embed_with_context(provider_context, provider_request)
                .await
                .map(|response| map::provider_embedding_response(endpoint, request_id, response))
                .map_err(|error| provider_error(error, &secrets))
        });
        SippEmbeddingRun::new(Box::pin(ProviderResponseFuture::new(join, executor)))
    }
}

struct ProviderResponseFuture<T> {
    join: tokio::task::JoinHandle<SippResult<T>>,
    _executor: IoExecutor,
}

impl<T> ProviderResponseFuture<T> {
    fn new(join: tokio::task::JoinHandle<SippResult<T>>, executor: IoExecutor) -> Self {
        Self {
            join,
            _executor: executor,
        }
    }
}

impl<T> Future for ProviderResponseFuture<T> {
    type Output = SippResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.join).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(error)) => Poll::Ready(Err(SippError::Internal(format!(
                "provider task failed: {error}"
            )))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Drop for ProviderResponseFuture<T> {
    fn drop(&mut self) {
        self.join.abort();
    }
}

async fn run_provider_query_stream(
    transport: ProviderTransport,
    endpoint: EndpointRef,
    request_id: Option<String>,
    context: ProviderRequestContext,
    request: ProviderGenerateRequest,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    secrets: Vec<String>,
) -> SippResult<SippTextResponse> {
    let stream = transport
        .stream_generate_with_context(context, request)
        .await
        .map_err(|error| provider_error(error, &secrets))?;
    collect_provider_stream(endpoint, request_id, stream, batch_tx, secrets).await
}

async fn run_provider_chat_stream(
    transport: ProviderTransport,
    endpoint: EndpointRef,
    request_id: Option<String>,
    context: ProviderRequestContext,
    request: ProviderChatRequest,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    secrets: Vec<String>,
) -> SippResult<SippTextResponse> {
    let stream = transport
        .stream_chat_with_context(context, request)
        .await
        .map_err(|error| provider_error(error, &secrets))?;
    collect_provider_stream(endpoint, request_id, stream, batch_tx, secrets).await
}

async fn collect_provider_stream(
    endpoint: EndpointRef,
    request_id: Option<String>,
    mut stream: crate::providers::ProviderStream<ProviderStreamEvent>,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    secrets: Vec<String>,
) -> SippResult<SippTextResponse> {
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut usage: Option<TokenUsage> = None;
    let mut upstream_request_id = None;

    while let Some(event) = stream.next().await {
        match event.map_err(|error| provider_error(error, &secrets))? {
            ProviderStreamEvent::TokenBatch(batch) => {
                if upstream_request_id.is_none() && !batch.request_id.is_empty() {
                    upstream_request_id = Some(batch.request_id.clone());
                }
                text.push_str(&batch.text);
                let _ = batch_tx.unbounded_send(batch);
            }
            ProviderStreamEvent::Usage { usage: next } => usage = Some(next),
            ProviderStreamEvent::Finished {
                finish_reason: reason,
            } => finish_reason = reason,
        }
    }

    Ok(SippTextResponse {
        endpoint,
        text,
        finish_reason,
        usage,
        local_stats: None,
        metadata: SippResponseMetadata {
            request_id,
            upstream_request_id,
            upstream_response_id: None,
        },
    })
}

fn provider_error(error: crate::providers::ProviderError, secrets: &[String]) -> SippError {
    SippError::Provider(ProviderEndpointError::from_provider_error(error, secrets))
}
