use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use cogentlm_core::{FinishReason, TokenBatch, TokenUsage};
use cogentlm_providers::{
    ProviderChatRequest, ProviderEmbedRequest, ProviderGenerateRequest, ProviderStreamEvent,
    ProviderTransport,
};
use futures::StreamExt;
use futures_channel::mpsc;

use crate::dispatch::InferenceEndpoint;
use crate::remote_executor::RemoteExecutor;
use crate::{
    map, validate, CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError,
    CogentQueryRequest, CogentResult, CogentTextResponse, CogentTextRun, CogentTokenBatches,
    EndpointCapabilities, EndpointRef, ProviderEndpointError,
};

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) struct ProviderEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    model: String,
    transport: ProviderTransport,
    executor: RemoteExecutor,
    secrets: Vec<String>,
}

impl ProviderEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        model: String,
        capabilities: EndpointCapabilities,
        transport: ProviderTransport,
        executor: RemoteExecutor,
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

    fn query(&self, request: CogentQueryRequest) -> CogentTextRun {
        if let Err(error) = validate::provider_query(&request) {
            return CogentTextRun::ready_err(error);
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

        if request.emit_tokens {
            let (batch_tx, batch_rx) = mpsc::unbounded();
            let join = executor.spawn(async move {
                run_provider_query_stream(transport, endpoint, provider_request, batch_tx, secrets)
                    .await
            });
            CogentTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                CogentTokenBatches::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                transport
                    .generate(provider_request)
                    .await
                    .map(|response| map::provider_text_response(endpoint, response))
                    .map_err(|error| provider_error(error, &secrets))
            });
            CogentTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                CogentTokenBatches::closed(),
            )
        }
    }

    fn chat(&self, request: CogentChatRequest) -> CogentTextRun {
        if let Err(error) = validate::provider_chat(&request) {
            return CogentTextRun::ready_err(error);
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

        if request.emit_tokens {
            let (batch_tx, batch_rx) = mpsc::unbounded();
            let join = executor.spawn(async move {
                run_provider_chat_stream(transport, endpoint, provider_request, batch_tx, secrets)
                    .await
            });
            CogentTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                CogentTokenBatches::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                transport
                    .chat(provider_request)
                    .await
                    .map(|response| map::provider_chat_response(endpoint, response))
                    .map_err(|error| provider_error(error, &secrets))
            });
            CogentTextRun::new(
                Box::pin(ProviderResponseFuture::new(join, executor)),
                CogentTokenBatches::closed(),
            )
        }
    }

    fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun {
        if let Err(error) = validate::provider_embed(&request) {
            return CogentEmbeddingRun::ready_err(error);
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
        let join = executor.spawn(async move {
            transport
                .embed(provider_request)
                .await
                .map(|response| map::provider_embedding_response(endpoint, response))
                .map_err(|error| provider_error(error, &secrets))
        });
        CogentEmbeddingRun::new(Box::pin(ProviderResponseFuture::new(join, executor)))
    }
}

struct ProviderResponseFuture<T> {
    join: tokio::task::JoinHandle<CogentResult<T>>,
    _executor: RemoteExecutor,
}

impl<T> ProviderResponseFuture<T> {
    fn new(join: tokio::task::JoinHandle<CogentResult<T>>, executor: RemoteExecutor) -> Self {
        Self {
            join,
            _executor: executor,
        }
    }
}

impl<T> Future for ProviderResponseFuture<T> {
    type Output = CogentResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.join).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(error)) => Poll::Ready(Err(CogentError::Internal(format!(
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
    request: ProviderGenerateRequest,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    secrets: Vec<String>,
) -> CogentResult<CogentTextResponse> {
    let stream = transport
        .stream_generate(request)
        .await
        .map_err(|error| provider_error(error, &secrets))?;
    collect_provider_stream(endpoint, stream, batch_tx, secrets).await
}

async fn run_provider_chat_stream(
    transport: ProviderTransport,
    endpoint: EndpointRef,
    request: ProviderChatRequest,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    secrets: Vec<String>,
) -> CogentResult<CogentTextResponse> {
    let stream = transport
        .stream_chat(request)
        .await
        .map_err(|error| provider_error(error, &secrets))?;
    collect_provider_stream(endpoint, stream, batch_tx, secrets).await
}

async fn collect_provider_stream(
    endpoint: EndpointRef,
    mut stream: cogentlm_providers::ProviderStream<ProviderStreamEvent>,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    secrets: Vec<String>,
) -> CogentResult<CogentTextResponse> {
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut usage: Option<TokenUsage> = None;

    while let Some(event) = stream.next().await {
        match event.map_err(|error| provider_error(error, &secrets))? {
            ProviderStreamEvent::TokenBatch(batch) => {
                text.push_str(&batch.text);
                let _ = batch_tx.unbounded_send(batch);
            }
            ProviderStreamEvent::Usage { usage: next } => usage = Some(next),
            ProviderStreamEvent::Finished {
                finish_reason: reason,
            } => finish_reason = reason,
        }
    }

    Ok(CogentTextResponse {
        endpoint,
        text,
        finish_reason,
        usage,
        local_stats: None,
    })
}

fn provider_error(error: cogentlm_providers::ProviderError, secrets: &[String]) -> CogentError {
    CogentError::Provider(ProviderEndpointError::from_provider_error(error, secrets))
}
