use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use cogentlm_core::{FinishReason, TokenBatch, TokenUsage};
use cogentlm_providers::{
    ProviderChatRequest, ProviderEmbedRequest, ProviderGenerateRequest, ProviderGenerationOptions,
    ProviderStreamEvent, ProviderTransport,
};
use futures::StreamExt;
use futures_channel::mpsc;

use crate::dispatch::InferenceEndpoint;
use crate::remote_executor::RemoteExecutor;
use crate::{
    map, validate, CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError,
    CogentQueryRequest, CogentResult, CogentTextResponse, CogentTextRun, CogentTokenBatches,
    EndpointCapabilities, EndpointRef,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/remote_endpoint_tests.rs"]
mod remote_endpoint_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) struct RemoteEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    model: String,
    transport: ProviderTransport,
    executor: RemoteExecutor,
}

impl RemoteEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        model: String,
        capabilities: EndpointCapabilities,
        transport: ProviderTransport,
        executor: RemoteExecutor,
    ) -> Self {
        Self {
            endpoint,
            capabilities,
            model,
            transport,
            executor,
        }
    }

    fn model(&self) -> String {
        self.model.clone()
    }
}

impl InferenceEndpoint for RemoteEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn query(&self, request: CogentQueryRequest) -> CogentTextRun {
        if request.emit_tokens {
            return CogentTextRun::ready_err(CogentError::UnsupportedOperation {
                endpoint: self.endpoint.clone(),
                operation: "query",
            });
        }
        if let Err(error) = validate::remote_query(&request) {
            return CogentTextRun::ready_err(error);
        }
        let provider_request = ProviderGenerateRequest {
            model: self.model(),
            prompt: request.prompt,
            options: remote_generation_options(request.options),
            provider_options: request.remote_options,
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let join = executor.spawn(async move {
            transport
                .generate(provider_request)
                .await
                .map(|response| map::remote_text_response(endpoint, response))
                .map_err(CogentError::from)
        });
        CogentTextRun::new(
            Box::pin(RemoteResponseFuture::new(join, executor)),
            CogentTokenBatches::closed(),
        )
    }

    fn chat(&self, request: CogentChatRequest) -> CogentTextRun {
        if let Err(error) = validate::remote_chat(&request) {
            return CogentTextRun::ready_err(error);
        }
        let provider_request = ProviderChatRequest {
            model: self.model(),
            messages: request.messages,
            options: remote_generation_options(request.options),
            provider_options: request.remote_options,
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();

        if request.emit_tokens {
            let (batch_tx, batch_rx) = mpsc::unbounded();
            let join = executor.spawn(async move {
                run_remote_stream(transport, endpoint, provider_request, batch_tx).await
            });
            CogentTextRun::new(
                Box::pin(RemoteResponseFuture::new(join, executor)),
                CogentTokenBatches::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                transport
                    .chat(provider_request)
                    .await
                    .map(|response| map::remote_text_response(endpoint, response))
                    .map_err(CogentError::from)
            });
            CogentTextRun::new(
                Box::pin(RemoteResponseFuture::new(join, executor)),
                CogentTokenBatches::closed(),
            )
        }
    }

    fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun {
        if let Err(error) = validate::remote_embed(&request) {
            return CogentEmbeddingRun::ready_err(error);
        }
        let provider_request = ProviderEmbedRequest {
            model: self.model(),
            input: request.input,
            provider_options: request.remote_options,
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let join = executor.spawn(async move {
            transport
                .embed(provider_request)
                .await
                .map(|response| map::remote_embedding_response(endpoint, response))
                .map_err(CogentError::from)
        });
        CogentEmbeddingRun::new(Box::pin(RemoteResponseFuture::new(join, executor)))
    }
}

struct RemoteResponseFuture<T> {
    join: tokio::task::JoinHandle<CogentResult<T>>,
    _executor: RemoteExecutor,
}

impl<T> RemoteResponseFuture<T> {
    fn new(join: tokio::task::JoinHandle<CogentResult<T>>, executor: RemoteExecutor) -> Self {
        Self {
            join,
            _executor: executor,
        }
    }
}

impl<T> Future for RemoteResponseFuture<T> {
    type Output = CogentResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.join).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(error)) => Poll::Ready(Err(CogentError::Internal(format!(
                "remote task failed: {error}"
            )))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Drop for RemoteResponseFuture<T> {
    fn drop(&mut self) {
        self.join.abort();
    }
}

async fn run_remote_stream(
    transport: ProviderTransport,
    endpoint: EndpointRef,
    request: ProviderChatRequest,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
) -> CogentResult<CogentTextResponse> {
    let mut stream = transport.stream_chat(request).await?;
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut usage: Option<TokenUsage> = None;

    while let Some(event) = stream.next().await {
        match event? {
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

fn remote_generation_options(options: crate::CogentTextOptions) -> ProviderGenerationOptions {
    ProviderGenerationOptions {
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        stop: options.stop,
    }
}
