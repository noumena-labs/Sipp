use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use cogentlm_core::{FinishReason, TokenBatch, TokenUsage};
use cogentlm_providers::{
    ProviderChatRequest, ProviderClient, ProviderEmbedRequest, ProviderGenerateRequest,
    ProviderGenerationOptions, ProviderStreamEvent,
};
use futures::StreamExt;
use futures_channel::mpsc;

use crate::dispatch::InferenceEndpoint;
use crate::remote_executor::RemoteExecutor;
use crate::run::TOKEN_STREAM_CHANNEL_CAPACITY;
use crate::{
    map, validate, CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError,
    CogentQueryRequest, CogentResult, CogentTextResponse, CogentTextRun, CogentTokenStream,
    EndpointCapabilities, EndpointRef,
};

pub(crate) struct RemoteEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    model: String,
    client: ProviderClient,
    executor: RemoteExecutor,
}

impl RemoteEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        model: String,
        capabilities: EndpointCapabilities,
        client: ProviderClient,
        executor: RemoteExecutor,
    ) -> Self {
        Self {
            endpoint,
            capabilities,
            model,
            client,
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
        if request.stream_tokens {
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
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let join = executor.spawn(async move {
            client
                .generate(provider_request)
                .await
                .map(|response| map::remote_text_response(endpoint, response))
                .map_err(CogentError::from)
        });
        CogentTextRun::new(
            Box::pin(RemoteResponseFuture::new(join, executor)),
            CogentTokenStream::closed(),
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
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();

        if request.stream_tokens {
            let (batch_tx, batch_rx) = mpsc::channel(TOKEN_STREAM_CHANNEL_CAPACITY);
            let join = executor.spawn(async move {
                run_remote_stream(client, endpoint, provider_request, batch_tx).await
            });
            CogentTextRun::new(
                Box::pin(RemoteResponseFuture::new(join, executor)),
                CogentTokenStream::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                client
                    .chat(provider_request)
                    .await
                    .map(|response| map::remote_text_response(endpoint, response))
                    .map_err(CogentError::from)
            });
            CogentTextRun::new(
                Box::pin(RemoteResponseFuture::new(join, executor)),
                CogentTokenStream::closed(),
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
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let executor = self.executor.clone();
        let join = executor.spawn(async move {
            client
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
    client: ProviderClient,
    endpoint: EndpointRef,
    request: ProviderChatRequest,
    mut batch_tx: mpsc::Sender<TokenBatch>,
) -> CogentResult<CogentTextResponse> {
    let mut stream = client.stream_chat(request).await?;
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut usage: Option<TokenUsage> = None;
    let mut pending_dropped_frames = 0_u64;

    while let Some(event) = stream.next().await {
        match event? {
            ProviderStreamEvent::TokenBatch(mut batch) => {
                text.push_str(&batch.text);
                if pending_dropped_frames > 0 {
                    batch.stats.frames_dropped = batch
                        .stats
                        .frames_dropped
                        .saturating_add(pending_dropped_frames);
                    pending_dropped_frames = 0;
                }
                if let Err(error) = batch_tx.try_send(batch) {
                    if error.is_full() {
                        pending_dropped_frames = pending_dropped_frames
                            .saturating_add(u64::from(error.into_inner().frame_count));
                    }
                }
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
