//! Tests the Swift UniFFI inference bridge.
//!
//! Covers request projection, independent token/response polling, one-shot
//! response ownership, and explicit cancellation with model-free core runs.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::channel::mpsc;
use sipp::core::{FinishReason, TokenBatch, TokenEmissionStats, TokenUsage};
use sipp::engine::{CacheSource, KvReuseMode, PoolingType, RequestStats};
use sipp::{
    EndpointRef, SippEmbeddingResponse, SippEmbeddingRun, SippResponseMetadata, SippTextResponse,
    SippTextRun, SippTokenBatches,
};

use super::*;
use crate::bridge::{FfiCancellationReason, FfiSippClient};

fn test_storage_root(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sipp-swift-{}-{name}", std::process::id()))
}

fn text_options() -> FfiTextOptions {
    FfiTextOptions {
        max_tokens: Some(32),
        temperature: Some(0.5),
        top_p: Some(0.9),
        stop: vec!["stop".to_owned()],
    }
}

fn local_text_options() -> FfiLocalTextOptions {
    FfiLocalTextOptions {
        context_key: Some("context".to_owned()),
        grammar: Some("root".to_owned()),
        json_schema: Some("{}".to_owned()),
        media: vec![vec![1, 2, 3]],
    }
}

fn request_stats() -> RequestStats {
    RequestStats {
        input_tokens: 4,
        output_tokens: 2,
        cache_mode: KvReuseMode::LiveSlotPrefix,
        cache_source: CacheSource::Live,
        cache_hits: 3,
        prefill_tokens: 1,
        ttft_ms: Some(1.5),
        inter_token_ms: Some(2.5),
        e2e_ms: Some(4.0),
        e2e_tokens_per_second: Some(500.0),
        decode_tokens_per_second: Some(400.0),
        prefill_tokens_per_second: Some(300.0),
        prefill_ms: 1.0,
        decode_ms: 2.0,
    }
}

fn text_response() -> SippTextResponse {
    SippTextResponse {
        endpoint: EndpointRef::from_id("chat"),
        text: "hello".to_owned(),
        finish_reason: FinishReason::Stop,
        usage: Some(TokenUsage {
            input_tokens: Some(4),
            output_tokens: Some(2),
            total_tokens: Some(6),
        }),
        local_stats: Some(request_stats()),
        metadata: SippResponseMetadata {
            request_id: Some("request".to_owned()),
            upstream_request_id: None,
            upstream_response_id: None,
        },
    }
}

fn token_batch() -> TokenBatch {
    TokenBatch {
        request_id: "request".to_owned(),
        stream_id: 7,
        sequence_start: 3,
        text: "hello".to_owned(),
        frame_count: 2,
        byte_count: 5,
        stats: TokenEmissionStats {
            frames_sent: 2,
            bytes_sent: 5,
            batches_sent: 1,
        },
    }
}

#[test]
fn request_projection_preserves_typed_swift_values() {
    let (context, query) = FfiQueryRequest {
        request_id: Some("query-request".to_owned()),
        endpoint: Some("query".to_owned()),
        prompt: "prompt".to_owned(),
        options: text_options(),
        local: local_text_options(),
        emit_tokens: true,
    }
    .into_core();

    assert_eq!(context.request_id.as_deref(), Some("query-request"));
    assert_eq!(query.endpoint.as_ref().map(EndpointRef::id), Some("query"));
    assert_eq!(query.prompt, "prompt");
    assert_eq!(query.options.max_tokens, Some(32));
    assert_eq!(query.local.media, vec![vec![1, 2, 3]]);
    assert!(query.emit_tokens);

    let (_, chat) = FfiChatRequest {
        request_id: None,
        endpoint: Some("chat".to_owned()),
        messages: vec![FfiChatMessage {
            role: FfiChatRole::User,
            content: "hello".to_owned(),
        }],
        options: text_options(),
        local: local_text_options(),
        emit_tokens: true,
    }
    .into_core();
    assert_eq!(chat.messages[0].role, CoreChatRole::User);
    assert_eq!(chat.messages[0].content, "hello");

    let (_, embed) = FfiEmbedRequest {
        request_id: None,
        endpoint: Some("embed".to_owned()),
        input: "vectorize".to_owned(),
        local: FfiLocalEmbedOptions {
            context_key: Some("embedding".to_owned()),
            normalize: Some(true),
        },
    }
    .into_core();
    assert_eq!(embed.endpoint.as_ref().map(EndpointRef::id), Some("embed"));
    assert_eq!(embed.local.context_key.as_deref(), Some("embedding"));
    assert_eq!(embed.local.normalize, Some(true));

    let (_, listen) = FfiListenRequest {
        request_id: Some("listen-request".to_owned()),
        endpoint: Some("listen".to_owned()),
        audio: vec![1, 2, 3],
        language: Some("en".to_owned()),
        max_tokens: Some(96),
    }
    .into_core();
    assert_eq!(
        listen.endpoint.as_ref().map(EndpointRef::id),
        Some("listen")
    );
    assert_eq!(listen.audio, vec![1, 2, 3]);
    assert_eq!(listen.language.as_deref(), Some("en"));
    assert_eq!(listen.max_tokens, Some(96));

    let (_, speak) = FfiSpeakRequest {
        request_id: Some("speak-request".to_owned()),
        endpoint: Some("speak".to_owned()),
        text: "hello".to_owned(),
        language: Some("en".to_owned()),
        speaker_audio: Some(vec![4, 5, 6]),
        max_duration_ms: Some(2_000),
    }
    .into_core();
    assert_eq!(speak.endpoint.as_ref().map(EndpointRef::id), Some("speak"));
    assert_eq!(speak.text, "hello");
    assert_eq!(speak.language.as_deref(), Some("en"));
    assert_eq!(speak.speaker_audio, Some(vec![4, 5, 6]));
    assert_eq!(speak.max_duration_ms, Some(2_000));
}

#[test]
fn text_run_polls_tokens_and_response_independently() {
    let (sender, receiver) = mpsc::unbounded();
    let run = SippTextRun::from_parts(
        SippTokenBatches::from_stream(Box::pin(receiver)),
        Box::pin(async { Ok(text_response()) }),
    );
    let run = Arc::new(FfiTextRun::from_core(run));
    let foreign_executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    foreign_executor.block_on(async {
        let token_run = Arc::clone(&run);
        let token_task = tokio::spawn(async move { token_run.next_token().await });

        let response = tokio::time::timeout(Duration::from_secs(1), run.take_response())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.endpoint, "chat");
        assert_eq!(response.text, "hello");
        assert_eq!(response.finish_reason, FfiFinishReason::Stop);
        assert_eq!(response.usage.unwrap().total_tokens, Some(6));
        assert_eq!(
            response.local_stats.unwrap().cache_source,
            FfiCacheSource::Live
        );

        sender.unbounded_send(token_batch()).unwrap();
        let batch = tokio::time::timeout(Duration::from_secs(1), token_task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(batch.text, "hello");
        assert_eq!(batch.sequence_start, 3);
        assert_eq!(batch.stats.batches_sent, 1);

        let error = run.take_response().await.unwrap_err();
        assert!(matches!(error, FfiError::InvalidState { .. }));
    });
}

#[test]
fn cancellation_maps_to_a_typed_bridge_error() {
    let text_run = SippTextRun::from_response(Box::pin(futures::future::pending()));
    let text_run = FfiTextRun::from_core(text_run);
    let embedding_run = SippEmbeddingRun::from_response(Box::pin(futures::future::pending()));
    let embedding_run = FfiEmbeddingRun::from_core(embedding_run);
    let foreign_executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    text_run.cancel();
    embedding_run.cancel();
    for error in [
        foreign_executor
            .block_on(text_run.take_response())
            .unwrap_err(),
        foreign_executor
            .block_on(embedding_run.take_response())
            .unwrap_err(),
    ] {
        assert!(matches!(
            error,
            FfiError::Cancelled {
                reason: FfiCancellationReason::CallerCancelled
            }
        ));
    }
}

#[test]
fn embedding_response_and_client_selection_errors_are_typed() {
    let embedding = SippEmbeddingRun::from_response(Box::pin(async {
        Ok(SippEmbeddingResponse {
            endpoint: EndpointRef::from_id("embed"),
            values: vec![0.25, 0.75],
            usage: None,
            local_stats: Some(request_stats()),
            pooling: Some(PoolingType::Mean),
            normalized: Some(true),
            metadata: SippResponseMetadata::default(),
        })
    }));
    let embedding = FfiEmbeddingRun::from_core(embedding);
    let storage_root = test_storage_root("inference-client");
    let client = FfiSippClient::new(storage_root.display().to_string(), None).unwrap();
    let foreign_executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    foreign_executor.block_on(async {
        let response = embedding.take_response().await.unwrap();
        assert_eq!(response.values, vec![0.25, 0.75]);
        assert_eq!(response.pooling, Some(FfiPoolingType::Mean));
        assert_eq!(response.normalized, Some(true));

        let run = client
            .query(FfiQueryRequest {
                request_id: None,
                endpoint: None,
                prompt: "prompt".to_owned(),
                options: text_options(),
                local: local_text_options(),
                emit_tokens: true,
            })
            .await;
        let error = run.take_response().await.unwrap_err();
        assert!(matches!(error, FfiError::EndpointSelection { .. }));
    });

    drop(client);
    std::fs::remove_dir_all(storage_root).unwrap();
}
