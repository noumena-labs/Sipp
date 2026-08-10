//! Tests shared cancellation ownership and public run result types.

use crate::core::{TokenBatch, TokenEmissionStats};
use futures::{stream, StreamExt};

use crate::client::{
    EndpointRef, SippAudioResponse, SippAudioRun, SippCancellationReason, SippEmbeddingRun,
    SippError, SippResponseMetadata, SippTextRun, SippTokenBatches,
};
use crate::engine::EngineCancellation;

#[test]
fn gateway_token_streams_are_exposed_without_transport_ownership() {
    let batch = TokenBatch {
        request_id: "request".to_string(),
        stream_id: 0,
        sequence_start: 0,
        text: "hello".to_string(),
        frame_count: 1,
        byte_count: 5,
        stats: TokenEmissionStats::default(),
    };
    let mut batches = SippTokenBatches::from_stream(Box::pin(stream::iter([batch.clone()])));
    let received = futures::executor::block_on(batches.next()).expect("token batch");
    assert_eq!(received, batch);
}

#[test]
fn cancelling_a_gateway_response_future_returns_cancelled() {
    let run = SippTextRun::from_response(Box::pin(futures::future::pending()));
    run.cancel(SippCancellationReason::CallerCancelled);
    assert!(matches!(
        futures::executor::block_on(run),
        Err(SippError::Cancelled {
            reason: SippCancellationReason::CallerCancelled
        })
    ));
}

#[test]
fn embedding_runs_keep_shared_cancellation_behavior() {
    let run = SippEmbeddingRun::from_response(Box::pin(futures::future::pending()));
    let cancellation = run.cancellation_handle();
    cancellation.cancel(SippCancellationReason::DeadlineExceeded);

    assert!(matches!(
        futures::executor::block_on(run),
        Err(SippError::Cancelled {
            reason: SippCancellationReason::DeadlineExceeded
        })
    ));
}

#[test]
fn audio_runs_complete_with_the_owned_wav_response() {
    let run = SippAudioRun::from_response(Box::pin(async {
        Ok(SippAudioResponse {
            endpoint: EndpointRef::from_id("speaker"),
            audio: vec![82, 73, 70, 70],
            sample_rate_hz: 24_000,
            channels: 1,
            duration_ms: 250,
            metadata: SippResponseMetadata::default(),
        })
    }));

    assert_eq!(
        futures::executor::block_on(run).expect("audio response"),
        SippAudioResponse {
            endpoint: EndpointRef::from_id("speaker"),
            audio: vec![82, 73, 70, 70],
            sample_rate_hz: 24_000,
            channels: 1,
            duration_ms: 250,
            metadata: SippResponseMetadata::default(),
        }
    );
}

#[test]
fn audio_runs_share_cancellation_and_ready_error_ownership() {
    let run = SippAudioRun::from_response(Box::pin(futures::future::pending()));
    run.cancel(SippCancellationReason::CallerCancelled);
    assert!(matches!(
        futures::executor::block_on(run),
        Err(SippError::Cancelled {
            reason: SippCancellationReason::CallerCancelled
        })
    ));

    let run = SippAudioRun::ready_err(SippError::InvalidRequest("invalid audio".to_string()));
    assert!(matches!(
        futures::executor::block_on(run),
        Err(SippError::InvalidRequest(message)) if message == "invalid audio"
    ));
}

#[test]
fn audio_cancellation_reaches_the_engine_before_response_polling() {
    let engine_cancellation = EngineCancellation::new();
    let run = SippAudioRun::new_with_engine_cancellation(
        Box::pin(futures::future::pending()),
        engine_cancellation.clone(),
    );

    run.cancel(SippCancellationReason::CallerCancelled);

    assert!(engine_cancellation.is_cancelled());
}

#[test]
fn explicit_audio_cancellation_wins_over_a_ready_response() {
    let response = SippAudioResponse {
        endpoint: EndpointRef::from_id("speaker"),
        audio: vec![82, 73, 70, 70],
        sample_rate_hz: 24_000,
        channels: 1,
        duration_ms: 250,
        metadata: SippResponseMetadata::default(),
    };
    let run = SippAudioRun::new_with_engine_cancellation(
        Box::pin(async move { Ok(response) }),
        EngineCancellation::new(),
    );

    run.cancel(SippCancellationReason::CallerCancelled);

    assert!(matches!(
        futures::executor::block_on(run),
        Err(SippError::Cancelled {
            reason: SippCancellationReason::CallerCancelled
        })
    ));
}
