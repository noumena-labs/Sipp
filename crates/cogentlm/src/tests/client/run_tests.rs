use crate::core::{TokenBatch, TokenEmissionStats};
use futures::{stream, StreamExt};

use crate::client::{CogentCancellationReason, CogentError, CogentTextRun, CogentTokenBatches};

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
    let mut batches = CogentTokenBatches::from_stream(Box::pin(stream::iter([batch.clone()])));
    let received = futures::executor::block_on(batches.next()).expect("token batch");
    assert_eq!(received, batch);
}

#[test]
fn cancelling_a_gateway_response_future_returns_cancelled() {
    let run = CogentTextRun::from_response(Box::pin(futures::future::pending()));
    run.cancel(CogentCancellationReason::CallerCancelled);
    assert!(matches!(
        futures::executor::block_on(run),
        Err(CogentError::Cancelled {
            reason: CogentCancellationReason::CallerCancelled
        })
    ));
}
