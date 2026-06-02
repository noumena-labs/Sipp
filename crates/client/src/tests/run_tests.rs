//! Tests the `run` module in `cogentlm-client`.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

use cogentlm_core::{FinishReason, TokenBatch, TokenEmissionStats};
use futures::executor::block_on;
use futures::StreamExt;

use super::*;
use crate::EndpointRef;

fn endpoint() -> EndpointRef {
    EndpointRef::Remote {
        id: "remote".to_string(),
    }
}

fn token_batch(text: &str) -> TokenBatch {
    TokenBatch {
        request_id: "req".to_string(),
        stream_id: 1,
        sequence_start: 0,
        text: text.to_string(),
        frame_count: 1,
        byte_count: text.len() as u32,
        stats: TokenEmissionStats {
            frames_sent: 1,
            bytes_sent: text.len() as u64,
            batches_sent: 1,
        },
    }
}

#[test]
fn ready_text_error_closes_token_stream() {
    let run = CogentTextRun::ready_err(CogentError::Internal("boom".to_string()));
    let (mut tokens, response) = run.into_parts();

    let error = block_on(response).expect_err("ready text error");
    assert!(matches!(error, CogentError::Internal(message) if message == "boom"));
    assert!(block_on(tokens.next()).is_none());
}

#[test]
fn embedding_ready_error_is_awaitable() {
    let run = CogentEmbeddingRun::ready_err(CogentError::Internal("embed boom".to_string()));

    let error = block_on(run.into_response()).expect_err("ready embedding error");

    assert!(matches!(
        error,
        CogentError::Internal(message) if message == "embed boom"
    ));
}

#[test]
fn text_run_splits_response_and_tokens() {
    let run = CogentTextRun::new(
        Box::pin(async {
            Ok(CogentTextResponse {
                endpoint: endpoint(),
                text: "done".to_string(),
                finish_reason: FinishReason::Stop,
                usage: None,
                local_stats: None,
            })
        }),
        CogentTokenBatches::closed(),
    );

    let (mut tokens, response) = run.into_parts();
    let response = block_on(response).expect("text response");

    assert_eq!(response.text, "done");
    assert!(block_on(tokens.next()).is_none());
}

#[cfg(feature = "providers")]
#[test]
fn receiver_token_stream_yields_batches_until_closed() {
    let (tx, rx) = futures_channel::mpsc::unbounded();
    tx.unbounded_send(token_batch("a")).expect("send first");
    tx.unbounded_send(token_batch("b")).expect("send second");
    drop(tx);

    let mut tokens = CogentTokenBatches::from_receiver(rx);

    assert_eq!(block_on(tokens.next()).expect("first").text, "a");
    assert_eq!(block_on(tokens.next()).expect("second").text, "b");
    assert!(block_on(tokens.next()).is_none());
}

#[test]
fn embedding_run_resolves_response_future() {
    let run = CogentEmbeddingRun::new(Box::pin(async {
        Ok(CogentEmbeddingResponse {
            endpoint: endpoint(),
            values: vec![1.0, 2.0],
            usage: None,
            local_stats: None,
            pooling: None,
            normalized: None,
        })
    }));

    let response = block_on(run).expect("embedding response");

    assert_eq!(response.values, vec![1.0, 2.0]);
}
