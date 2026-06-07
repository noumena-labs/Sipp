use std::sync::{Arc, Mutex};

use cogentlm_client::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingResponse, CogentEmbeddingRun,
    CogentQueryRequest, CogentResponseMetadata, CogentTextResponse, CogentTextRun,
    CogentTokenBatches, EndpointRef,
};
use cogentlm_core::{FinishReason, TokenBatch, TokenEmissionStats};
use futures_util::{stream, StreamExt};

use super::{
    AdmissionController, AdmissionPermit, Authorizer, GatewayExecutor, GatewayPipeline, Operation,
    TargetResolver,
};
use crate::{
    GatewayCancellationReason, GatewayError, GatewayErrorKind, GatewayRequestContext,
    GatewayResult, GatewayStreamEvent,
};

#[tokio::test]
async fn pipeline_orders_policy_before_execution_and_releases_permit() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let drops = Arc::new(Mutex::new(0));
    let pipeline = pipeline(events.clone(), drops.clone(), false);
    let response = pipeline
        .query(
            &GatewayRequestContext::new(Some("request".to_string())),
            "public",
            CogentQueryRequest::default(),
        )
        .await
        .expect("query");

    assert_eq!(response.text, "ok");
    assert_eq!(
        *events.lock().expect("events"),
        ["resolve", "authorize", "admit", "execute"]
    );
    assert_eq!(*drops.lock().expect("drops"), 1);
}

#[tokio::test]
async fn authorization_stops_admission_and_execution() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let drops = Arc::new(Mutex::new(0));
    let pipeline = pipeline(events.clone(), drops, true);
    let error = pipeline
        .chat(
            &GatewayRequestContext::default(),
            "public",
            CogentChatRequest::default(),
        )
        .await
        .expect_err("authorization");

    assert_eq!(error.kind, GatewayErrorKind::Authorization);
    assert_eq!(*events.lock().expect("events"), ["resolve", "authorize"]);
}

#[tokio::test]
async fn streaming_holds_admission_until_terminal_event() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let drops = Arc::new(Mutex::new(0));
    let pipeline = pipeline(events, drops.clone(), false);
    let mut stream = pipeline
        .stream_query(
            &GatewayRequestContext::default(),
            "public",
            CogentQueryRequest::default(),
        )
        .expect("stream");

    assert_eq!(*drops.lock().expect("drops"), 0);
    assert!(matches!(
        stream.next().await,
        Some(Ok(GatewayStreamEvent::TokenBatch(_)))
    ));
    while stream.next().await.is_some() {}
    assert_eq!(*drops.lock().expect("drops"), 1);
}

#[tokio::test]
async fn cancellation_reaches_the_active_client_run() {
    let context = GatewayRequestContext::default();
    let pipeline = GatewayPipeline::new(
        Arc::new(Resolver {
            events: Arc::new(Mutex::new(Vec::new())),
        }),
        Arc::new(Policy {
            events: Arc::new(Mutex::new(Vec::new())),
            deny: false,
        }),
        Arc::new(Admission {
            events: Arc::new(Mutex::new(Vec::new())),
            drops: Arc::new(Mutex::new(0)),
        }),
        Arc::new(PendingExecutor),
    );
    let task_context = context.clone();
    let task = tokio::spawn(async move {
        pipeline
            .query(&task_context, "public", CogentQueryRequest::default())
            .await
    });
    tokio::task::yield_now().await;
    context
        .cancellation
        .cancel(GatewayCancellationReason::CallerCancelled);
    let error = task.await.expect("task").expect_err("cancelled");
    assert_eq!(error.kind, GatewayErrorKind::Cancelled);
}

fn pipeline(
    events: Arc<Mutex<Vec<&'static str>>>,
    drops: Arc<Mutex<u32>>,
    deny: bool,
) -> GatewayPipeline {
    GatewayPipeline::new(
        Arc::new(Resolver {
            events: events.clone(),
        }),
        Arc::new(Policy {
            events: events.clone(),
            deny,
        }),
        Arc::new(Admission {
            events: events.clone(),
            drops,
        }),
        Arc::new(Executor { events }),
    )
}

struct Resolver {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl TargetResolver for Resolver {
    fn resolve(
        &self,
        _context: &GatewayRequestContext,
        target: &str,
        _operation: Operation,
    ) -> GatewayResult<EndpointRef> {
        self.events.lock().expect("events").push("resolve");
        if target == "public" {
            Ok(EndpointRef::gateway("resolved"))
        } else {
            Err(GatewayError::new(
                GatewayErrorKind::Resolution,
                "missing target",
            ))
        }
    }
}

struct Policy {
    events: Arc<Mutex<Vec<&'static str>>>,
    deny: bool,
}

impl Authorizer for Policy {
    fn authorize(
        &self,
        _context: &GatewayRequestContext,
        _target: &str,
        _endpoint: &EndpointRef,
        _operation: Operation,
    ) -> GatewayResult<()> {
        self.events.lock().expect("events").push("authorize");
        if self.deny {
            Err(GatewayError::new(GatewayErrorKind::Authorization, "denied"))
        } else {
            Ok(())
        }
    }
}

struct Admission {
    events: Arc<Mutex<Vec<&'static str>>>,
    drops: Arc<Mutex<u32>>,
}

impl AdmissionController for Admission {
    fn acquire(
        &self,
        _context: &GatewayRequestContext,
        _target: &str,
        _endpoint: &EndpointRef,
        _operation: Operation,
    ) -> GatewayResult<Box<dyn AdmissionPermit>> {
        self.events.lock().expect("events").push("admit");
        Ok(Box::new(Permit {
            drops: self.drops.clone(),
        }))
    }
}

struct Permit {
    drops: Arc<Mutex<u32>>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        *self.drops.lock().expect("drops") += 1;
    }
}

struct Executor {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl GatewayExecutor for Executor {
    fn query(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentQueryRequest,
    ) -> CogentTextRun {
        self.events.lock().expect("events").push("execute");
        let batch = TokenBatch {
            request_id: "request".to_string(),
            stream_id: 0,
            sequence_start: 0,
            text: "ok".to_string(),
            frame_count: 1,
            byte_count: 2,
            stats: TokenEmissionStats::default(),
        };
        CogentTextRun::from_parts(
            CogentTokenBatches::from_stream(Box::pin(stream::iter([batch]))),
            Box::pin(async { Ok(text_response()) }),
        )
    }

    fn chat(&self, context: &GatewayRequestContext, request: CogentChatRequest) -> CogentTextRun {
        self.query(
            context,
            CogentQueryRequest {
                endpoint: request.endpoint,
                ..CogentQueryRequest::default()
            },
        )
    }

    fn embed(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        CogentEmbeddingRun::from_response(Box::pin(async {
            Ok(CogentEmbeddingResponse {
                endpoint: EndpointRef::gateway("resolved"),
                values: vec![1.0],
                usage: None,
                local_stats: None,
                pooling: None,
                normalized: None,
                metadata: CogentResponseMetadata::default(),
            })
        }))
    }
}

struct PendingExecutor;

impl GatewayExecutor for PendingExecutor {
    fn query(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentQueryRequest,
    ) -> CogentTextRun {
        CogentTextRun::from_response(Box::pin(futures_util::future::pending()))
    }

    fn chat(&self, context: &GatewayRequestContext, _request: CogentChatRequest) -> CogentTextRun {
        self.query(context, CogentQueryRequest::default())
    }

    fn embed(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        CogentEmbeddingRun::from_response(Box::pin(futures_util::future::pending()))
    }
}

fn text_response() -> CogentTextResponse {
    CogentTextResponse {
        endpoint: EndpointRef::gateway("resolved"),
        text: "ok".to_string(),
        finish_reason: FinishReason::Stop,
        usage: None,
        local_stats: None,
        metadata: CogentResponseMetadata::default(),
    }
}
