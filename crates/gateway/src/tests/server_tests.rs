use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body},
    http::{
        header::{
            ACCESS_CONTROL_ALLOW_ORIGIN, ACCESS_CONTROL_EXPOSE_HEADERS,
            ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, AUTHORIZATION, ORIGIN,
            VARY,
        },
        Request, StatusCode,
    },
    Router,
};
use serde_json::json;
use tokio::sync::Notify;
use tower::ServiceExt;

use super::constant_time_eq;
use crate::{
    BackendChatRequest, BackendEmbedRequest, BackendEmbeddingOutput, BackendQueryRequest,
    BackendTextOutput, GatewayAccess, GatewayAlias, GatewayAliasLimits, GatewayBackend,
    GatewayError, GatewayErrorKind, GatewayResult, GatewayService, GatewayServiceLimits,
    GatewayState, GatewayStream, GatewayStreamEvent, GatewayToken, MockBackend, Operation,
    OperationSet,
};

fn test_service() -> GatewayService {
    GatewayService::new(test_state(), Vec::new(), GatewayServiceLimits::default())
}

#[test]
fn gateway_token_debug_redacts_secret() {
    let token = GatewayToken::new("secret-token", GatewayAccess::all());

    let debug = format!("{token:?}");

    assert!(!debug.contains("secret-token"));
    assert!(debug.contains("[redacted]"));
}

#[test]
fn bearer_token_compare_rejects_mismatch_and_prefix() {
    assert!(constant_time_eq(b"secret-token", b"secret-token"));
    assert!(!constant_time_eq(b"secret-token", b"secret-other"));
    assert!(!constant_time_eq(b"secret-token", b"secret"));
    assert!(!constant_time_eq(b"secret", b"secret-token"));
}

#[tokio::test]
async fn query_requires_bearer_token() {
    let response = test_service()
        .router()
        .expect("router")
        .oneshot(
            Request::post("/v1/query")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "mock",
                        "prompt": "hello"
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "authentication");
}

#[tokio::test]
async fn query_routes_to_alias_backend() {
    let response = authed_post(
        "/v1/query",
        json!({
            "model": "mock",
            "prompt": "hello",
            "max_tokens": 8
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["model"], "mock");
    assert_eq!(body["text"], "answer: hello");
    assert_eq!(body["finish_reason"], "stop");
    assert_eq!(body["usage"]["input_tokens"], 1);
}

#[tokio::test]
async fn successful_response_includes_gateway_request_id() {
    let response = authed_post(
        "/v1/query",
        json!({
            "model": "mock",
            "prompt": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("request id")
        .to_string();
    assert!(request_id.starts_with("gw_"));
    let body = response_json(response).await;
    assert_eq!(body["id"], request_id);
}

#[tokio::test]
async fn error_response_includes_gateway_request_id() {
    let response = test_service()
        .router()
        .expect("router")
        .oneshot(
            Request::post("/v1/query")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "mock",
                        "prompt": "hello"
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("request id");
    assert!(request_id.starts_with("gw_"));
}

#[tokio::test]
async fn unsupported_alias_operation_is_normalized() {
    let response = authed_post(
        "/v1/query",
        json!({
            "model": "chat-only",
            "prompt": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "unsupported_feature");
    assert_eq!(
        body["error"]["message"],
        "model alias does not support query"
    );
}

#[tokio::test]
async fn stream_query_emits_normalized_sse() {
    let response = authed_post(
        "/v1/query",
        json!({
            "model": "mock",
            "prompt": "stream",
            "stream": true
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_text(response).await;
    assert!(body.contains("event: token"));
    assert!(body.contains(r#""text":"answer: stream""#));
    assert!(body.contains(r#""sequence":0"#));
    assert!(body.contains("event: done"));
    assert!(body.contains(r#"data: {"finish_reason":"stop"}"#));
}

#[tokio::test]
async fn cors_preflight_allows_configured_browser_origin() {
    let response = GatewayService::new(
        test_state(),
        vec!["https://app.example".to_string()],
        GatewayServiceLimits::default(),
    )
    .router()
    .expect("router")
    .oneshot(
        Request::options("/v1/query")
            .header(ORIGIN, "https://app.example")
            .header(ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(ACCESS_CONTROL_REQUEST_HEADERS, "authorization,content-type")
            .body(Body::empty())
            .expect("request"),
    )
    .await
    .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&"https://app.example".parse().expect("origin header"))
    );
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("preflight request id");
    assert!(request_id.starts_with("gw_"));
    let vary = response
        .headers()
        .get(VARY)
        .and_then(|value| value.to_str().ok())
        .expect("vary header");
    assert!(vary.to_ascii_lowercase().contains("origin"));

    let response = GatewayService::new(
        test_state(),
        vec!["https://app.example".to_string()],
        GatewayServiceLimits::default(),
    )
    .router()
    .expect("router")
    .oneshot(
        Request::post("/v1/query")
            .header("content-type", "application/json")
            .header(AUTHORIZATION, "Bearer test-token")
            .header(ORIGIN, "https://app.example")
            .body(Body::from(
                json!({
                    "model": "mock",
                    "prompt": "hello"
                })
                .to_string(),
            ))
            .expect("request"),
    )
    .await
    .expect("response");
    let exposed = response
        .headers()
        .get(ACCESS_CONTROL_EXPOSE_HEADERS)
        .and_then(|value| value.to_str().ok())
        .expect("expose headers");
    assert!(exposed.contains("x-request-id"));
    assert!(exposed.contains("retry-after"));
    assert!(exposed.contains("retry-after-ms"));
}

#[tokio::test]
async fn cors_preflight_does_not_allow_unconfigured_browser_origin() {
    let response = GatewayService::new(
        test_state(),
        vec!["https://app.example".to_string()],
        GatewayServiceLimits::default(),
    )
    .router()
    .expect("router")
    .oneshot(
        Request::options("/v1/query")
            .header(ORIGIN, "https://evil.example")
            .header(ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(ACCESS_CONTROL_REQUEST_HEADERS, "authorization,content-type")
            .body(Body::empty())
            .expect("request"),
    )
    .await
    .expect("response");

    assert!(response
        .headers()
        .get(ACCESS_CONTROL_ALLOW_ORIGIN)
        .is_none());
}

#[test]
fn cors_config_accepts_https_and_loopback_http_origins() {
    for origin in [
        "https://app.example",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://[::1]:5173",
    ] {
        match GatewayService::new(
            test_state(),
            vec![origin.to_string()],
            GatewayServiceLimits::default(),
        )
        .router()
        {
            Ok(_) => {}
            Err(error) => panic!("{origin} should be accepted: {error}"),
        }
    }
}

#[test]
fn cors_config_rejects_wildcard_path_and_non_loopback_http_origins() {
    for (origin, message) in [
        ("*", "CORS origin must be an exact application origin"),
        ("null", "CORS origin must be an exact application origin"),
        (
            "https://app.example/path",
            "CORS origin must not include a path, query, or fragment",
        ),
        (
            "http://app.example",
            "CORS origin must use HTTPS unless it targets loopback",
        ),
    ] {
        let error = match GatewayService::new(
            test_state(),
            vec![origin.to_string()],
            GatewayServiceLimits::default(),
        )
        .router()
        {
            Ok(_) => panic!("{origin} should be rejected"),
            Err(error) => error,
        };

        assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
        assert_eq!(error.message, message);
    }
}

#[tokio::test]
async fn scoped_token_rejects_unauthorized_operation() {
    let mut state = GatewayState::with_tokens([GatewayToken::new(
        "scoped-token",
        GatewayAccess::new([("mock".to_string(), OperationSet::new([Operation::Query]))]),
    )]);
    state
        .add_alias(GatewayAlias::new(
            "mock",
            OperationSet::all(),
            Arc::new(MockBackend::new("answer: ", 3)),
            GatewayAliasLimits::default(),
        ))
        .expect("add mock alias");
    let response = post(
        GatewayService::new(state, Vec::new(), GatewayServiceLimits::default())
            .router()
            .expect("router"),
        "/v1/embed",
        "scoped-token",
        json!({
            "model": "mock",
            "input": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "authorization");
}

#[tokio::test]
async fn request_size_limit_returns_normalized_error() {
    let response = post(
        GatewayService::new(
            test_state(),
            Vec::new(),
            GatewayServiceLimits {
                max_request_bytes: 16,
            },
        )
        .router()
        .expect("router"),
        "/v1/query",
        "test-token",
        json!({
            "model": "mock",
            "prompt": "body larger than the configured limit"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "request_too_large");
}

#[tokio::test]
async fn duplicate_typed_fields_are_rejected_before_gateway_options() {
    let response = post_raw(
        test_service().router().expect("router"),
        "/v1/query",
        "test-token",
        r#"{
            "model": "mock",
            "prompt": "first",
            "prompt": "override"
        }"#,
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn gateway_requests_reject_local_only_fields() {
    let query = authed_post(
        "/v1/query",
        json!({
            "model": "mock",
            "prompt": "hello",
            "grammar": "root ::= \"ok\""
        }),
    )
    .await;

    assert_eq!(query.status(), StatusCode::BAD_REQUEST);
    let body = response_json(query).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(
        body["error"]["message"],
        "gateway request cannot contain local-only field: grammar"
    );

    let embed = authed_post(
        "/v1/embed",
        json!({
            "model": "mock",
            "input": "hello",
            "normalize": true
        }),
    )
    .await;

    assert_eq!(embed.status(), StatusCode::BAD_REQUEST);
    let body = response_json(embed).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(
        body["error"]["message"],
        "gateway request cannot contain local-only field: normalize"
    );
}

#[tokio::test]
async fn text_option_ranges_are_rejected_before_backend_calls() {
    let response = authed_post(
        "/v1/query",
        json!({
            "model": "mock",
            "prompt": "hello",
            "temperature": -0.1
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(
        body["error"]["message"],
        "temperature must be greater than or equal to zero"
    );

    let response = authed_post(
        "/v1/chat",
        json!({
            "model": "mock",
            "messages": [{ "role": "user", "content": "hello" }],
            "top_p": 1.1
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["message"], "top_p must be between 0 and 1");
}

#[tokio::test]
async fn rate_limit_returns_retry_after() {
    let response = limited_alias_second_response(GatewayAliasLimits {
        max_requests_per_minute: Some(1),
        ..GatewayAliasLimits::default()
    })
    .await;

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(response.headers().get("retry-after").is_some());
    assert!(response.headers().get("retry-after-ms").is_some());
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "rate_limited");
}

#[tokio::test]
async fn quota_limit_returns_payment_required() {
    let response = limited_alias_second_response(GatewayAliasLimits {
        max_requests_total: Some(1),
        ..GatewayAliasLimits::default()
    })
    .await;

    assert_eq!(response.status(), StatusCode::PAYMENT_REQUIRED);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "quota_exceeded");
}

#[tokio::test]
async fn concurrency_limit_rejects_overlapping_request() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let backend = Arc::new(BlockingBackend {
        started: started.clone(),
        release: release.clone(),
    });
    let mut state = GatewayState::new("test-token");
    state
        .add_alias(GatewayAlias::new(
            "limited",
            OperationSet::new([Operation::Query]),
            backend,
            GatewayAliasLimits {
                max_concurrent_requests: Some(1),
                ..GatewayAliasLimits::default()
            },
        ))
        .expect("add limited alias");
    let router = GatewayService::new(state, Vec::new(), GatewayServiceLimits::default())
        .router()
        .expect("router");

    let first_router = router.clone();
    let first = tokio::spawn(async move {
        post(
            first_router,
            "/v1/query",
            "test-token",
            json!({
                "model": "limited",
                "prompt": "first"
            }),
        )
        .await
    });
    started.notified().await;

    let second = post(
        router,
        "/v1/query",
        "test-token",
        json!({
            "model": "limited",
            "prompt": "second"
        }),
    )
    .await;
    assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_json(second).await;
    assert_eq!(body["error"]["code"], "overloaded");

    release.notify_one();
    assert_eq!(first.await.expect("join").status(), StatusCode::OK);
}

async fn authed_post(path: &str, body: serde_json::Value) -> axum::response::Response {
    post(
        test_service().router().expect("router"),
        path,
        "test-token",
        body,
    )
    .await
}

async fn post(
    router: Router,
    path: &str,
    token: &str,
    body: serde_json::Value,
) -> axum::response::Response {
    post_raw(router, path, token, &body.to_string()).await
}

async fn post_raw(router: Router, path: &str, token: &str, body: &str) -> axum::response::Response {
    router
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response")
}

fn test_state() -> GatewayState {
    let mut state = GatewayState::new("test-token");
    state
        .add_alias(GatewayAlias::new(
            "mock",
            OperationSet::all(),
            Arc::new(MockBackend::new("answer: ", 3)),
            GatewayAliasLimits::default(),
        ))
        .expect("add mock alias");
    state
        .add_alias(GatewayAlias::new(
            "chat-only",
            OperationSet::new([Operation::Chat]),
            Arc::new(MockBackend::new("chat: ", 3)),
            GatewayAliasLimits::default(),
        ))
        .expect("add chat alias");
    state
}

async fn limited_alias_second_response(limits: GatewayAliasLimits) -> axum::response::Response {
    let mut state = GatewayState::new("test-token");
    state
        .add_alias(GatewayAlias::new(
            "limited",
            OperationSet::new([Operation::Query]),
            Arc::new(MockBackend::new("limited: ", 3)),
            limits,
        ))
        .expect("add limited alias");
    let router = GatewayService::new(state, Vec::new(), GatewayServiceLimits::default())
        .router()
        .expect("router");
    let first = post(
        router.clone(),
        "/v1/query",
        "test-token",
        json!({
            "model": "limited",
            "prompt": "first"
        }),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    post(
        router,
        "/v1/query",
        "test-token",
        json!({
            "model": "limited",
            "prompt": "second"
        }),
    )
    .await
}

struct BlockingBackend {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl GatewayBackend for BlockingBackend {
    async fn query(&self, req: BackendQueryRequest) -> GatewayResult<BackendTextOutput> {
        self.started.notify_one();
        self.release.notified().await;
        Ok(BackendTextOutput {
            text: req.prompt,
            finish_reason: cogentlm_core::FinishReason::Stop,
            usage: None,
            response_id: Some("blocking".to_string()),
        })
    }

    async fn stream_query(
        &self,
        _req: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        Err(GatewayError::new(
            GatewayErrorKind::UnsupportedFeature,
            "stream query is not supported",
        ))
    }

    async fn chat(&self, _req: BackendChatRequest) -> GatewayResult<BackendTextOutput> {
        Err(GatewayError::new(
            GatewayErrorKind::UnsupportedFeature,
            "chat is not supported",
        ))
    }

    async fn stream_chat(
        &self,
        _req: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        Err(GatewayError::new(
            GatewayErrorKind::UnsupportedFeature,
            "stream chat is not supported",
        ))
    }

    async fn embed(&self, _req: BackendEmbedRequest) -> GatewayResult<BackendEmbeddingOutput> {
        Err(GatewayError::new(
            GatewayErrorKind::UnsupportedFeature,
            "embed is not supported",
        ))
    }
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json")
}

async fn response_text(response: axum::response::Response) -> String {
    String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf8")
}
