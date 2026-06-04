//! Smoke tests for provider-backed gateway routes.
//!
//! Covers OpenAI, OpenAI-compatible, and Anthropic provider adapters through
//! the public CogentLM gateway HTTP interface with deterministic `wiremock`
//! upstreams and no live provider network calls.

use axum::{
    body::{to_bytes, Body},
    http::{header::AUTHORIZATION, Request, StatusCode},
    Router,
};
use cogentlm_gateway::GatewayFileConfig;
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const GATEWAY_TOKEN_ENV: &str = "COGENTLM_PROVIDER_GATEWAY_SMOKE_TOKEN";
const OPENAI_KEY_ENV: &str = "COGENTLM_PROVIDER_GATEWAY_SMOKE_OPENAI_KEY";
const COMPAT_KEY_ENV: &str = "COGENTLM_PROVIDER_GATEWAY_SMOKE_COMPAT_KEY";
const ANTHROPIC_KEY_ENV: &str = "COGENTLM_PROVIDER_GATEWAY_SMOKE_ANTHROPIC_KEY";

const GATEWAY_TOKEN: &str = "gateway-smoke-token";
const OPENAI_TOKEN: &str = "openai-provider-token";
const COMPAT_TOKEN: &str = "compatible-provider-token";
const ANTHROPIC_TOKEN: &str = "anthropic-provider-token";

#[tokio::test]
async fn provider_backends_work_through_gateway_routes() {
    let openai = MockServer::start().await;
    let compatible = MockServer::start().await;
    let anthropic = MockServer::start().await;
    mount_openai_mocks(&openai).await;
    mount_openai_compatible_mocks(&compatible).await;
    mount_anthropic_mocks(&anthropic).await;

    let router = provider_gateway_router(&openai, &compatible, &anthropic).await;

    let openai_query = post_json(
        router.clone(),
        "/v1/query",
        json!({
            "model": "openai-query",
            "prompt": "gateway prompt",
            "max_tokens": 4,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(openai_query.status(), StatusCode::OK);
    let body = response_json(openai_query).await;
    assert_eq!(body["model"], "openai-query");
    assert_eq!(body["text"], "openai answer");
    assert_eq!(body["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 3);

    let openai_embed = post_json(
        router.clone(),
        "/v1/embed",
        json!({
            "model": "openai-embed",
            "input": "embed me"
        }),
    )
    .await;
    assert_eq!(openai_embed.status(), StatusCode::OK);
    let body = response_json(openai_embed).await;
    assert_eq!(body["model"], "openai-embed");
    assert_eq!(body["embedding"], json!([0.25, -0.5]));
    assert_eq!(body["usage"]["input_tokens"], 2);

    let compatible_chat = post_json(
        router.clone(),
        "/v1/chat",
        json!({
            "model": "compatible-chat",
            "messages": [{ "role": "user", "content": "hello compatible" }],
            "max_tokens": 5
        }),
    )
    .await;
    assert_eq!(compatible_chat.status(), StatusCode::OK);
    let body = response_json(compatible_chat).await;
    assert_eq!(body["model"], "compatible-chat");
    assert_eq!(body["text"], "compatible answer");
    assert_eq!(body["usage"]["total_tokens"], 5);

    let anthropic_query = post_json(
        router.clone(),
        "/v1/query",
        json!({
            "model": "anthropic-query",
            "prompt": "anthropic prompt",
            "max_tokens": 6
        }),
    )
    .await;
    assert_eq!(anthropic_query.status(), StatusCode::OK);
    let body = response_json(anthropic_query).await;
    assert_eq!(body["model"], "anthropic-query");
    assert_eq!(body["text"], "anthropic answer");
    assert_eq!(body["finish_reason"], "length");
    assert_eq!(body["usage"]["total_tokens"], 5);

    let anthropic_stream = post_json(
        router.clone(),
        "/v1/chat",
        json!({
            "model": "anthropic-chat",
            "messages": [{ "role": "user", "content": "stream please" }],
            "stream": true
        }),
    )
    .await;
    assert_eq!(anthropic_stream.status(), StatusCode::OK);
    let body = response_text(anthropic_stream).await;
    assert!(body.contains("event: token"));
    assert!(body.contains(r#""text":"stream""#));
    assert!(body.contains("event: usage"));
    assert!(body.contains(r#""total_tokens":5"#));
    assert!(body.contains("event: done"));
    assert!(body.contains(r#""finish_reason":"stop""#));

    let openai_error = post_json(
        router,
        "/v1/query",
        json!({
            "model": "openai-error",
            "prompt": "fail upstream"
        }),
    )
    .await;
    assert_eq!(openai_error.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        openai_error
            .headers()
            .get("retry-after-ms")
            .and_then(|value| value.to_str().ok()),
        Some("500")
    );
    let body = response_json(openai_error).await;
    assert_eq!(body["error"]["code"], "rate_limited");
    assert_eq!(body["error"]["message"], "provider rate limit exceeded");
    assert!(!body.to_string().contains("provider-secret-token"));
}

async fn mount_openai_mocks(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_TOKEN}")))
        .and(body_json(json!({
            "model": "openai-private-query",
            "prompt": "gateway prompt",
            "max_tokens": 4,
            "temperature": 0.0
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-openai-query")
                .set_body_json(json!({
                    "id": "cmpl-openai-query",
                    "object": "text_completion",
                    "model": "openai-private-query",
                    "choices": [{
                        "text": "openai answer",
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3
                    }
                })),
        )
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", format!("Bearer {OPENAI_TOKEN}")))
        .and(body_json(json!({
            "model": "openai-private-embed",
            "input": "embed me",
            "encoding_format": "float"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-openai-embed")
                .set_body_json(json!({
                    "object": "list",
                    "model": "openai-private-embed",
                    "data": [{
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.25, -0.5]
                    }],
                    "usage": {
                        "prompt_tokens": 2,
                        "total_tokens": 2
                    }
                })),
        )
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_TOKEN}")))
        .and(body_json(json!({
            "model": "openai-private-error",
            "prompt": "fail upstream"
        })))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after-ms", "500")
                .set_body_json(json!({
                    "error": {
                        "message": "provider rejected provider-secret-token",
                        "code": "rate_limit_error"
                    }
                })),
        )
        .mount(server)
        .await;
}

async fn mount_openai_compatible_mocks(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", format!("Bearer {COMPAT_TOKEN}")))
        .and(header("x-provider-static", "static-secret"))
        .and(body_json(json!({
            "model": "compatible-private-chat",
            "messages": [{ "role": "user", "content": "hello compatible" }],
            "max_tokens": 5
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", "req-compatible-chat")
                .set_body_json(json!({
                    "id": "chatcmpl-compatible",
                    "model": "compatible-private-chat",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "compatible answer"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5
                    }
                })),
        )
        .mount(server)
        .await;
}

async fn mount_anthropic_mocks(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", ANTHROPIC_TOKEN))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_json(json!({
            "model": "anthropic-private-query",
            "messages": [{ "role": "user", "content": "anthropic prompt" }],
            "max_tokens": 6
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", "req-anthropic-query")
                .set_body_json(json!({
                    "id": "msg-anthropic-query",
                    "type": "message",
                    "role": "assistant",
                    "model": "anthropic-private-query",
                    "content": [{ "type": "text", "text": "anthropic answer" }],
                    "stop_reason": "max_tokens",
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 2
                    }
                })),
        )
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", ANTHROPIC_TOKEN))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_json(json!({
            "model": "anthropic-private-chat",
            "messages": [{ "role": "user", "content": "stream please" }],
            "max_tokens": 1024,
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .insert_header("request-id", "req-anthropic-stream")
                .set_body_string(concat!(
                    "event: message_start\n",
                    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"anthropic-private-chat\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stream\"}}\n\n",
                    "event: message_delta\n",
                    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n",
                    "event: message_stop\n",
                    "data: {\"type\":\"message_stop\"}\n\n"
                )),
        )
        .mount(server)
        .await;
}

async fn provider_gateway_router(
    openai: &MockServer,
    compatible: &MockServer,
    anthropic: &MockServer,
) -> Router {
    set_provider_smoke_env();
    let openai_url = openai.uri();
    let compatible_url = compatible.uri();
    let anthropic_url = anthropic.uri();
    let config = format!(
        r#"
[server]
bind = "127.0.0.1:0"

[auth]
token_env = "{GATEWAY_TOKEN_ENV}"

[[aliases]]
name = "openai-query"
operations = ["query"]

[aliases.backend]
kind = "open_ai"
model = "openai-private-query"
api_key_env = "{OPENAI_KEY_ENV}"
base_url = "{openai_url}"

[[aliases]]
name = "openai-embed"
operations = ["embed"]

[aliases.backend]
kind = "open_ai"
model = "openai-private-embed"
api_key_env = "{OPENAI_KEY_ENV}"
base_url = "{openai_url}"

[[aliases]]
name = "openai-error"
operations = ["query"]

[aliases.backend]
kind = "open_ai"
model = "openai-private-error"
api_key_env = "{OPENAI_KEY_ENV}"
base_url = "{openai_url}"

[[aliases]]
name = "compatible-chat"
operations = ["chat"]

[aliases.backend]
kind = "open_ai_compatible"
model = "compatible-private-chat"
base_url = "{compatible_url}"

[aliases.backend.auth]
kind = "bearer"
token_env = "{COMPAT_KEY_ENV}"

[[aliases.backend.static_headers]]
name = "x-provider-static"
value = "static-secret"

[[aliases]]
name = "anthropic-query"
operations = ["query"]

[aliases.backend]
kind = "anthropic"
model = "anthropic-private-query"
api_key_env = "{ANTHROPIC_KEY_ENV}"
base_url = "{anthropic_url}"

[[aliases]]
name = "anthropic-chat"
operations = ["chat"]

[aliases.backend]
kind = "anthropic"
model = "anthropic-private-chat"
api_key_env = "{ANTHROPIC_KEY_ENV}"
base_url = "{anthropic_url}"
"#
    );
    GatewayFileConfig::from_toml_str(&config)
        .expect("provider gateway config")
        .build()
        .await
        .expect("provider gateway service")
        .service
        .router()
}

fn set_provider_smoke_env() {
    std::env::set_var(GATEWAY_TOKEN_ENV, GATEWAY_TOKEN);
    std::env::set_var(OPENAI_KEY_ENV, OPENAI_TOKEN);
    std::env::set_var(COMPAT_KEY_ENV, COMPAT_TOKEN);
    std::env::set_var(ANTHROPIC_KEY_ENV, ANTHROPIC_TOKEN);
}

async fn post_json(router: Router, path: &str, body: Value) -> axum::response::Response {
    router
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header(AUTHORIZATION, format!("Bearer {GATEWAY_TOKEN}"))
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response")
}

async fn response_json(response: axum::response::Response) -> Value {
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
