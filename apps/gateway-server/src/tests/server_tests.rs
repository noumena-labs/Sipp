use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use cogentlm_client::{
    CogentClient, EndpointDescriptor, GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes,
    GatewayTimeoutPolicy,
};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::config::{GatewayServerConfig, GatewayServerRuntime, LoadedToken, RouteConfig};
use crate::http::GatewayHttpService;
use crate::metrics::GatewayMetrics;

async fn service(base_url: String) -> GatewayHttpService {
    let routes = GatewayRoutes {
        query: "/generate".to_string(),
        chat: "/conversation".to_string(),
        embed: "/vectorize".to_string(),
        index: Some("/about".to_string()),
        health: Some("/live".to_string()),
        readiness: Some("/ready".to_string()),
        metrics: Some("/telemetry".to_string()),
    };
    let mut client = CogentClient::new();
    let endpoint = client
        .add(
            "gateway-upstream",
            EndpointDescriptor::gateway(GatewayEndpointConfig {
                target: "upstream".to_string(),
                base_url,
                routes: GatewayRoutes::default(),
                authentication: GatewayAuthentication::None,
                static_headers: Default::default(),
                timeouts: GatewayTimeoutPolicy::default(),
                protocol_options: Default::default(),
            }),
        )
        .await
        .expect("gateway endpoint");
    let runtime = GatewayServerRuntime {
        client: Arc::new(client),
        targets: Arc::new(BTreeMap::from([("allowed".to_string(), endpoint)])),
    };

    GatewayHttpService::new(
        runtime,
        routes,
        vec![LoadedToken {
            secret: "test-secret".to_string(),
            caller: "test-client".to_string(),
            targets: BTreeSet::from(["allowed".to_string()]),
        }],
        Arc::new(GatewayMetrics::new()),
        1024,
        &[],
        None,
    )
    .expect("service")
}

#[test]
fn config_accepts_typed_custom_routes() {
    let source = r#"
        public_bind = "127.0.0.1:8080"
        management_bind = "127.0.0.1:9090"

        [routes]
        query = "/generate"
        chat = "/conversation"
        embed = "/vectorize"
        index = "/about"
        health = "/live"
        readiness = "/ready"
        metrics = "/telemetry"

        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["local"]

        [[targets]]
        name = "local"
        type = "local"
        model = "model.gguf"
    "#;

    let config: GatewayServerConfig = toml::from_str(source).expect("config");
    config.validate().expect("valid config");
    assert_eq!(config.routes.query, "/generate");
    assert_eq!(config.routes.metrics.as_deref(), Some("/telemetry"));
}

#[test]
fn config_rejects_missing_application_policy() {
    let config = GatewayServerConfig {
        routes: RouteConfig::default(),
        ..GatewayServerConfig::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn config_rejects_duplicate_routes_on_the_same_listener() {
    let mut routes = RouteConfig::default();
    routes.chat = routes.query.clone();
    let config = GatewayServerConfig {
        routes,
        ..GatewayServerConfig::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn shipped_production_config_matches_the_new_schema() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/production.toml");
    GatewayServerConfig::from_path(&path).expect("production config");
}

#[tokio::test]
async fn public_routes_require_authentication_apply_policy_and_call_client() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "upstream-response",
            "model": "upstream",
            "text": "hello",
            "finish_reason": "stop"
        })))
        .mount(&upstream)
        .await;

    let router = service(upstream.uri()).await.public_router();
    let missing = router
        .clone()
        .oneshot(
            Request::post("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"allowed","prompt":"hello"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);

    let forbidden = router
        .clone()
        .oneshot(
            Request::post("/generate")
                .header("authorization", "Bearer test-secret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"blocked","prompt":"hello"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);

    let allowed = router
        .oneshot(
            Request::post("/generate")
                .header("authorization", "Bearer test-secret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"allowed","prompt":"hello"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(allowed.status(), StatusCode::OK);
    let body = to_bytes(allowed.into_body(), 4096).await.expect("body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(body["text"], "hello");
    assert_eq!(body["model"], "allowed");
}

#[tokio::test]
async fn management_routes_are_application_owned() {
    let upstream = MockServer::start().await;
    let router = service(upstream.uri()).await.management_router();

    for (route, expected) in [("/live", "ok"), ("/ready", "ready")] {
        let response = router
            .clone()
            .oneshot(Request::get(route).body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.expect("body");
        assert_eq!(body, expected);
    }

    let metrics = router
        .oneshot(
            Request::get("/telemetry")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(metrics.status(), StatusCode::OK);
}
