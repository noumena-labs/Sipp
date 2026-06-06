//! Tests the framework-agnostic gateway adapter policy and execution layer.

use std::sync::Arc;

use crate::{
    BackendGenerationOptions, BackendQueryRequest, GatewayAccess, GatewayAdapter, GatewayAlias,
    GatewayAliasLimits, GatewayCaller, GatewayRequestLimits, GatewayResult, MockBackend, Operation,
    OperationSet,
};

#[tokio::test]
async fn adapter_executes_query_through_alias() {
    let adapter = adapter_with_limits(GatewayAliasLimits::default()).expect("adapter");
    let output = adapter
        .query(
            &GatewayCaller::anonymous(),
            "mock",
            BackendQueryRequest {
                prompt: "hello".to_string(),
                options: BackendGenerationOptions::default(),
                gateway_options: Default::default(),
            },
        )
        .await
        .expect("query");

    assert_eq!(output.text, "mock: hello");
    let snapshot = adapter.snapshot().expect("snapshot");
    assert_eq!(snapshot.aliases[0].global_total_requests, 1);
}

#[tokio::test]
async fn adapter_enforces_global_rate_limits() {
    let adapter = adapter_with_limits(GatewayAliasLimits {
        global: GatewayRequestLimits {
            max_requests_per_minute: Some(1),
            ..GatewayRequestLimits::default()
        },
        ..GatewayAliasLimits::default()
    })
    .expect("adapter");

    query_once(&adapter).await.expect("first query");
    let error = query_once(&adapter)
        .await
        .expect_err("second query should be rate limited");

    assert_eq!(error.code(), "rate_limited");
    assert!(error.retry_after.is_some());
}

#[tokio::test]
async fn adapter_requires_caller_id_for_per_caller_limits() {
    let adapter = adapter_with_limits(GatewayAliasLimits {
        per_caller: Some(GatewayRequestLimits {
            max_requests_per_minute: Some(1),
            ..GatewayRequestLimits::default()
        }),
        ..GatewayAliasLimits::default()
    })
    .expect("adapter");

    let error = query_once(&adapter)
        .await
        .expect_err("anonymous caller should fail");

    assert_eq!(error.code(), "invalid_request");
    assert_eq!(
        error.message,
        "caller ID is required for per-caller gateway limits"
    );
}

#[tokio::test]
async fn adapter_bounds_tracked_callers() {
    let adapter = adapter_with_limits(GatewayAliasLimits {
        per_caller: Some(GatewayRequestLimits {
            max_requests_total: Some(2),
            ..GatewayRequestLimits::default()
        }),
        max_tracked_callers: 1,
        ..GatewayAliasLimits::default()
    })
    .expect("adapter");

    query_for(&adapter, "caller-one")
        .await
        .expect("first caller query");
    let error = query_for(&adapter, "caller-two")
        .await
        .expect_err("second caller should exceed tracked caller capacity");

    assert_eq!(error.code(), "overloaded");
}

#[tokio::test]
async fn adapter_enforces_access_scopes() {
    let adapter = adapter_with_limits(GatewayAliasLimits::default()).expect("adapter");
    let caller = GatewayCaller {
        id: Some("caller".to_string()),
        access: GatewayAccess::new([("mock".to_string(), OperationSet::new([Operation::Embed]))])
            .expect("access"),
    };

    let error = adapter
        .query(
            &caller,
            "mock",
            BackendQueryRequest {
                prompt: "hello".to_string(),
                options: BackendGenerationOptions::default(),
                gateway_options: Default::default(),
            },
        )
        .await
        .expect_err("query should be forbidden");

    assert_eq!(error.code(), "authorization");
}

fn adapter_with_limits(limits: GatewayAliasLimits) -> GatewayResult<GatewayAdapter> {
    GatewayAdapter::builder()
        .alias(GatewayAlias::new(
            "mock",
            OperationSet::all(),
            Arc::new(MockBackend::new("mock: ", 4)),
            limits,
        )?)
        .build()
}

async fn query_once(adapter: &GatewayAdapter) -> GatewayResult<crate::BackendTextOutput> {
    adapter
        .query(
            &GatewayCaller::anonymous(),
            "mock",
            BackendQueryRequest {
                prompt: "hello".to_string(),
                options: BackendGenerationOptions::default(),
                gateway_options: Default::default(),
            },
        )
        .await
}

async fn query_for(
    adapter: &GatewayAdapter,
    caller_id: &str,
) -> GatewayResult<crate::BackendTextOutput> {
    adapter
        .query(
            &GatewayCaller::identified(caller_id.to_string())?,
            "mock",
            BackendQueryRequest {
                prompt: "hello".to_string(),
                options: BackendGenerationOptions::default(),
                gateway_options: Default::default(),
            },
        )
        .await
}
