//! Integration tests for the `cogentlm` facade public API.
//!
//! Covers root-level client re-exports and nested native config modules without
//! loading local models or calling remote gateways.

use cogentlm::{engine::ContextRuntimeConfig, CogentClient, NativeRuntimeConfig};

#[test]
fn facade_reexports_client_and_native_runtime_config() {
    let client = CogentClient::new();
    let config = NativeRuntimeConfig {
        context: ContextRuntimeConfig {
            n_ctx: Some(128),
            ..Default::default()
        },
        ..Default::default()
    };

    assert_eq!(config.context.n_ctx, Some(128));
    drop(client);
}
