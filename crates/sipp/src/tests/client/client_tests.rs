//! Tests the `client::client` module in `sipp`.
//!
//! Covers endpoint replacement, implicit selection, and client-owned I/O
//! execution with deterministic local HTTP fixtures and no model loading.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use futures::executor::block_on;

use crate::client::{
    EndpointRef, GatewayAuthentication, GatewayDescriptor, GatewayRoutes, GatewayTimeoutPolicy,
    SippClient, SippError,
};
use crate::lifecycle::test_support::TempDir;
use crate::lifecycle::ModelError;

#[test]
fn registers_gateway_endpoint_through_add() {
    let mut client = SippClient::new().expect("client");
    let endpoint = block_on(client.add("gateway", gateway_descriptor())).expect("gateway endpoint");

    assert_eq!(endpoint, EndpointRef::from_id("gateway"));
    assert!(client.resolve(Some(&endpoint), "chat").is_ok());
}

#[test]
fn replacing_an_id_keeps_single_registered_endpoint() {
    let mut client = SippClient::new().expect("client");
    let first = block_on(client.add("service", gateway_descriptor())).expect("first endpoint");
    let second =
        block_on(client.add("service", gateway_descriptor())).expect("replacement endpoint");

    assert_eq!(first, EndpointRef::from_id("service"));
    assert_eq!(second, EndpointRef::from_id("service"));
    assert!(client.resolve(Some(&second), "query").is_ok());
}

#[test]
fn gateway_endpoints_are_never_selected_implicitly() {
    let mut client = SippClient::new().expect("client");
    block_on(client.add("gateway", gateway_descriptor())).expect("gateway endpoint");

    assert!(matches!(
        client.resolve(None, "query"),
        Err(SippError::NoSupportedEndpoint { operation: "query" })
    ));
}

#[tokio::test]
async fn adding_a_remote_source_uses_the_client_model_store() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let address = listener.local_addr().expect("test server address");
    let server = thread::spawn(move || {
        for _ in 0..4 {
            let (mut stream, _) = listener.accept().expect("accept metadata request");
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).expect("read metadata request");
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nRetry-After: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("write metadata response");
        }
    });
    let root = TempDir::new("client", "remote-owned-runtime");
    let client = SippClient::with_storage_root(root.path.clone()).expect("client");
    let error = client
        .models()
        .add([format!("http://{address}/model.gguf")])
        .await
        .expect_err("503 metadata response must fail");
    server.join().expect("test server");

    assert!(matches!(
        error,
        ModelError::RemoteMetadataUnavailable {
            status: Some(503),
            retry_after_ms: Some(0),
            ..
        }
    ));
}

fn gateway_descriptor() -> GatewayDescriptor {
    GatewayDescriptor {
        target: "local".to_string(),
        base_url: "http://127.0.0.1:8080".to_string(),
        routes: GatewayRoutes::default(),
        authentication: GatewayAuthentication::None,
        static_headers: Default::default(),
        timeouts: GatewayTimeoutPolicy::default(),
        protocol_options: Default::default(),
    }
}
