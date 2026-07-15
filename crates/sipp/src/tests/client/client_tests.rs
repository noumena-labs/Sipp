//! Tests the `client::client` module in `sipp`.
//!
//! Covers endpoint replacement, implicit selection, and client-owned I/O
//! execution with deterministic local HTTP fixtures and no model loading.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use futures::executor::block_on;

use crate::client::{
    EndpointDescriptor, EndpointRef, GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes,
    GatewayTimeoutPolicy, LocalModelDescriptor, SippClient, SippError,
};
use crate::engine::NativeRuntimeConfig;
use crate::lifecycle::test_support::TempDir;
use crate::lifecycle::{ModelError, ModelSource};

#[test]
fn registers_gateway_endpoint_through_add() {
    let mut client = SippClient::new();
    let endpoint = block_on(client.add("gateway", EndpointDescriptor::gateway(gateway_config())))
        .expect("gateway endpoint");

    assert_eq!(
        endpoint,
        EndpointRef::Gateway {
            id: "gateway".to_string()
        }
    );
    assert!(client.resolve(Some(&endpoint), "chat").is_ok());
}

#[test]
fn replacing_an_id_keeps_single_registered_endpoint() {
    let mut client = SippClient::new();
    let first = block_on(client.add("service", EndpointDescriptor::gateway(gateway_config())))
        .expect("first endpoint");
    let second = block_on(client.add("service", EndpointDescriptor::gateway(gateway_config())))
        .expect("replacement endpoint");

    assert_eq!(
        first,
        EndpointRef::Gateway {
            id: "service".to_string()
        }
    );
    assert_eq!(
        second,
        EndpointRef::Gateway {
            id: "service".to_string()
        }
    );
    assert!(client.resolve(Some(&second), "query").is_ok());
}

#[test]
fn gateway_endpoints_are_never_selected_implicitly() {
    let mut client = SippClient::new();
    block_on(client.add("gateway", EndpointDescriptor::gateway(gateway_config())))
        .expect("gateway endpoint");

    assert!(matches!(
        client.resolve(None, "query"),
        Err(SippError::NoSupportedEndpoint { operation: "query" })
    ));
}

#[test]
fn remote_add_uses_the_client_owned_io_runtime() {
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
    let mut client = SippClient::new();
    let error = block_on(client.add(
        "remote",
        EndpointDescriptor::LocalModel(LocalModelDescriptor {
            source: ModelSource::Remote {
                model_urls: vec![format!("http://{address}/model.gguf")],
                projector_url: None,
            },
            storage_root: root.path.clone(),
            config: NativeRuntimeConfig::default(),
        }),
    ))
    .expect_err("503 metadata response must fail");
    server.join().expect("test server");

    assert!(matches!(
        error,
        SippError::ModelLifecycle(ModelError::RemoteMetadataUnavailable {
            status: Some(503),
            retry_after_ms: Some(0),
            ..
        })
    ));
}

fn gateway_config() -> GatewayEndpointConfig {
    GatewayEndpointConfig {
        target: "local".to_string(),
        base_url: "http://127.0.0.1:8080".to_string(),
        routes: GatewayRoutes::default(),
        authentication: GatewayAuthentication::None,
        static_headers: Default::default(),
        timeouts: GatewayTimeoutPolicy::default(),
        protocol_options: Default::default(),
    }
}
