//! Tests the `client::client` module in `sipp`.
//!
//! Covers endpoint replacement and shutdown, implicit selection, and
//! client-owned I/O execution with deterministic fakes plus explicitly ignored
//! model-backed smoke tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use futures::executor::block_on;

use crate::client::dispatch::{EndpointCloseFuture, InferenceEndpoint};
use crate::client::{
    EndpointCapabilities, EndpointRef, GatewayAuthentication, GatewayDescriptor, GatewayRoutes,
    GatewayTimeoutPolicy, SippAudioRun, SippChatRequest, SippClient, SippEmbedRequest,
    SippEmbeddingRun, SippError, SippListenRequest, SippQueryRequest, SippRequestContext,
    SippSpeakRequest, SippTextRun,
};
#[cfg(feature = "providers")]
use crate::client::{ProviderDescriptor, ProviderSecret};
use crate::core::{CapabilitySupport, Operation};
use crate::endpoint::Local;
use crate::lifecycle::test_support::TempDir;
use crate::lifecycle::ModelError;
struct CloseTrackingEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    close_count: Arc<AtomicUsize>,
}

impl InferenceEndpoint for CloseTrackingEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn close(&self) -> EndpointCloseFuture<'_> {
        Box::pin(async {
            self.close_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }

    fn query_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippQueryRequest,
    ) -> SippTextRun {
        unreachable!("close tracking endpoint does not run queries")
    }

    fn chat_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippChatRequest,
    ) -> SippTextRun {
        unreachable!("close tracking endpoint does not run chat")
    }

    fn embed_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippEmbedRequest,
    ) -> SippEmbeddingRun {
        unreachable!("close tracking endpoint does not run embeddings")
    }

    fn listen_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippListenRequest,
    ) -> SippTextRun {
        unreachable!("close tracking endpoint does not run transcription")
    }

    fn speak_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippSpeakRequest,
    ) -> SippAudioRun {
        unreachable!("close tracking endpoint does not run synthesis")
    }
}

fn isolated_client(scope: &str) -> (TempDir, SippClient) {
    let root = TempDir::new("client", scope);
    let client = SippClient::with_storage_root(root.path.join("store")).expect("client");
    (root, client)
}

#[test]
fn registers_gateway_endpoint_through_add() {
    let (_root, mut client) = isolated_client("register-gateway");
    let endpoint = block_on(client.add("gateway", gateway_descriptor())).expect("gateway endpoint");

    assert_eq!(endpoint, EndpointRef::from_id("gateway"));
    assert!(client.resolve(Some(&endpoint), Operation::Chat).is_ok());
}

#[test]
fn replacing_an_id_keeps_single_registered_endpoint() {
    let (_root, mut client) = isolated_client("replace-gateway");
    let endpoint = EndpointRef::from_id("service");
    let close_count = Arc::new(AtomicUsize::new(0));
    client.endpoints.insert(
        endpoint.clone(),
        Arc::new(CloseTrackingEndpoint {
            endpoint: endpoint.clone(),
            capabilities: EndpointCapabilities::remote_text(),
            close_count: close_count.clone(),
        }),
    );

    let replacement =
        block_on(client.add("service", gateway_descriptor())).expect("replacement endpoint");

    assert_eq!(close_count.load(Ordering::SeqCst), 1);
    assert_eq!(replacement, endpoint);
    assert_eq!(client.endpoints.len(), 1);
    assert!(client.resolve(Some(&replacement), Operation::Query).is_ok());
}

#[test]
fn failed_local_replacement_destroys_the_old_endpoint_and_publishes_nothing() {
    let (root, mut client) = isolated_client("failed-local-replacement");
    let model_path = root.path.join("invalid.gguf");
    std::fs::write(&model_path, b"not a gguf").expect("model fixture");
    let model = block_on(client.models.add([&model_path])).expect("managed model");
    let endpoint = EndpointRef::from_id("service");
    let close_count = Arc::new(AtomicUsize::new(0));
    client.endpoints.insert(
        endpoint.clone(),
        Arc::new(CloseTrackingEndpoint {
            endpoint: endpoint.clone(),
            capabilities: EndpointCapabilities::remote_text(),
            close_count: close_count.clone(),
        }),
    );
    client
        .local_models
        .insert(endpoint.id().to_string(), model.id.clone());
    block_on(client.models.mark_used(&model.id));

    let error = block_on(client.add(endpoint.id(), Local::new(&model)))
        .expect_err("invalid model activation");

    assert!(matches!(error, SippError::Local(_)));
    assert_eq!(close_count.load(Ordering::SeqCst), 1);
    assert!(matches!(
        client.resolve(Some(&endpoint), Operation::Query),
        Err(SippError::EndpointNotFound(actual)) if actual == endpoint
    ));
    block_on(client.models.remove(&model.id)).expect("released model usage");
}

#[test]
fn removing_an_endpoint_awaits_shutdown() {
    let root = TempDir::new("client", "remove-awaits-shutdown");
    let mut client = SippClient::with_storage_root(root.path.clone()).expect("client");
    let endpoint = EndpointRef::from_id("local");
    let close_count = Arc::new(AtomicUsize::new(0));
    client.endpoints.insert(
        endpoint.clone(),
        Arc::new(CloseTrackingEndpoint {
            endpoint,
            capabilities: EndpointCapabilities::remote_text(),
            close_count: close_count.clone(),
        }),
    );
    block_on(client.remove("local")).expect("remove endpoint");

    assert_eq!(close_count.load(Ordering::SeqCst), 1);
}

#[test]
fn gateway_endpoints_are_never_selected_implicitly() {
    let (_root, mut client) = isolated_client("implicit-gateway-selection");
    block_on(client.add("gateway", gateway_descriptor())).expect("gateway endpoint");

    assert!(matches!(
        client.resolve(None, Operation::Query),
        Err(SippError::NoSupportedEndpoint { operation: "query" })
    ));
}

#[test]
fn typed_resolution_selects_the_only_local_listen_endpoint() {
    let root = TempDir::new("client", "local-listen-selection");
    let mut client = SippClient::with_storage_root(root.path.clone()).expect("client");
    let endpoint = EndpointRef::from_id("listener");
    client.endpoints.insert(
        endpoint.clone(),
        Arc::new(CloseTrackingEndpoint {
            endpoint: endpoint.clone(),
            capabilities: EndpointCapabilities {
                query: CapabilitySupport::Unsupported,
                chat: CapabilitySupport::Unsupported,
                embed: CapabilitySupport::Unsupported,
                listen: CapabilitySupport::Supported,
                speak: CapabilitySupport::Unsupported,
            },
            close_count: Arc::new(AtomicUsize::new(0)),
        }),
    );
    client
        .local_models
        .insert(endpoint.id().to_string(), "asr-model".to_string());
    let resolved = client
        .resolve(None, Operation::Listen)
        .expect("single local listen endpoint");
    assert_eq!(resolved.endpoint(), &endpoint);
    assert!(client.resolve(Some(&endpoint), Operation::Listen).is_ok());
}

#[test]
fn gateway_speech_is_rejected_before_transport_execution() {
    let (_root, mut client) = isolated_client("gateway-speech-rejection");
    let endpoint = block_on(client.add("gateway", gateway_descriptor())).expect("gateway endpoint");

    let listen_error = block_on(client.listen(SippListenRequest {
        endpoint: Some(endpoint.clone()),
        audio: vec![1],
        language: None,
        max_tokens: None,
    }))
    .expect_err("gateway listen must be unsupported");
    assert!(matches!(
        listen_error,
        SippError::UnsupportedOperation {
            endpoint: actual,
            operation: "listen"
        } if actual == endpoint
    ));

    let speak_error = block_on(client.speak(SippSpeakRequest {
        endpoint: Some(endpoint.clone()),
        text: "hello".to_string(),
        language: None,
        speaker_audio: None,
        max_duration_ms: None,
    }))
    .expect_err("gateway speak must be unsupported");
    assert!(matches!(
        speak_error,
        SippError::UnsupportedOperation {
            endpoint: actual,
            operation: "speak"
        } if actual == endpoint
    ));
}

#[cfg(feature = "providers")]
#[test]
fn provider_speech_is_rejected_before_transport_execution() {
    let (_root, mut client) = isolated_client("provider-speech-rejection");
    let endpoint = block_on(client.add(
        "provider",
        ProviderDescriptor::openai("model", ProviderSecret::new("test-key")),
    ))
    .expect("provider endpoint");

    let listen_error = block_on(client.listen(SippListenRequest {
        endpoint: Some(endpoint.clone()),
        audio: vec![1],
        language: None,
        max_tokens: None,
    }))
    .expect_err("provider listen must be unsupported");
    assert!(matches!(
        listen_error,
        SippError::UnsupportedOperation {
            operation: "listen",
            ..
        }
    ));

    let speak_error = block_on(client.speak(SippSpeakRequest {
        endpoint: Some(endpoint),
        text: "hello".to_string(),
        language: None,
        speaker_audio: None,
        max_duration_ms: None,
    }))
    .expect_err("provider speak must be unsupported");
    assert!(matches!(
        speak_error,
        SippError::UnsupportedOperation {
            operation: "speak",
            ..
        }
    ));
}

#[test]
fn speech_request_fields_are_validated_once_at_the_client_boundary() {
    let (_root, client) = isolated_client("speech-validation");

    assert!(matches!(
        block_on(client.listen(SippListenRequest::new([]))),
        Err(SippError::InvalidRequest(message)) if message == "listen audio must not be empty"
    ));
    assert!(matches!(
        block_on(client.listen(SippListenRequest {
            endpoint: None,
            audio: vec![1],
            language: Some(String::new()),
            max_tokens: None,
        })),
        Err(SippError::InvalidRequest(message)) if message == "listen language must not be empty"
    ));
    assert!(matches!(
        block_on(client.listen(SippListenRequest::new([1]).max_tokens(0))),
        Err(SippError::InvalidRequest(message)) if message == "max_tokens must be positive"
    ));
    assert!(matches!(
        block_on(client.speak(SippSpeakRequest {
            endpoint: None,
            text: String::new(),
            language: None,
            speaker_audio: None,
            max_duration_ms: None,
        })),
        Err(SippError::InvalidRequest(message)) if message == "speak text must not be empty"
    ));
    assert!(matches!(
        block_on(client.speak(SippSpeakRequest {
            endpoint: None,
            text: "hello".to_string(),
            language: None,
            speaker_audio: Some(Vec::new()),
            max_duration_ms: None,
        })),
        Err(SippError::InvalidRequest(message))
            if message == "speak speaker audio must not be empty"
    ));
    assert!(matches!(
        block_on(client.speak(SippSpeakRequest::new("hello").max_duration_ms(0))),
        Err(SippError::InvalidRequest(message))
            if message == "max_duration_ms must be positive"
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
