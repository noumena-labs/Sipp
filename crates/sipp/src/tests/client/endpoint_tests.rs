//! Tests typed endpoint capability lookup.

use crate::core::{CapabilitySupport, Operation};
use crate::engine::{ModelCapabilities, ModelClass};
use crate::lifecycle::{ManagedModel, ModelModality, ModelStatus};

use crate::client::endpoint::{EndpointKind, Local};
use crate::client::{Endpoint, EndpointCapabilities, EndpointRef};

#[test]
fn local_input_owns_only_model_identity_and_runtime_configuration() {
    let model = ManagedModel {
        id: "model-a".to_string(),
        name: "Model A".to_string(),
        bytes: 64,
        modality: ModelModality::Text,
        status: ModelStatus::Ready,
    };
    let runtime = crate::engine::NativeRuntimeConfig {
        context: crate::engine::ContextRuntimeConfig {
            n_ctx: Some(256),
            ..Default::default()
        },
        ..Default::default()
    };

    let endpoint: Endpoint = Local::new(&model).runtime(runtime.clone()).into();
    let EndpointKind::Local(local) = endpoint.kind else {
        panic!("expected local endpoint input");
    };

    assert_eq!(local.model_id, model.id);
    assert_eq!(local.runtime, runtime);
}

#[test]
fn endpoint_reference_exposes_only_its_id() {
    let endpoint = EndpointRef::from_id("edge");
    assert_eq!(endpoint.id(), "edge");
}

#[test]
fn remote_text_capabilities_defer_text_and_reject_speech() {
    let capabilities = EndpointCapabilities::remote_text();
    assert_eq!(
        capabilities.for_operation(Operation::Query),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        capabilities.for_operation(Operation::Chat),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        capabilities.for_operation(Operation::Embed),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        capabilities.for_operation(Operation::Listen),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        capabilities.for_operation(Operation::Speak),
        CapabilitySupport::Unsupported
    );
}

#[test]
fn local_listen_requires_generation_audio_and_a_model_chat_template() {
    let mut model = ModelCapabilities {
        model_class: ModelClass::DecoderOnly,
        supports_text_generation: true,
        supports_embeddings: false,
        supports_vision: false,
        audio_sample_rate_hz: Some(16_000),
        generated_audio_sample_rate_hz: None,
        has_chat_template: true,
        embedding: None,
    };

    assert_eq!(
        EndpointCapabilities::from_local(&model).listen,
        CapabilitySupport::Supported
    );
    model.has_chat_template = false;
    assert_eq!(
        EndpointCapabilities::from_local(&model).listen,
        CapabilitySupport::Unsupported
    );
    model.has_chat_template = true;
    model.audio_sample_rate_hz = None;
    assert_eq!(
        EndpointCapabilities::from_local(&model).listen,
        CapabilitySupport::Unsupported
    );
}

#[test]
fn local_audio_generation_endpoint_supports_only_speak() {
    let capabilities = EndpointCapabilities::from_local(&ModelCapabilities {
        model_class: ModelClass::DecoderOnly,
        supports_text_generation: false,
        supports_embeddings: true,
        supports_vision: false,
        audio_sample_rate_hz: Some(24_000),
        generated_audio_sample_rate_hz: Some(24_000),
        has_chat_template: true,
        embedding: None,
    });

    assert_eq!(capabilities.speak, CapabilitySupport::Supported);
    for operation in [
        Operation::Query,
        Operation::Chat,
        Operation::Embed,
        Operation::Listen,
    ] {
        assert_eq!(
            capabilities.for_operation(operation),
            CapabilitySupport::Unsupported
        );
    }
}
