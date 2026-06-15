//! Integration tests for the `sipp` crate-level public API.
//!
//! Covers the root client re-exports, nested native config modules, and the
//! `shard`, client, and `providers` public surfaces without loading local
//! models or calling gateway endpoints.

use sipp::{
    engine::ContextRuntimeConfig, lifecycle::BackendPreference,
    runtime::request::GenerateResponseStatus, NativeRuntimeConfig, SippClient,
};

#[test]
fn facade_reexports_client_and_native_runtime_config() {
    let client = SippClient::new();
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

#[test]
fn facade_reexports_lifecycle_and_runtime_modules() {
    assert_eq!(BackendPreference::Cpu.as_str(), "cpu");
    assert_eq!(GenerateResponseStatus::Completed.as_str(), "completed");
}

mod client_api {
    use sipp::{EndpointCapabilities, EndpointDescriptor, EndpointRef, GatewayEndpointConfig};

    #[test]
    fn gateway_descriptor_is_registered_through_add_contract() {
        let endpoint = EndpointRef::gateway("service");
        assert_eq!(endpoint.kind(), "gateway");
        let descriptor = EndpointDescriptor::gateway(GatewayEndpointConfig {
            target: "local".to_string(),
            base_url: "http://127.0.0.1:8080".to_string(),
            routes: Default::default(),
            authentication: Default::default(),
            static_headers: Default::default(),
            timeouts: Default::default(),
            protocol_options: Default::default(),
        });
        assert!(matches!(descriptor, EndpointDescriptor::Gateway(_)));
        assert_eq!(
            EndpointCapabilities::unknown().query,
            sipp::core::CapabilitySupport::Unknown
        );
    }
}

#[cfg(feature = "providers")]
mod providers_api {
    use sipp::providers::{
        OpenAiCompatibleAdapterConfig, OpenAiCompatibleProtocol, ProviderAuth, ProviderKind,
        SecretString,
    };

    #[test]
    fn provider_kind_has_stable_wire_labels() {
        assert_eq!(ProviderKind::OpenAiCompatible.as_str(), "openai_compatible");
        assert_eq!(ProviderKind::OpenAi.as_str(), "openai");
        assert_eq!(ProviderKind::Anthropic.as_str(), "anthropic");
    }

    #[test]
    fn secret_debug_output_is_redacted() {
        let secret = SecretString::new("real-token");
        let debug = format!("{secret:?}");

        assert!(debug.contains("redacted"));
        assert!(!debug.contains("real-token"));
        assert_eq!(secret.expose(), "real-token");
    }

    #[test]
    fn openai_compatible_config_debug_redacts_static_header_values() {
        let config = OpenAiCompatibleAdapterConfig {
            base_url: "https://provider.example".to_string(),
            auth: ProviderAuth::Bearer(SecretString::new("gateway-token")),
            protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
            static_headers: vec![("x-provider-secret".to_string(), "secret-value".to_string())],
            correlation_header: None,
            timeout: None,
        };
        let debug = format!("{config:?}");

        assert!(debug.contains("x-provider-secret"));
        assert!(debug.contains("[redacted]"));
        assert!(!debug.contains("gateway-token"));
        assert!(!debug.contains("secret-value"));
    }
}

mod shard_api {
    use sipp::shard::{
        detect_model_from_gguf_bytes, inspect_gguf_metadata, plan_gguf_split, split_gguf,
        AssetRole, BrowserCacheLayout, BrowserCachePolicy, GgufError, GgufReadAt, GgufShard,
        GgufShardSink, GgufSplitManifest, GgufSplitOptions, ModelDetectionMethod,
    };

    use std::io::{self, Write};
    use std::path::{Path, PathBuf};

    const GGUF_MAGIC: u32 = 0x4655_4747;
    const DEFAULT_ALIGNMENT: u64 = 32;

    #[test]
    fn default_cache_policy_keeps_small_models_as_single_files() {
        let policy = BrowserCachePolicy::default();

        assert_eq!(
            policy.resolve_layout(Some(policy.direct_load_max_bytes)),
            BrowserCacheLayout::SingleFile
        );
        assert_eq!(
            policy.resolve_layout(Some(policy.direct_load_max_bytes + 1)),
            BrowserCacheLayout::SplitGguf
        );
    }

    #[test]
    fn unknown_source_size_uses_split_layout() {
        assert_eq!(
            BrowserCachePolicy::default().resolve_layout(None),
            BrowserCacheLayout::SplitGguf
        );
    }

    #[test]
    fn public_split_types_are_constructible_and_comparable() {
        let options = GgufSplitOptions::default();
        let shard = GgufShard {
            index: 0,
            count: 1,
            path: PathBuf::from("model-00001-of-00001.gguf"),
            tensor_count: 2,
            bytes: options.shard_max_bytes,
        };
        let manifest = GgufSplitManifest {
            source_bytes: 128,
            total_tensors: 2,
            shards: vec![shard.clone()],
        };

        assert_eq!(manifest.shards[0], shard);
        assert_eq!(manifest.total_tensors, 2);
    }

    #[test]
    fn public_metadata_inspection_and_detection_report_model_role() {
        let bytes = metadata_fixture();
        let metadata = inspect_gguf_metadata(&bytes)
            .expect("inspection")
            .expect("metadata");

        assert_eq!(metadata.general_architecture.as_deref(), Some("qwen2vl"));
        assert_eq!(metadata.clip_has_vision_encoder, Some(true));

        let detection = detect_model_from_gguf_bytes("model.gguf", &bytes).expect("detection");
        assert_eq!(
            detection.detection_method,
            ModelDetectionMethod::GgufMetadata
        );
        assert_eq!(detection.inspection.role, AssetRole::Model);
        assert!(detection.inspection.vision_capable);
    }

    #[test]
    fn public_split_and_plan_use_read_at_and_sink_traits() {
        let bytes = split_fixture();
        let source_bytes = u64::try_from(bytes.len()).expect("source length");

        let mut planner_source = MemoryReadAt::new(bytes.clone());
        let plan = plan_gguf_split(
            source_bytes,
            &mut planner_source,
            "model",
            GgufSplitOptions {
                shard_max_bytes: 64,
            },
        )
        .expect("plan");
        assert_eq!(plan.shards.len(), 2);
        assert_eq!(
            plan.shards[0].path,
            PathBuf::from("model-00001-of-00002.gguf")
        );
        assert_eq!(plan.shards[1].bytes, 64);

        let mut source = MemoryReadAt::new(bytes);
        let mut sink = MemoryShardSink::default();
        let manifest = split_gguf(
            source_bytes,
            &mut source,
            "model",
            GgufSplitOptions {
                shard_max_bytes: 64,
            },
            &mut sink,
        )
        .expect("split");

        assert_eq!(manifest.shards.len(), 2);
        assert_eq!(sink.shards.len(), 2);
        assert_eq!(
            sink.shards[0].path,
            PathBuf::from("model-00001-of-00002.gguf")
        );
        assert!(!sink.shards[0].bytes.is_empty());
    }

    fn metadata_fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        write_header(&mut bytes, 0, 2);
        write_kv_string(&mut bytes, "general.architecture", "qwen2vl");
        write_kv_bool(&mut bytes, "clip.has_vision_encoder", true);
        bytes
    }

    fn split_fixture() -> Vec<u8> {
        let tensors = [("a.weight", vec![1_u8; 64]), ("b.weight", vec![2_u8; 64])];
        let mut metadata = Vec::new();
        write_header(&mut metadata, 2, 1);
        write_kv_string(&mut metadata, "general.architecture", "llama");

        let mut tensor_data = Vec::new();
        let mut offsets = Vec::new();
        for (_, data) in &tensors {
            let offset = align_to(
                u64::try_from(tensor_data.len()).expect("tensor data length"),
                DEFAULT_ALIGNMENT,
            );
            tensor_data.resize(usize::try_from(offset).expect("offset"), 0);
            offsets.push(offset);
            tensor_data.extend_from_slice(data);
        }

        for ((name, _), offset) in tensors.iter().zip(offsets) {
            write_string(&mut metadata, name);
            write_u32(&mut metadata, 1);
            write_u64(&mut metadata, 1);
            write_u32(&mut metadata, 0);
            write_u64(&mut metadata, offset);
        }

        let data_offset = align_to(
            u64::try_from(metadata.len()).expect("metadata length"),
            DEFAULT_ALIGNMENT,
        );
        metadata.resize(usize::try_from(data_offset).expect("data offset"), 0);
        metadata.extend_from_slice(&tensor_data);
        metadata
    }

    fn write_header(bytes: &mut Vec<u8>, tensor_count: u64, kv_count: u64) {
        write_u32(bytes, GGUF_MAGIC);
        write_u32(bytes, 3);
        write_u64(bytes, tensor_count);
        write_u64(bytes, kv_count);
    }

    fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        write_string(bytes, key);
        write_u32(bytes, 8);
        write_string(bytes, value);
    }

    fn write_kv_bool(bytes: &mut Vec<u8>, key: &str, value: bool) {
        write_string(bytes, key);
        write_u32(bytes, 7);
        bytes.push(u8::from(value));
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        write_u64(bytes, u64::try_from(value.len()).expect("string length"));
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn align_to(value: u64, alignment: u64) -> u64 {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    struct MemoryReadAt {
        bytes: Vec<u8>,
    }

    impl MemoryReadAt {
        fn new(bytes: Vec<u8>) -> Self {
            Self { bytes }
        }
    }

    impl GgufReadAt for MemoryReadAt {
        fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), GgufError> {
            let offset = usize::try_from(offset)
                .map_err(|_| GgufError::Invalid("offset too large".to_string()))?;
            let end = offset
                .checked_add(dst.len())
                .ok_or_else(|| GgufError::Invalid("read offset overflow".to_string()))?;
            let Some(bytes) = self.bytes.get(offset..end) else {
                return Err(GgufError::Invalid("read exceeds source".to_string()));
            };
            dst.copy_from_slice(bytes);
            Ok(())
        }
    }

    #[derive(Default)]
    struct MemoryShardSink {
        shards: Vec<MemoryShard>,
    }

    struct MemoryShard {
        path: PathBuf,
        bytes: Vec<u8>,
    }

    impl GgufShardSink for MemoryShardSink {
        type Writer = MemoryShardWriter;

        fn create_shard(
            &mut self,
            path: &Path,
            _index: u16,
            _count: u16,
        ) -> Result<Self::Writer, GgufError> {
            Ok(MemoryShardWriter {
                path: path.to_path_buf(),
                bytes: Vec::new(),
            })
        }

        fn finish_shard(&mut self, writer: Self::Writer) -> Result<u64, GgufError> {
            let bytes = u64::try_from(writer.bytes.len())
                .map_err(|_| GgufError::Invalid("shard length too large".to_string()))?;
            self.shards.push(MemoryShard {
                path: writer.path,
                bytes: writer.bytes,
            });
            Ok(bytes)
        }
    }

    struct MemoryShardWriter {
        path: PathBuf,
        bytes: Vec<u8>,
    }

    impl Write for MemoryShardWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
