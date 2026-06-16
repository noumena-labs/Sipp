from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Iterator, Literal, Mapping, Optional, Sequence, TypedDict, Union

PathLike = Union[str, Path]
GpuLayerConfig = Union[str, dict[str, int]]
ActivePythonBackend = Literal["cpu", "cuda", "metal", "vulkan"]
DEFAULT_CONTEXT_KEY: Final[str]
DEFAULT_MAX_TOKENS: Final[int]

class ModelPlacementConfig:
    def __init__(
        self,
        *,
        devices: Optional[Sequence[str]] = None,
        gpu_layers: Optional[GpuLayerConfig] = None,
        split_mode: Optional[str] = None,
        main_gpu: Optional[int] = None,
        tensor_split: Optional[Sequence[float]] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        fit_params: Optional[bool] = None,
        fit_params_min_ctx: Optional[int] = None,
        fit_params_target_bytes: Optional[Sequence[int]] = None,
        check_tensors: Optional[bool] = None,
        no_extra_bufts: Optional[bool] = None,
        no_host: Optional[bool] = None,
    ) -> None: ...

class ContextRuntimeConfig:
    def __init__(
        self,
        *,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_parallel: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        flash_attention: Optional[str] = None,
        kv_unified: Optional[bool] = None,
        cache_type_k: Optional[str] = None,
        cache_type_v: Optional[str] = None,
        offload_kqv: Optional[bool] = None,
        op_offload: Optional[bool] = None,
        swa_full: Optional[bool] = None,
        warmup: Optional[bool] = None,
        rope_scaling: Optional[str] = None,
        rope_freq_base: Optional[float] = None,
        rope_freq_scale: Optional[float] = None,
        yarn_orig_ctx: Optional[int] = None,
        yarn_ext_factor: Optional[float] = None,
        yarn_attn_factor: Optional[float] = None,
        yarn_beta_fast: Optional[float] = None,
        yarn_beta_slow: Optional[float] = None,
        embeddings: Optional[bool] = None,
        pooling: Optional[str] = None,
    ) -> None: ...

class SamplingRuntimeConfig:
    def __init__(
        self,
        *,
        samplers: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        xtc_probability: Optional[float] = None,
        xtc_threshold: Optional[float] = None,
        top_n_sigma: Optional[float] = None,
        temperature: Optional[float] = None,
        dynatemp_range: Optional[float] = None,
        dynatemp_exponent: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        dry_multiplier: Optional[float] = None,
        dry_base: Optional[float] = None,
        dry_allowed_length: Optional[int] = None,
        dry_penalty_last_n: Optional[int] = None,
        dry_sequence_breakers: Optional[Sequence[str]] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        min_keep: Optional[int] = None,
        n_probs: Optional[int] = None,
        logit_bias: Optional[Sequence[tuple[int, float]]] = None,
        ignore_eos: Optional[bool] = None,
        grammar_lazy: Optional[bool] = None,
        preserved_tokens: Optional[Sequence[int]] = None,
        backend_sampling: Optional[bool] = None,
    ) -> None: ...

SamplingRuntimeOverride = SamplingRuntimeConfig

class SchedulerPolicyConfig:
    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        decode_token_reserve: Optional[int] = None,
        enable_adaptive_prefill_chunking: Optional[bool] = None,
    ) -> None: ...

class SchedulerRuntimeConfig:
    def __init__(
        self,
        *,
        continuous_batching: Optional[bool] = None,
        policy: Optional[SchedulerPolicyConfig] = None,
        prefill_chunk_size: Optional[int] = None,
        max_running_requests: Optional[int] = None,
        max_queued_requests: Optional[int] = None,
    ) -> None: ...

class CacheRuntimeConfig:
    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        retained_prefix_tokens: Optional[int] = None,
        snapshot_interval_tokens: Optional[int] = None,
        max_snapshot_entries: Optional[int] = None,
        max_snapshot_bytes: Optional[int] = None,
    ) -> None: ...

class MultimodalRuntimeConfig:
    def __init__(
        self,
        *,
        projector_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        image_min_tokens: Optional[int] = None,
        image_max_tokens: Optional[int] = None,
    ) -> None: ...

class ResidencyRuntimeConfig:
    def __init__(
        self,
        *,
        max_gpu_models_per_device: Optional[int] = None,
        allow_cpu_models_while_gpu_loaded: Optional[bool] = None,
        require_gpu_lease: Optional[bool] = None,
        gpu_memory_safety_margin_bytes: Optional[int] = None,
    ) -> None: ...

class ObservabilityRuntimeConfig:
    def __init__(self, *, runtime_metrics: bool = False, backend_profiling: bool = False) -> None: ...

class NativeRuntimeConfig:
    def __init__(
        self,
        *,
        placement: Optional[ModelPlacementConfig] = None,
        context: Optional[ContextRuntimeConfig] = None,
        sampling: Optional[SamplingRuntimeConfig] = None,
        scheduler: Optional[SchedulerRuntimeConfig] = None,
        cache: Optional[CacheRuntimeConfig] = None,
        multimodal: Optional[MultimodalRuntimeConfig] = None,
        residency: Optional[ResidencyRuntimeConfig] = None,
        observability: Optional[ObservabilityRuntimeConfig] = None,
    ) -> None: ...

class ChatMessage:
    role: str
    content: str
    def __init__(self, role: str, content: str) -> None: ...

class TokenEmissionStats(TypedDict):
    frames_sent: int
    bytes_sent: int
    batches_sent: int

class TokenBatch(TypedDict):
    request_id: str
    stream_id: int
    sequence_start: int
    text: str
    frame_count: int
    byte_count: int
    stats: TokenEmissionStats

class TokenUsage(TypedDict):
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]

EndpointOptions = dict[str, Any]
ProviderOptions = dict[str, Any]

class EndpointRefDict(TypedDict):
    kind: str
    id: str

class SippTextResponse(TypedDict):
    endpoint: EndpointRefDict
    text: str
    finish_reason: str
    usage: Optional[TokenUsage]
    local_stats: Optional[dict[str, Any]]

class SippEmbeddingResponse(TypedDict):
    endpoint: EndpointRefDict
    values: list[float]
    usage: Optional[TokenUsage]
    local_stats: Optional[dict[str, Any]]
    pooling: Optional[str]
    normalized: Optional[bool]

class UnsupportedOperationError(Exception): ...

class EndpointError(Exception):
    kind: str
    status: Optional[int]
    code: Optional[str]
    request_id: Optional[str]
    retry_after_ms: Optional[float]
    raw_body: Any

class ProviderError(Exception):
    kind: str
    provider: str
    status: Optional[int]
    code: Optional[str]
    request_id: Optional[str]
    retry_after_ms: Optional[float]
    raw_body: Any

class GatewayDescriptor:
    def __init__(
        self,
        target: str,
        base_url: str,
        *,
        authentication_kind: Literal["none", "bearer", "header"] = "none",
        authentication_value: Optional[str] = None,
        authentication_header: Optional[str] = None,
        static_headers: Optional[Mapping[str, str]] = None,
        timeout_ms: Optional[int] = None,
        query_route: Optional[str] = None,
        chat_route: Optional[str] = None,
        embed_route: Optional[str] = None,
        protocol_options: Optional[Mapping[str, Any]] = None,
    ) -> None: ...
class EndpointRef:
    @staticmethod
    def local(id: str) -> EndpointRef: ...
    @staticmethod
    def gateway(id: str) -> EndpointRef: ...
    @staticmethod
    def provider(id: str) -> EndpointRef: ...
    @property
    def kind(self) -> str: ...

class LocalModelDescriptor:
    def __init__(
        self,
        model_path: PathLike,
        config: Optional[NativeRuntimeConfig] = None,
    ) -> None: ...

class ProviderDescriptor:
    def __init__(
        self,
        provider: Literal["openai", "anthropic", "openai_compatible", "openai-compatible"],
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        version: Optional[str] = None,
        auth_header_name: Optional[str] = None,
        auth_header_value: Optional[str] = None,
        static_headers: Optional[Sequence[tuple[str, str]]] = None,
    ) -> None: ...

EndpointDescriptor = Union[
    LocalModelDescriptor,
    ProviderDescriptor,
    GatewayDescriptor,
]

class SippTextOptions:
    def __init__(
        self,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> None: ...

class LocalTextOptions:
    def __init__(
        self,
        *,
        context_key: Optional[str] = None,
        grammar: Optional[str] = None,
        json_schema: Optional[str] = None,
        sampling: Optional[SamplingRuntimeOverride] = None,
        media: Optional[Sequence[bytes]] = None,
    ) -> None: ...

class LocalEmbedOptions:
    def __init__(
        self,
        *,
        context_key: Optional[str] = None,
        normalize: Optional[bool] = None,
    ) -> None: ...

class SippTokenIterator(Iterator[TokenBatch]): ...

class SippTextRun:
    def result(self) -> SippTextResponse: ...
    def tokens(self) -> SippTokenIterator: ...

class SippEmbeddingRun:
    def result(self) -> SippEmbeddingResponse: ...

class SippClient:
    def __init__(self) -> None: ...
    def add(
        self,
        id: str,
        descriptor: EndpointDescriptor,
    ) -> EndpointRef: ...
    def query(
        self,
        prompt: str,
        *,
        endpoint: Optional[EndpointRef] = None,
        options: Optional[SippTextOptions] = None,
        local: Optional[LocalTextOptions] = None,
        endpoint_options: Optional[EndpointOptions] = None,
        provider_options: Optional[ProviderOptions] = None,
        emit_tokens: bool = False,
    ) -> SippTextRun: ...
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        endpoint: Optional[EndpointRef] = None,
        options: Optional[SippTextOptions] = None,
        local: Optional[LocalTextOptions] = None,
        endpoint_options: Optional[EndpointOptions] = None,
        provider_options: Optional[ProviderOptions] = None,
        emit_tokens: bool = False,
    ) -> SippTextRun: ...
    def embed(
        self,
        input: str,
        *,
        endpoint: Optional[EndpointRef] = None,
        local: Optional[LocalEmbedOptions] = None,
        endpoint_options: Optional[EndpointOptions] = None,
        provider_options: Optional[ProviderOptions] = None,
    ) -> SippEmbeddingRun: ...

def backend_observability_json(include_details: bool = True) -> str: ...
def get_active_backend() -> ActivePythonBackend: ...
def set_llama_log_quiet(quiet: bool) -> None: ...
