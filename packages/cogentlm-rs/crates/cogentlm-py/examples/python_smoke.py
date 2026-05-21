from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from cogentlm import (
    ChatMessage,
    ContextRuntimeConfig,
    ModelPlacementConfig,
    ModelLoadOptions,
    ModelService,
    NativeRuntimeConfig,
    QueryOptions,
    SamplingRuntimeConfig,
    backend_observability_json,
    set_llama_log_quiet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CogentLM Python binding smoke test")
    parser.add_argument("model")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Describe browser LLM inference.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "metal", "vulkan", "webgpu"),
        default="auto",
    )
    parser.add_argument(
        "--model-store",
        default=str(Path(tempfile.gettempdir()) / "cogentlm-rs-model-store"),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-llama", action="store_true")
    args = parser.parse_args()
    set_llama_log_quiet(not args.verbose_llama)

    runtime = NativeRuntimeConfig(
        placement=ModelPlacementConfig(
            gpu_layers=None if args.gpu_layers is None else {"count": args.gpu_layers},
        ),
        context=ContextRuntimeConfig(
            n_ctx=args.ctx_size,
            n_threads=args.threads,
            n_threads_batch=args.threads,
        ),
        sampling=SamplingRuntimeConfig(
            temperature=args.temperature,
            seed=args.seed,
        ),
    )
    load_options = ModelLoadOptions(backend=args.backend, stats="basic", runtime=runtime)
    options = QueryOptions(max_tokens=args.max_tokens)

    print("backend_before_load=" + backend_observability_json(True))
    engine = ModelService(args.model_store)
    try:
        loaded = engine.load_path(args.model, load_options)
        print(f"loaded_model={loaded['model']}")
        print(f"selected_backend={loaded['backend']}")
        print("backend_after_load=" + backend_observability_json(True))
        print(f"engine_state_after_load={engine.state()}")

        # Streaming chat: on_tokens receives TokenBatch dicts.
        pieces: list[str] = []

        def on_tokens(batch: dict[str, object]) -> None:
            text = str(batch["text"])
            pieces.append(text)
            print(text, end="", flush=True)

        print("\nchat_stream=", end="", flush=True)
        result = engine.chat(
            [ChatMessage.user(args.prompt)], options, on_tokens=on_tokens
        )
        print()  # newline after streaming output
        streamed_text = "".join(pieces)
        if streamed_text != result["text"]:
            raise RuntimeError("streamed token batches did not match final response text")

        stats = result["stats"]
        print(f"finish_reason={result['finish_reason']}")
        print(f"stream_batches={len(pieces)}")
        print(f"engine_state_after_chat={engine.state()}")
        event_counts: dict[str, int] = {}
        for event in engine.drain_events():
            event_counts[event["type"]] = event_counts.get(event["type"], 0) + 1
        print(f"engine_events={event_counts}")
        print(
            "metrics="
            f"ttft_ms:{stats['ttft_ms']} "
            f"decode_ms:{stats['decode_ms']:.3f} "
            f"output_tokens:{stats['output_tokens']} "
            f"tps:{stats['tokens_per_second']}"
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
