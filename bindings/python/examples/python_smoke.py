from __future__ import annotations

import argparse

from cogentlm import (
    ChatMessage,
    CogentClient,
    CogentTextOptions,
    ContextRuntimeConfig,
    ModelPlacementConfig,
    NativeRuntimeConfig,
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
    options = CogentTextOptions(max_tokens=args.max_tokens)

    print("backend_before_load=" + backend_observability_json(True))
    client = CogentClient()
    client.load_engine("default", args.model, runtime)
    print("backend_after_load=" + backend_observability_json(True))

    run = client.chat(
        [ChatMessage("user", args.prompt)],
        options=options,
        stream_tokens=True,
    )
    pieces: list[str] = []
    for batch in run.tokens():
        pieces.append(batch["text"])
        print(batch["text"], end="", flush=True)
    result = run.result()
    print()
    if "".join(pieces) != result["text"]:
        raise RuntimeError("streamed token batches did not match final response text")

    stats = result["local_stats"]
    if stats is None:
        raise RuntimeError("local CogentClient response did not include local_stats")
    print(f"endpoint={result['endpoint']}")
    print(f"finish_reason={result['finish_reason']}")
    print(f"stream_batches={len(pieces)}")
    print(f"text={result['text']}")
    print(
        "metrics="
        f"ttft_ms:{stats['ttft_ms']} "
        f"decode_ms:{stats['decode_ms']:.3f} "
        f"output_tokens:{stats['output_tokens']} "
        f"tps:{stats['tokens_per_second']}"
    )


if __name__ == "__main__":
    main()
