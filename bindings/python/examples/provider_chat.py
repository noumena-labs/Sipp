from __future__ import annotations

from cogentlm import ChatMessage

from _common import (
    load_openai_provider_client,
    print_text,
    provider_endpoint,
    read_provider_args,
    text_options,
)


def main() -> None:
    model, prompt = read_provider_args("Explain provider inference in one sentence.")
    client = load_openai_provider_client(model)
    run = client.chat(
        [
            ChatMessage("system", "Answer concisely."),
            ChatMessage("user", prompt),
        ],
        endpoint=provider_endpoint(model),
        options=text_options(),
        stream_tokens=True,
    )
    streamed = ""
    for batch in run.tokens():
        print(batch["text"], end="", flush=True)
        streamed += batch["text"]
    print()
    result = run.result()
    if streamed != result["text"]:
        raise RuntimeError("streamed token batches did not match final response text")
    print_text(result)


if __name__ == "__main__":
    main()
