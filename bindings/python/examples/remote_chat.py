from __future__ import annotations

from cogentlm import ChatMessage

from _common import (
    CogentClient,
    add_openai_remote,
    print_text,
    read_remote_args,
    text_options,
)


def main() -> None:
    model, prompt = read_remote_args("Explain remote inference in one sentence.")
    client = CogentClient()
    endpoint = add_openai_remote(client, model)
    run = client.chat(
        [
            ChatMessage("system", "Answer concisely."),
            ChatMessage("user", prompt),
        ],
        endpoint=endpoint,
        options=text_options(),
        token_delivery="batch",
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
