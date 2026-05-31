from __future__ import annotations

from cogentlm import ChatMessage, LocalTextOptions

from _common import load_client, print_text, read_args, text_options


def main() -> None:
    model, prompt = read_args("Explain the CogentClient API in one sentence.")
    client = load_client(model)
    run = client.chat(
        [
            ChatMessage("system", "Answer concisely."),
            ChatMessage("user", prompt),
        ],
        options=text_options(),
        local=LocalTextOptions(context_key="python-chat-smoke"),
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
