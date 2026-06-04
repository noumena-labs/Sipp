from __future__ import annotations

from pathlib import Path

from cogentlm import ChatMessage, LocalTextOptions

from _common import load_client, print_text, read_vision_args, text_options


def main() -> None:
    model, projector, image, prompt = read_vision_args("Describe this image in one sentence.")
    client = load_client(model, projector_path=projector)
    run = client.chat(
        [
            ChatMessage("user", prompt),
        ],
        options=text_options(),
        local=LocalTextOptions(
            context_key="python-vision-chat-smoke",
            media=[Path(image).read_bytes()],
        ),
        emit_tokens=True,
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
