from __future__ import annotations

from _common import (
    load_openai_provider_client,
    print_text,
    provider_endpoint,
    read_provider_args,
    text_options,
)


def main() -> None:
    model, prompt = read_provider_args("Write one sentence about provider inference.")
    client = load_openai_provider_client(model)
    run = client.query(
        prompt,
        endpoint=provider_endpoint(model),
        options=text_options(),
    )
    print_text(run.result())


if __name__ == "__main__":
    main()
