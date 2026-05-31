from __future__ import annotations

from _common import (
    load_openai_provider_client,
    print_embedding,
    provider_endpoint,
    read_provider_args,
)


def main() -> None:
    model, input_text = read_provider_args("CogentClient provider embedding smoke input.")
    client = load_openai_provider_client(model)
    run = client.embed(
        input_text,
        endpoint=provider_endpoint(model),
    )
    print_embedding(run.result())


if __name__ == "__main__":
    main()
