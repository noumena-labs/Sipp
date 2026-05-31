from __future__ import annotations

from _common import (
    CogentClient,
    add_openai_remote,
    print_embedding,
    read_remote_args,
)


def main() -> None:
    model, input_text = read_remote_args("CogentClient remote embedding smoke input.")
    client = CogentClient()
    endpoint = add_openai_remote(client, model)
    run = client.embed(
        input_text,
        endpoint=endpoint,
    )
    print_embedding(run.result())


if __name__ == "__main__":
    main()
