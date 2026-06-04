from __future__ import annotations

from _common import (
    CogentClient,
    add_gateway_remote,
    print_text,
    read_remote_args,
    text_options,
)


def main() -> None:
    alias, prompt = read_remote_args("Write one sentence about remote inference.")
    client = CogentClient()
    endpoint = add_gateway_remote(client, alias)
    run = client.query(
        prompt,
        endpoint=endpoint,
        options=text_options(),
    )
    print_text(run.result())


if __name__ == "__main__":
    main()
