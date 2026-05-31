from __future__ import annotations

from cogentlm import LocalTextOptions

from _common import load_client, print_text, read_args, text_options


def main() -> None:
    model, prompt = read_args("Write one sentence about local inference.")
    client = load_client(model)
    run = client.query(
        prompt,
        options=text_options(),
        local=LocalTextOptions(context_key="python-query-smoke"),
    )
    print_text(run.result())


if __name__ == "__main__":
    main()
