from __future__ import annotations

from cogentlm import LocalEmbedOptions

from _common import load_client, print_embedding, read_args


def main() -> None:
    model, input_text = read_args("CogentClient embedding smoke input.")
    client = load_client(model, embeddings=True)
    run = client.embed(
        input_text,
        local=LocalEmbedOptions(
            context_key="python-embed-smoke",
            normalize=True,
        ),
    )
    print_embedding(run.result())


if __name__ == "__main__":
    main()
