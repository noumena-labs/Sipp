from __future__ import annotations

import os
import sys

from sipp import (
    ChatMessage,
    SippClient,
    SippTextOptions,
    ProviderDescriptor,
)

from _support import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    float_env,
    int_env,
    print_text,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
OPENAI_DEFAULT_MODEL = "gpt-5-mini"


def provider_descriptor() -> ProviderDescriptor:
    provider = provider_name()
    if provider == "gemini":
        return ProviderDescriptor(
            "openai_compatible",
            env_any(("SIPP_PROVIDER_MODEL", "GEMINI_MODEL"), GEMINI_DEFAULT_MODEL),
            api_key=required_env_any(("SIPP_PROVIDER_API_KEY", "GEMINI_API_KEY")),
            base_url=env("SIPP_PROVIDER_BASE_URL") or GEMINI_BASE_URL,
            timeout_ms=provider_timeout_ms(),
        )
    if provider == "openai":
        return ProviderDescriptor(
            "openai",
            env_any(("SIPP_PROVIDER_MODEL", "OPENAI_MODEL"), OPENAI_DEFAULT_MODEL),
            api_key=required_env_any(("SIPP_PROVIDER_API_KEY", "OPENAI_API_KEY")),
            base_url=optional_env_any(("SIPP_PROVIDER_BASE_URL", "OPENAI_BASE_URL")),
            timeout_ms=provider_timeout_ms(),
        )
    if provider == "anthropic":
        return ProviderDescriptor(
            "anthropic",
            required_env_any(("SIPP_PROVIDER_MODEL", "ANTHROPIC_MODEL")),
            api_key=required_env_any(("SIPP_PROVIDER_API_KEY", "ANTHROPIC_API_KEY")),
            base_url=optional_env_any(
                ("SIPP_PROVIDER_BASE_URL", "ANTHROPIC_BASE_URL")
            ),
            version=env("ANTHROPIC_VERSION"),
            timeout_ms=provider_timeout_ms(),
        )
    if provider == "openai_compatible":
        return ProviderDescriptor(
            "openai_compatible",
            required_env_any(("SIPP_PROVIDER_MODEL",)),
            base_url=required_env_any(("SIPP_PROVIDER_BASE_URL",)),
            timeout_ms=provider_timeout_ms(),
            **openai_compatible_auth(),
        )
    raise RuntimeError(
        "SIPP_PROVIDER must be gemini, openai, anthropic, or openai_compatible"
    )


def openai_compatible_auth() -> dict[str, str]:
    header_name = env("SIPP_PROVIDER_AUTH_HEADER_NAME")
    header_value = env("SIPP_PROVIDER_AUTH_HEADER_VALUE")
    if header_name is not None or header_value is not None:
        if header_name is None or header_value is None:
            raise RuntimeError(
                "SIPP_PROVIDER_AUTH_HEADER_NAME and "
                "SIPP_PROVIDER_AUTH_HEADER_VALUE must be set together"
            )
        return {
            "auth_header_name": header_name,
            "auth_header_value": header_value,
        }
    return {"api_key": required_env_any(("SIPP_PROVIDER_API_KEY",))}


def text_options() -> SippTextOptions:
    return SippTextOptions(
        max_tokens=int_env("SIPP_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        temperature=float_env("SIPP_TEMPERATURE", DEFAULT_TEMPERATURE),
        top_p=float_env("SIPP_TOP_P", DEFAULT_TOP_P),
    )


def provider_name() -> str:
    return (env("SIPP_PROVIDER") or "gemini").lower().replace("-", "_")


def provider_timeout_ms() -> int | None:
    return int_env("SIPP_PROVIDER_TIMEOUT_MS", 30_000)


def env(name: str) -> str | None:
    value = os.getenv(name)
    return None if value is None or value == "" else value


def env_any(names: tuple[str, ...], default: str | None = None) -> str:
    value = optional_env_any(names)
    if value is not None:
        return value
    if default is None:
        raise RuntimeError(f"{' or '.join(names)} is required")
    return default


def optional_env_any(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = env(name)
        if value is not None:
            return value
    return None


def required_env_any(names: tuple[str, ...]) -> str:
    return env_any(names)


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Say hello from a direct provider."

    # Direct providers belong in trusted Python processes. Browser code should
    # call a gateway or application route instead of holding provider credentials.
    client = SippClient()
    endpoint = client.add("provider", provider_descriptor())
    run = client.chat(
        [ChatMessage("user", prompt)],
        endpoint=endpoint,
        options=text_options(),
    )
    print_text(run.result())


if __name__ == "__main__":
    main()
