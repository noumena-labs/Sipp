# Providers

CogentLM can call external providers either directly from server-side packages
or indirectly through a gateway. For most applications, provider credentials
belong in the gateway or another trusted server process.

## Recommended Pattern

Use a gateway when:

- Browser code would otherwise need provider credentials.
- Multiple applications need the same target policy.
- You want consistent authentication, observability, and rate controls.
- Keep provider-specific options behind an application boundary.

Use a direct provider descriptor only when the calling process is trusted and
owns the credential lifecycle.

## Provider Options

Pass provider-only request fields as provider options. Gateway-specific profile
extensions use endpoint options or descriptor-level protocol options.

See the Rust OpenAI provider example for the direct provider path and the
gateway server reference for provider-backed serving.
