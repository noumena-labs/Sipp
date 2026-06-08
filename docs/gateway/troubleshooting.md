# Gateway Troubleshooting

Use this page when the first-party gateway starts, serves, or responds
differently than expected.

## `check` Succeeds But `serve` Fails

`check` parses and validates TOML only. It does not read token environment
variables, load model files, contact providers, or bind ports.

If `serve` fails after `check` succeeds, verify:

- Bearer token env vars named by `[[tokens]].env` are present and non-empty.
- Provider secret env vars such as `OPENAI_API_KEY` are present for provider
  targets.
- Local GGUF paths exist from the process point of view.
- `public_bind` and `management_bind` are available and not already in use.
- Requested GPU backends were compiled and are available on the host.

## Missing DLL Or Shared Library

Direct executable runs must put `.build/artifacts/gateway-server` on the
dynamic loader path. The staged executable depends on runtime libraries and
GGML backend plugins in that same directory.

- Windows: prepend the artifact directory to `PATH`.
- Linux: prepend the artifact directory to `LD_LIBRARY_PATH`.
- macOS: prepend the artifact directory to `DYLD_LIBRARY_PATH`.

The `clm run gateway-server ...` workflow handles this automatically.

## Relative Model Path Is Wrong

Relative local target `model` paths resolve from the process working directory.
`clm run gateway-server ...` runs from the workspace root. Direct executable
commands run wherever the shell is currently located.

Use absolute model paths when starting the executable from another directory.
For Docker, use the container path, not the host path.

## Docker Port Is Published But Host Cannot Connect

In Docker mode, `public_bind` and `management_bind` are addresses inside the
container. Use container listener values such as:

```toml
public_bind = "0.0.0.0:8080"
management_bind = "0.0.0.0:9090"
```

Then use Compose `ports` to control host exposure. The local Compose template
maps both host ports to `127.0.0.1` for workstation-only access.

## `401 Unauthorized`

The public route did not receive a valid bearer token. Check:

- Header is `Authorization: Bearer <token>`.
- Token value matches the environment variable named by a `[[tokens]]` block.
- Token contains no whitespace.
- The gateway process was restarted after changing the token environment.

## `403 Forbidden`

The bearer token is valid, but its `targets` allowlist does not include the
request `model` target. Add the target name to the relevant `[[tokens]]` block
or use a token that grants that target.

## `404 Target Not Found`

The request `model` value does not match any configured `[[targets]].name`.
The `model` field in public HTTP requests is a public gateway target name, not
necessarily the provider model or GGUF file name.

## CORS Failure In Browser

Browser requests require the public listener to allow the page origin. Add the
exact origin to `allowed_origins`:

```toml
allowed_origins = ["http://localhost:5173"]
```

An empty `allowed_origins` array disables the CORS layer.

## GPU Backend Fails

Explicit local target backends fail when the backend was not compiled or is not
available at runtime. Try `backend = "auto"` or `backend = "cpu"` first.

Docker GPU builds also require host runtime support:

- CUDA requires NVIDIA host drivers and container runtime support.
- Vulkan requires GPU device access, Vulkan loader, and driver support.
- Metal is macOS-only and not available from Linux Docker.

## Admin Dashboard Login Fails

The dashboard password is the literal `admin_password` from the selected TOML
file. If you copied a config, confirm the gateway is using that copy through
`--config` or `COGENTLM_GATEWAY_CONFIG`.

The dashboard is served on the management listener only.

