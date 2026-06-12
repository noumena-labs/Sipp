# Gateway Docker

Gateway Docker workflows use explicit Compose files plus the gateway TOML and a
secrets-only `.env` file.

The separation is strict:

- `.env` contains secret values only.
- TOML contains gateway application configuration.
- Compose YAML contains Docker build, image, port, mount, healthcheck, and
  container orchestration settings.

The container runs:

```bash
sipp-gateway serve --config /etc/sipp/gateway.toml
```

## Files

- `apps/gateway-server/Dockerfile` builds the staged gateway distribution.
- `apps/gateway-server/.env.example` is the secrets-only env template.
- `apps/gateway-server/development.yml.example` builds and runs a local
  model-serving image.
- `apps/gateway-server/development-provider-only.yml.example` builds and runs a
  provider-router image with no model mount.
- `apps/gateway-server/production.yml.example` runs a prebuilt production
  model-serving image.
- `apps/gateway-server/production-provider-only.yml.example` runs a prebuilt
  provider-router image with no model mount.
- `apps/gateway-server/config/*.toml.example` are gateway application config
  templates.

## Local Model-Serving Docker

From the repository root:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
```

Edit `apps/gateway-server/.env` and set only secrets:

```bash
SIPP_GATEWAY_ADMIN_PASSWORD=replace-me
SIPP_GATEWAY_TOKEN=replace-me
OPENAI_API_KEY=replace-me
ANTHROPIC_API_KEY=replace-me
```

Edit `apps/gateway-server/config/development.toml`:

- Set the local target `model` to the path the container sees, usually
  `/models/<file>.gguf`.
- Keep `public_bind = "0.0.0.0:8080"` and
  `management_bind = "0.0.0.0:9090"` so the gateway listens inside the
  container.
- Keep `admin_password_env = "SIPP_GATEWAY_ADMIN_PASSWORD"` unless the
  `.env` secret name also changes.

Edit `apps/gateway-server/development.yml` for Docker concerns such as image
tag, build backend, build images, model mount, port publishing, and
healthcheck.

Build and run with one backend profile. CPU works on Windows, macOS, and
Linux. GPU containers require host-specific device support.

> [!WARNING]
> Windows Docker Desktop does not support the first-party Vulkan gateway path.
> NVIDIA Windows hosts should use the `cuda` profile. Do not use old
> `vulkan-windows` configs; `ggml_vulkan: No devices found` means the
> container cannot enumerate a usable Vulkan physical device.

```bash
# CPU, portable across Windows, macOS, and Linux
docker compose --profile cpu --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --profile cpu --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build gateway-cpu
docker compose --profile cpu --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up gateway-cpu

# CUDA, Linux or Windows Docker Desktop with NVIDIA GPU support
docker compose --profile cuda --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --profile cuda --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build gateway-cuda
docker compose --profile cuda --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up gateway-cuda

# Vulkan on native Linux, uses /dev/dri
docker compose --profile vulkan-linux --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --profile vulkan-linux --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build gateway-vulkan-linux
docker compose --profile vulkan-linux --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up gateway-vulkan-linux
```

If Compose reports orphan containers after switching service names, remove the
old containers once:

```bash
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml down --remove-orphans
```

## Provider-Only Docker

Provider-only Docker runs use the provider-only Compose template and no model
mount:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development-provider-only.yml.example apps/gateway-server/development-provider-only.yml
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
```

Set secrets in `apps/gateway-server/.env`, then run:

```bash
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml build
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml up
```

The provider-only template builds a CPU gateway image because inference happens
upstream.

## Production Docker

Keep production TOML, Compose, and `.env` copies outside the repository:

```bash
mkdir -p /opt/sipp/gateway
cp apps/gateway-server/.env.example /opt/sipp/gateway/.env
cp apps/gateway-server/production.yml.example /opt/sipp/gateway/production.yml
cp apps/gateway-server/config/production.toml.example /opt/sipp/gateway/production.toml
```

Edit `/opt/sipp/gateway/.env` for secret values only. Edit
`/opt/sipp/gateway/production.toml` for gateway runtime configuration.
Edit `/opt/sipp/gateway/production.yml` for image names, host model
mounts, ports, restart policy, and healthcheck.

Deploy with one backend profile:

```bash
# CPU
docker compose --profile cpu --env-file /opt/sipp/gateway/.env -f /opt/sipp/gateway/production.yml config
docker compose --profile cpu --env-file /opt/sipp/gateway/.env -f /opt/sipp/gateway/production.yml up -d gateway-cpu

# CUDA, requires NVIDIA Container Toolkit on the host
docker compose --profile cuda --env-file /opt/sipp/gateway/.env -f /opt/sipp/gateway/production.yml config
docker compose --profile cuda --env-file /opt/sipp/gateway/.env -f /opt/sipp/gateway/production.yml up -d gateway-cuda

# Vulkan on Linux hosts, requires /dev/dri rendering devices
docker compose --profile vulkan-linux --env-file /opt/sipp/gateway/.env -f /opt/sipp/gateway/production.yml config
docker compose --profile vulkan-linux --env-file /opt/sipp/gateway/.env -f /opt/sipp/gateway/production.yml up -d gateway-vulkan-linux
```

For provider-only production, copy `production-provider-only.yml.example` and
`config/provider-only.toml.example` instead.

## Bind And Mount Behavior

The TOML file always uses the same schema, but bind and path interpretation
changes by runtime mode.

| Runtime | TOML bind values | Host exposure | Local target `model` path |
| --- | --- | --- | --- |
| Source/exe | Host addresses, usually `127.0.0.1:*` for development | The process binds directly on the host | Path seen from the process working directory |
| Local Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Compose `ports` map host ports to `127.0.0.1` in local templates | `/models/<file>.gguf` |
| Production Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Compose exposes public and keeps management host-local by default | `/models/<file>.gguf` |
| Provider-only Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Provider-only templates follow the same port rules | No local model path |

Keep management private in production. Put public ingress, TLS, and external
auth controls in front of the public listener when needed.

## Raw Docker Build

Raw Docker commands are supported as an escape hatch. Supply every build arg
explicitly:

```bash
docker build \
  --build-arg SIPP_GATEWAY_BACKEND=vulkan \
  --build-arg SIPP_GATEWAY_BUILDER_IMAGE=rust:bookworm \
  --build-arg SIPP_GATEWAY_RUNTIME_IMAGE=ubuntu:22.04 \
  --build-arg SIPP_GATEWAY_INSTALL_RUSTUP=0 \
  -f apps/gateway-server/Dockerfile \
  -t sipp-gateway:vulkan .
```

## Backend Hardware & Docker Constraints

Published gateway images use backend-specific tags: `latest-cpu`,
`latest-cuda`, and `latest-vulkan`.

Supported first-party Docker profiles:

| Host runtime | GPU vendor | Supported profile | Backend | Notes |
| --- | --- | --- | --- | --- |
| Linux Docker | NVIDIA | `cuda` | CUDA | Recommended NVIDIA GPU path. Requires NVIDIA drivers and container runtime support. |
| Linux Docker | AMD or Intel | `vulkan-linux` | Vulkan | Requires host `/dev/dri` rendering devices and a usable Vulkan driver stack. |
| Linux Docker | No supported GPU | `cpu` | CPU | Portable diagnostic and fallback path. |
| Windows Docker Desktop | NVIDIA | `cuda` | CUDA | Requires Docker Desktop WSL2 GPU support and NVIDIA container GPU passthrough. |
| Windows Docker Desktop | AMD or Intel | `cpu` | CPU | First-party Docker does not support Windows Vulkan GPU inference. |
| macOS Docker | Any | `cpu` | CPU | Metal is available only through native macOS execution, not Linux Docker. |

### CPU Backend (`latest-cpu` / `cpu` profile)
- Standard portable execution. Works on any host without special driver dependencies.
- This is the Docker path for macOS local development.

### CUDA Backend (`latest-cuda` / `cuda` profile)
- Requires the **NVIDIA Container Toolkit** to be installed and configured on the host.
- Requires NVIDIA host GPU drivers.
- Exposed using Docker Compose GPU device reservation capabilities.
- Supported on Linux and Windows Docker Desktop WSL2 hosts with NVIDIA GPU support.

### CUDA Architecture Selection

Set `SIPP_CUDA_ARCHITECTURES` to control the compiled GPU architecture
list. The value is passed verbatim to CMake, so use semicolon-separated
entries. In Docker builds, pass it as the `SIPP_CUDA_ARCHITECTURES` build
arg; the Compose CUDA service forwards it to the builder stage.

Defaults are layered:

- `cargo xtask build` CUDA targets (node, python, cli, gateway-server) default
  to the portable cloud GPU list below so packaged artifacts stay
  deterministic across build hosts. Docker gateway builds run xtask, so they
  inherit the same default when the build arg is empty.
- Raw `cargo build` of `sipp-sys` outside xtask does not set
  `CMAKE_CUDA_ARCHITECTURES`, which lets vendored llama.cpp choose
  CUDA-version-aware defaults for the local toolkit.

Portable cloud GPU release images use:

```text
75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real
```

| Entry | Target GPUs |
| --- | --- |
| `75-virtual` | T4 and other Turing cloud GPUs |
| `80-virtual` | A100 and other Ampere data-center GPUs |
| `86-real` | A10, A40, RTX A6000-class Ampere |
| `89-real` | L4, L40S, Ada |
| `90-virtual` | H100, H200 Hopper |
| `120a-real` | Blackwell architecture-specific target |
| `121a-real` | Newer Blackwell architecture-specific target |

For faster builds targeting a known GPU, narrow the list. For example, `80`
for A100 only, `90` for H100/H200 only, or `89` for L4/L40S only.

CUDA 13 removes offline compilation support for GPU architectures before
compute capability 7.5, so `61` (Pascal) and `70` (Volta) are excluded from
CUDA 13 builds. Supporting those GPUs requires a separate legacy build using a
CUDA 12.x toolkit image with an explicit `SIPP_CUDA_ARCHITECTURES` list.

The `a`-suffix Blackwell entries are architecture-specific and not
forward-compatible; keep them aligned with the targets vendored llama.cpp
uses. Plain TensorRT-free CUDA images are the default because the gateway
links against CUDA runtime libraries only; use TensorRT images only if a
TensorRT dependency is introduced.

### Vulkan Backend (`latest-vulkan` image)
- Supported first-party Docker profile is Linux-only: `vulkan-linux`.
- Linux runs expose host rendering devices with `/dev/dri:/dev/dri`.
- Windows Docker Desktop Vulkan is unsupported for gateway inference. NVIDIA Windows hosts should use `cuda` instead.
- The runtime container packages `libvulkan1` and `mesa-vulkan-drivers` for the supported Linux Vulkan profile.

### Apple Metal Backend (macOS hypervisor constraints)
> [!WARNING]
> **Metal cannot run inside a standard Linux Docker container.**
> Docker on macOS runs within a virtualized Linux hypervisor VM. Apple does not support direct forwarding of the Metal GPU API from macOS into Linux VMs.
>
> Due to this hard architectural boundary:
> 1. **Docker Limitation:** Running the gateway container on macOS will result in a CPU-only fallback or Vulkan device discovery failure (no Metal GPU acceleration).
> 2. **Native Execution:** To utilize Apple Silicon GPU acceleration (Metal), macOS users must compile and run the gateway server natively:
>    ```bash
>    cargo xtask build gateway-server --backend metal
>    ./.build/artifacts/gateway-server/sipp-gateway serve --config apps/gateway-server/config/development.toml
>    ```

## Health Check

The Compose templates probe the management readiness route:

```bash
curl --fail --silent http://127.0.0.1:9090/readyz
```

If you change the readiness route in TOML, update the Compose healthcheck too.
