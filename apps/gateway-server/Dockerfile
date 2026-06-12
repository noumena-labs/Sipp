ARG SIPP_GATEWAY_BUILDER_IMAGE=rust:bookworm
ARG SIPP_GATEWAY_RUNTIME_IMAGE=debian:bookworm-slim

FROM ${SIPP_GATEWAY_BUILDER_IMAGE} AS builder

ARG SIPP_GATEWAY_BACKEND
ARG SIPP_GATEWAY_INSTALL_RUSTUP=0
ARG SIPP_CUDA_ARCHITECTURES

ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV SIPP_CUDA_ARCHITECTURES=${SIPP_CUDA_ARCHITECTURES}
ENV PATH=/usr/local/cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        glslang-tools \
        libvulkan-dev \
        ninja-build \
        pkg-config \
        unzip \
    && if [ "${SIPP_GATEWAY_INSTALL_RUSTUP}" = "1" ]; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal --default-toolchain stable; \
    fi \
    && rm -rf /var/lib/apt/lists/*

ENV BUN_INSTALL=/usr/local/bun
ENV PATH=/usr/local/bun/bin:${PATH}

RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /src
COPY . .
RUN test -n "${SIPP_GATEWAY_BACKEND}" \
    || (echo "SIPP_GATEWAY_BACKEND build arg is required" >&2; exit 1) \
    && cargo run --target-dir .build/xtask --package xtask -- \
    build gateway-server --backend "${SIPP_GATEWAY_BACKEND}"

FROM ${SIPP_GATEWAY_RUNTIME_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin sipp \
    && mkdir -p /workspace /models \
    && chown -R sipp:sipp /workspace /models

COPY --from=builder /src/.build/artifacts/gateway-server/ /usr/local/bin/

RUN echo "/usr/local/bin" > /etc/ld.so.conf.d/sipp-gateway.conf \
    && ldconfig

WORKDIR /workspace
USER 10001:10001
EXPOSE 8080 9090
ENTRYPOINT ["sipp-gateway"]
CMD ["serve", "--config", "/etc/sipp/gateway.toml"]
