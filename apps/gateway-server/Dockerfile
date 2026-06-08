ARG COGENTLM_GATEWAY_BUILDER_IMAGE=rust:bookworm
ARG COGENTLM_GATEWAY_RUNTIME_IMAGE=debian:bookworm-slim

FROM ${COGENTLM_GATEWAY_BUILDER_IMAGE} AS builder

ARG COGENTLM_GATEWAY_BACKEND=cpu
ARG COGENTLM_GATEWAY_INSTALL_RUSTUP=0

ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
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
    && if [ "${COGENTLM_GATEWAY_INSTALL_RUSTUP}" = "1" ]; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal --default-toolchain stable; \
    fi \
    && rm -rf /var/lib/apt/lists/*

ENV BUN_INSTALL=/usr/local/bun
ENV PATH=/usr/local/bun/bin:${PATH}

RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /src
COPY . .
RUN cargo xtask build gateway-server --backend "${COGENTLM_GATEWAY_BACKEND}"

FROM ${COGENTLM_GATEWAY_RUNTIME_IMAGE}

ENV LD_LIBRARY_PATH=/usr/local/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin cogentlm \
    && mkdir -p /workspace/.build/models \
    && chown -R cogentlm:cogentlm /workspace

COPY --from=builder /src/.build/artifacts/gateway-server/ /usr/local/bin/

WORKDIR /workspace
USER 10001:10001
EXPOSE 8080 9090
ENTRYPOINT ["cogentlm-gateway"]
CMD ["serve", "--config", "/etc/cogentlm/gateway.toml"]
