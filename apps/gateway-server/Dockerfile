FROM rust:bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cargo xtask build core

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin cogentlm

COPY --from=builder /src/.build/cargo/release/cogentlm-gateway /usr/local/bin/cogentlm-gateway

USER 10001:10001
EXPOSE 8080 9090
ENTRYPOINT ["cogentlm-gateway"]
CMD ["serve", "--config", "/etc/cogentlm/gateway.toml"]
