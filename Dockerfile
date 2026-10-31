# Multi-stage professional Dockerfile for loopmoe / miniinfer
FROM rust:1.82-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --features server,cuda

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app
COPY --from=builder /app/target/release/sentinel-server /usr/local/bin/
EXPOSE 8000
CMD ["sentinel-server"]
