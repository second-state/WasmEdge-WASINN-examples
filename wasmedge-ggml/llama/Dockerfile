# syntax=docker/dockerfile:1

# stage 1: building the wasm binary
# using rust slim image to keep things light
FROM rust:1.77-slim-bookworm AS builder

WORKDIR /src

# copying the cargo config and source code
COPY Cargo.toml .
COPY src ./src

# adding the wasm32-wasi target since rust doesnt do it by default
RUN rustup target add wasm32-wasi

# compiling the app in release mode to get the .wasm binary
RUN cargo build --target wasm32-wasi --release

# stage 2: setting up the runtime environment
# switching to ubuntu for the actual execution
FROM ubuntu:22.04

# installing git and python3 because the installer script needs them
# also adding libgomp1 for the ai math computations
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# downloading wasmedge 0.16.0 and adding the wasi-nn plugin for running llama
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- \
    -v 0.16.0 \
    -p /usr/local \
    --plugins wasi_nn-ggml

WORKDIR /app

# copying just the compiled wasm file from the builder stage
COPY --from=builder /src/target/wasm32-wasi/release/*.wasm ./llama-chat.wasm

# entrypoint for running the app
# remembering to mount the model to /app/model.gguf when running
ENTRYPOINT ["wasmedge", "--dir", ".:.", "--nn-preload", "default:GGML:CPU:/app/model.gguf", "llama-chat.wasm", "default"]
