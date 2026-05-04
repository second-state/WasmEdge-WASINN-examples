# Gemma-4 Example For WASI-NN With GGML Backend
## Requirements

- WasmEdge with the WASI-NN GGML plugin installed
- Rust target `wasm32-wasip1`
- A Gemma 4 GGUF model file and its matching multimodal projector (`mmproj`) file (option)

## Get A Gemma-4 Model

Download a Gemma 4 multimodal GGUF model and the matching `mmproj` file from Hugging Face or your own model store, then place them in this directory.

Example filenames:

```text
gemma-4-*.gguf
mmproj-gemma-4-*.gguf
```

Replace the filenames used below with the exact files you downloaded.

In this example, we use the two following command to download GGUF model and `mmproj` file.
```bash
curl -LO https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf
curl -LO https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf
```

## Prepare The Image

```bash
curl -LO https://llava-vl.github.io/static/images/monalisa.jpg
```

## Build The Wasm Binary

```bash
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1
```

The output binary will be:

```text
target/wasm32-wasip1/release/wasmedge-ggml-gemma-4.wasm
```

## Execute


```console
$ ${WASMEDGE:-wasmedge} --dir .:. \
  --nn-preload default:GGML:AUTO:gemma-4-E4B-it-Q4_K_M.gguf \
  target/wasm32-wasip1/release/wasmedge-ggml-gemma-4.wasm default
```

## Optional Environment Variables

- `system_prompt`: Override the default system prompt. Default: `You are a helpful assistant.`
- `enable_thinking`: Enable or disable Gemma 4 thinking mode. Default: `true`
- `enable_log`: Enable WASI-NN backend logging. Default: `false`
- `enable_debug_log`: Enable verbose backend logging. Default: `false`
- `ctx_size`: Context size. Default: `4096`
- `n_gpu_layers`: Number of layers offloaded to GPU. Default: `0`

Example:

```console
$ ${WASMEDGE:-wasmedge} --dir .:. \
  --env mmproj=mmproj-F16.gguf \
  --env image=monalisa.jpg \
  --env system_prompt="You are a concise visual assistant." \
  --env enable_thinking=false \
  --nn-preload default:GGML:AUTO:gemma-4-E4B-it-Q4_K_M.gguf \
  target/wasm32-wasip1/release/wasmedge-ggml-gemma-4.wasm default "Summarize the image in one sentence."
```

## Prompt Format

This example formats requests for Gemma 4 as:

```text
<|turn>system
<|think|>{system prompt}<turn|>
<|turn>user
<image>
{user prompt}<turn|>
<|turn>model
```

When `enable_thinking=false`, the `<|think|>` prefix is omitted from the system turn.
When multimodal input is enabled via `mmproj` and `image`, the prompt must contain the literal `<image>` marker so the number of image markers matches the number of input bitmaps expected by the WasmEdge GGML backend.
