# Whisper example with WasmEdge WASI-NN MLX plugin

This example demonstrates using WasmEdge WASI-NN MLX plugin to perform an inference task with whisper model.

## Install WasmEdge with WASI-NN MLX plugin

The MLX backend relies on [MLX](https://github.com/ml-explore/mlx), but we will auto-download MLX when you build WasmEdge. You do not need to install it yourself. If you want to custom MLX, install it yourself or set the `CMAKE_PREFIX_PATH` variable when configuring cmake.

Build and install WasmEdge from source:

``` bash
cd <path/to/your/wasmedge/source/folder>

cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="mlx"
cmake --build build

# For the WASI-NN plugin, you should install this project.
cmake --install build
```

Then you will have an executable `wasmedge` runtime under `/usr/local/bin` and the WASI-NN with MLX backend plug-in under `/usr/local/lib/wasmedge/libwasmedgePluginWasiNN.so` after installation.

## Download the model and tokenizer

In this example, we will use `whisper-tiny`.

``` bash
git clone https://huggingface.co/grorge123/whisper-tiny
cp -r whisper-tiny/assets .
wget https://raw.githubusercontent.com/ml-explore/mlx-examples/refs/heads/main/whisper/mlx_whisper/assets/multilingual.tiktoken -P assets
```

## Build wasm

Run the following command to build wasm, the output WASM file will be at `target/wasm32-wasip1/release/`.
Then we use AOT-compiled WASM to improve the performance.

```bash
cargo build --target wasm32-wasip1 --release
```
## Execute 

Download the audio and save as `audio.mp3`

Execute the WASM with the `wasmedge` using nn-preload to load model. 

``` bash
wasmedge --dir .:. \
--nn-preload default:mlx:AUTO:whisper-tiny/weights.safetensors \
target/wasm32-wasip1/release/wasmedge-whisper.wasm default whisper-tiny audio.mp3
```