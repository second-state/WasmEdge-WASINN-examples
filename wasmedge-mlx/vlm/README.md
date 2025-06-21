# VLM example with WasmEdge WASI-NN MLX plugin

This example demonstrates using WasmEdge WASI-NN MLX plugin to perform an inference task with VLM model.

## Supported Models

| Family | Models |
|--------|--------|
| Gemma 3 | gemma-3-4b-pt-bf16 |

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

In this example, we will use `gemma-3-4b-it-4bit`.

``` bash
git clone https://huggingface.co/mlx-community/gemma-3-4b-it-4bit
```

## Build wasm

Run the following command to build wasm, the output WASM file will be at `target/wasm32-wasip1/release/`.
Then we use AOT-compiled WASM to improve the performance.

```bash
cargo build --target wasm32-wasip1 --release
wasmedge compiler ./target/wasm32-wasip1/release/wasmedge-vlm.wasm wasmedge-vlm_aot.wasm
```
## Execute 

Execute the WASM with the `wasmedge` using nn-preload to load model. 

``` bash
# Download sample image
wget https://github.com/WasmEdge/WasmEdge/raw/master/docs/wasmedge-runtime-logo.png 


wasmedge --dir .:. \
--nn-preload default:mlx:AUTO:gemma-3-4b-it-4bit/model.safetensors \
./wasmedge-vlm_aot.wasm default gemma-3-4b-it-4bit

```

If your model has multiple weight files, you need to provide all in the nn-preload.

For example:
``` bash
wasmedge --dir .:. \
--nn-preload default:mlx:AUTO:gemma-3-4b-it-4bit/model-00001-of-00002.safetensors:gemma-3-4b-it-4bit/model-00002-of-00002.safetensors \
./target/wasm32-wasip1/release/wasmedge-vlm.wasm default
```

## Other 

There are some metadata for MLX plugin you can set.

### Basic setting

- model_type (required): model type.
- max_token (option): maximum generate token number, default is 1024.
- enable_debug_log (option): if print debug log, default is false.