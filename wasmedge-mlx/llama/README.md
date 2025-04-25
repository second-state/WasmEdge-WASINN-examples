# MLX example with WasmEdge WASI-NN MLX plugin

This example demonstrates using WasmEdge WASI-NN MLX plugin to perform an inference task with LLM model.

## Supported Models

| Family | Models |
|--------|--------|
| LLaMA 2 | llama_2_7b_chat_hf |
| LLaMA 3 | llama_3_8b |
| TinyLLaMA | tiny_llama_1.1B_chat_v1.0 |

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

In this example, we will use `tiny_llama_1.1B_chat_v1.0`, which you can change to `llama_2_7b_chat_hf` or `llama_3_8b`.

``` bash
# Download model weight
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
# Download tokenizer
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
```

## Build wasm

Run the following command to build wasm, the output WASM file will be at `target/wasm32-wasip1/release/`

```bash
cargo build --target wasm32-wasip1 --release
```
## Execute 

Execute the WASM with the `wasmedge` using nn-preload to load model. 

``` bash
wasmedge --dir .:. \
 --nn-preload default:mlx:AUTO:model.safetensors \
  ./target/wasm32-wasip1/release/wasmedge-mlx.wasm default

```

If your model has multiple weight files, you need to provide all in the nn-preload.

For example:
``` bash
wasmedge --dir .:. \
 --nn-preload default:mlx:AUTO:llama2-7b/model-00001-of-00002.safetensors:llama2-7b/model-00002-of-00002.safetensors \
  ./target/wasm32-wasip1/release/wasmedge-mlx.wasm default
```

## Other 

There are some metadata for MLX plugin you can set.

### Basic setting

- model_type (required): LLM model type.
- tokenizer (required): tokenizer.json path
- max_token (option): maximum generate token number, default is 1024.
- enable_debug_log (option): if print debug log, default is false.

### Quantization

The following three parameters need to be set together.
- is_quantized (option): If the weight is quantized. If is_quantized is false, then MLX backend will quantize the weight. 
- group_size (option): The group size to use for quantization.
- q_bits (option): The number of bits to quantize to.

``` rust
let graph = GraphBuilder::new(GraphEncoding::Mlx, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&json!({"model_type": "tiny_llama_1.1B_chat_v1.0", "tokenizer":tokenizer_path, "max_token":100}))
```