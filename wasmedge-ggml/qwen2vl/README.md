# Qwen-2VL Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of the Qwen2-VL model.

## Get Qwen2-VL Model

In this example, we are going to use the pre-converted [Qwen2-VL-2B](https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/tree/main) model.

Download the model:

```bash
curl -LO https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-vision-encoder.gguf
curl -LO https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q5_K_M.gguf
```

## Prepare the Image

Download the image you want to perform inference on:

```bash
curl -LO https://llava-vl.github.io/static/images/monalisa.jpg
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

In Qwen2-VL inference, we recommend to use the `ctx-size` at least `4096` for better results.

```rust
options.insert("ctx-size", Value::from(4096));
```

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:Qwen2-VL-2B-Instruct-Q5_K_M.gguf \
  --env mmproj=Qwen2-VL-2B-Instruct-vision-encoder.gguf \
  --env image=monalisa.jpg \
  --env ctx_size=4096 \
  wasmedge-ggml-qwen2vl.wasm default
```
