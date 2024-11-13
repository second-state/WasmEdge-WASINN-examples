# Basic Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example for the models without prompt templates.

## StarCoder2

### Get the Model

This example is for the models without prompt template. For example, the `StarCoder2` model.

Download the model:

```bash
curl -LO https://huggingface.co/second-state/StarCoder2-7B-GGUF/resolve/main/starcoder2-7b-Q5_K_M.gguf
```

### Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

- For GPU offloading, please adjust the `n-gpu-layers` options to the number of layers that you want to offload to the GPU.
- When using the `StarCoder2` model, the `n-predict` option can be used to adjust the number of predictions. Since the inference may not stop as expected, it is recommended to set a limit for the number of predictions.

### Execute

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

```console
$ wasmedge --dir .:. \
  --env n_predict=100 \
  --nn-preload default:GGML:AUTO:starcoder2-7b-Q5_K_M.gguf \
  wasmedge-ggml-basic.wasm default

USER:
def print_hello_world():
ASSISTANT:

    print("Hello World!")

def main():
    print_hello_world()


if __name__ == "__main__":
    main()/README.md
# python-learning

This repository is for learning Python.

## References

* [Python 3.8.0 Documentation](https://docs.python.org/3/)
```

## Grok-1

### Get the Model

Download the split grok-1 model:

```bash
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00001-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00002-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00003-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00004-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00005-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00006-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00007-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00008-of-00009.gguf
curl -LO https://huggingface.co/Arki05/Grok-1-GGUF/resolve/main/grok-1-Q2_K-split-00009-of-00009.gguf
```

### Execute

Since the `ggml` plugin supports the split `gguf` model, you can set the `nn-preload` option with the initial split model. The plugin will then automatically load the remaining split models from the same directory.

```console
$ wasmedge --dir .:. \
  --env n_predict=100 \
  --nn-preload default:GGML:AUTO:grok-1-Q2_K-split-00001-of-00009.gguf \
  wasmedge-ggml-basic.wasm default 'hello'
```

## TriLM & BitNet Models

After the following pull requests are merged, the `TriLM` and `BitNet` models will be supported by the `ggml` plugin with model type `TQ1_0` and `TQ2_0`:
- https://github.com/ggerganov/llama.cpp/pull/7931
- https://github.com/ggerganov/llama.cpp/pull/8151

### Get the Model

Download the `TriLM` model:

```bash
curl -LO https://huggingface.co/Green-Sky/TriLM_3.9B-GGUF/resolve/main/TriLM_3.9B_Unpacked-4.0B-TQ2_0.gguf
```

### Execute

```console
$ wasmedge --dir .:. \
  --env n_predict=100 \
  --nn-preload default:GGML:AUTO:TriLM_3.9B_Unpacked-4.0B-TQ2_0.gguf \
  wasmedge-ggml-basic.wasm default
```
