
# Basic Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example for the models without prompt templates.

## Get the Model

This example is for the models without prompt template. For example, the `StarCoder2` model.

Download the model:

```bash
curl -LO https://huggingface.co/second-state/StarCoder2-7B-GGUF/resolve/main/starcoder2-7b-Q5_K_M.gguf
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

- For GPU offloading, please adjust the `n-gpu-layers` options to the number of layers that you want to offload to the GPU.
- When using the `StarCoder2` model, the `n-predict` option can be used to adjust the number of predictions. Since the inference may not stop as expected, it is recommended to set a limit for the number of predictions.

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

```console
$ wasmedge --dir .:. \
  --env n_predict=100 \
  --nn-preload default:GGML:AUTO:/disk/starcoder2-7b-Q5_K_M.gguf \
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
