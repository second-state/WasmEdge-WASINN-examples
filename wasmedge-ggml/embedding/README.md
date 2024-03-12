# Embedding Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of generating embeddings.

## Get the Model

In this example, we are going to use the pre-converted `all-MiniLM-L6-v2` model.

Download the model:

```bash
curl -LO https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

```console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  wasmedge-ggml-llama-embedding.wasm default

Prompt:
What's the capital of the United States?
Raw Embedding Output: {"n_embedding": 384, "embedding": [0.5426152349,-0.03840282559,-0.03644151986,0.3677068651,-0.115977712...(omitted)...,-0.003531290218]}
Interact with Embedding:
N_Embd: 384
Show the first 5 elements:
embd[0] = 0.5426152349
embd[1] = -0.03840282559
embd[2] = -0.03644151986
embd[3] = 0.3677068651
embd[4] = -0.115977712
```
