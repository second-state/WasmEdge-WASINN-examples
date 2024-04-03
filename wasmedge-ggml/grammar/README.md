# Grammar Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of using grammar in ggml.

## Get the Model

In this example, we are going to use the [llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) model. Please note that we are not using a fine-tuned chat model.

```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

In this example, we are going to use the `grammar` option to constrain the model to generate the JSON output in a specific format.

You can check [the documents at llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master/grammars) for more details about grammars.

## Execute

In this example, we are going to use the `n_predict` option to avoid the model from generating too many outputs.

```console
$ wasmedge --dir .:. \
  --env n_predict=99 \
  --nn-preload default:GGML:AUTO:llama-2-7b.Q5_K_M.gguf \
  wasmedge-ggml-grammar.wasm default

USER:
JSON object with 5 country names as keys and their capitals as values:
ASSISTANT:
{"US": "Washington", "UK": "London", "Germany": "Berlin", "France": "Paris", "Italy": "Rome"}
```
