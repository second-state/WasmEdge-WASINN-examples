# JSON Schema Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of using json schema in ggml.

## Get the Model

In this example, we are going to use the [llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) model. Please note that we are not using a fine-tuned chat model.

```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

In this example, we are going to use the `json-schema` option to constrain the model to generate the JSON output in a specific format.

You can check [the documents at llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/main#grammars--json-schemas) for more details about this.

## Execute

```console
$ wasmedge --dir .:. \
  --env n_predict=99 \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-json-schema.wasm default

USER:
Give me a JSON array of Apple products.
ASSISTANT:
[
{
"productId": 1,
"productName": "iPhone 12 Pro",
"price": 799.99
},
{
"productId": 2,
"productName": "iPad Air",
"price": 599.99
},
{
"productId": 3,
"productName": "MacBook Air",
"price": 999.99
},
{
"productId": 4,
"productName": "Apple Watch Series 7",
"price": 399.99
},
{
"productId": 5,
"productName": "AirPods Pro",
"price": 249.99
}
]
```
