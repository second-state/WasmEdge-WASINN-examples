# Command-R Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This example will focus on the Command-R prompt template.

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

## Get Model

```bash
curl -LO https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/resolve/main/c4ai-command-r-v01-Q5_K_M.gguf
```

## Execute

```console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:c4ai-command-r-v01-Q5_K_M.gguf \
  ./wasmedge-ggml-command-r.wasm default

USER:
What's the capital of the United States?
ASSISTANT:
The capital of the United States is Washington, D.C.
USER:
How about Japan?
ASSISTANT:
Tokyo is the capital of Japan.
```