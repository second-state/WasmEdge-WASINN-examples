# Command-R Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This example will focus on the Command-R prompt template.

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

## Get Model

Here we use the `c4ai-command-r-plus-GGUF` model as an example. You can download the model from the Hugging Face model hub.

```bash
curl -LO https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF/resolve/main/command-r-plus-Q5_K_M-00001-of-00002.gguf
curl -LO https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF/resolve/main/command-r-plus-Q5_K_M-00002-of-00002.gguf
```

## Execute

In this example, we use the system prompt with the definition of avaiable tools from [Example Rendered Tool Use Prompt](https://huggingface.co/CohereForAI/c4ai-command-r-plus).

````console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:command-r-plus-Q5_K_M-00001-of-00002.gguf \
  ./wasmedge-ggml-command-r.wasm default

USER:
Whats the biggest penguin in the world?
ASSISTANT:
Action: ```json
[
    {
        "tool_name": "internet_search",
        "parameters": {
            "query": "biggest penguin species"
        }
    }
]
```
````
