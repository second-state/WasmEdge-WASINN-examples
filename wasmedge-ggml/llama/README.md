# `llama`

## Execute - llama 3

### Model Download Link

```console
wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf
```

### Execution Command

Please make sure you have the `Meta-Llama-3-8B-Instruct.Q5_K_M.gguf` file in the current directory.
Don't forget to set the `llama3` environment variable to `true` to enable the llama3 prompt template.
If you want to enable GPU support, please set the `n_gpu_layers` environment variable.
You can also change the `ctx_size` to have a larger context window via `--env ctx_size=8192`. The default value is 1024.

```console
$ wasmedge --dir .:. \
  --env llama3=true \
  --env n_gpu_layers=100 \
  --nn-preload default:GGML:AUTO:Meta-Llama-3-8B-Instruct.Q5_K_M.gguf \
  wasmedge-ggml-llama.wasm default

USER:
What's WasmEdge?
ASSISTANT:
WasmEdge is an open-source WebAssembly runtime and compiler that can run WebAssembly code in various environments, including web browsers, mobile devices, and server-side applications.
USER:
Does it support in Docker?
ASSISTANT:
Yes, WasmEdge supports running in Docker containers.
USER:
Does it support in Podman?
ASSISTANT:
Yes, WasmEdge also supports running in Podman containers.
USER:
Does it work with crun?
ASSISTANT:
 Yes, WasmEdge supports running in crun containers.
```

## Execute - llama 2

```console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama.wasm default

USER:
What's the capital of U.S.?
ASSISTANT:
The capital of the United States is Washington, D.C. (District of Columbia).
USER:
How about France?
ASSISTANT:
The capital of France is Paris.
```
