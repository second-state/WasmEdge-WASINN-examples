# RPC Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of the WASI-NN RPC usage.

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

For GPU offloading, please adjust the `n-gpu-layers` options to the number of layers that you want to offload to the GPU.

```rust
options.insert("n-gpu-layers", Value::from(...));
```

In llava inference, we recommend to use the `ctx-size` at least `2048` when using llava-v1.5 and at least `4096` when using llava-v1.6 for better results.

```rust
options.insert("ctx-size", Value::from(4096));
```

## Execute


```console
# Run the RPC server.
$ wasi_nn_rpcserver --nn-rpc-uri unix://$PWD/nn_server.sock \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf

# Run the wasmedge and inference though the RPC server.
$ wasmedge \
  --nn-rpc-uri unix://$PWD/nn_server.sock \
  wasmedge-ggml-nnrpc.wasm default

USER:
What's the capital of the United States?
ASSISTANT:
The capital of the United States is Washington, D.C. (District of Columbia).
USER:
How about France?
ASSISTANT:
The capital of France is Paris.
```
