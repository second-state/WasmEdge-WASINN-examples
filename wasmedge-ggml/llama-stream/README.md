# `llama-stream`

## Execute

```console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama-stream.wasm default

USER:
What's the capital of U.S.?
ASSISTANT:
The capital of the United States is Washington, D.C. (District of Columbia).
USER:
How about France?
ASSISTANT:
The capital of France is Paris.
```
