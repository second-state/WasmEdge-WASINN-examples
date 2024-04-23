# `model-not-found`

Ensure that we get the `ModelNotFound` error when the model does not exist.

## Execute

```console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:model-not-found.gguf \
  wasmedge-ggml-model-not-found.wasm default
```
