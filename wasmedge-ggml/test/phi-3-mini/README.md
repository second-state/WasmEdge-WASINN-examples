# `phi-3-mini`

Ensure that we can use the `phi-3-mini` model.

## Execute

```console
$ curl -LO https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:Phi-3-mini-4k-instruct-q4.gguf \
  wasmedge-ggml-phi-3-mini.wasm default \
  <prompt>
```
