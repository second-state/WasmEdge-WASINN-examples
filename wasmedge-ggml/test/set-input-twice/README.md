# `set-input-twice`

Ensure that we get the same result from executing `set_input` twice.

## Execute

```console
$ curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-set-input-twice.wasm default '<PROMPT>'
```
