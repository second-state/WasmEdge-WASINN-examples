# Llama Example For WASI-NN with GGML Backend

## Dependencies

Install the latest wasmedge with plugins:

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml
```

## Build

Compile the application to WebAssembly:

```bash
cargo build --target wasm32-wasi --release
```

The output WASM file will be at `target/wasm32-wasi/release/`.

## Get Model

Download llama model:

```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf
```

### Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model:

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b.Q5_K_M.gguf \
  target/wasm32-wasi/release/wasmedge-ggml-llama.wasm default 'Once upon a time, '
```

After executing the command, it takes some time to wait for the output.
Once the execution is complete, the following output will be generated:

```console
Loaded model into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 18
Executed model inference
Output: Once upon a time, 100 years ago, there was a small village nestled in the rolling hills of the countryside. Unterscheidung between the two is not always clear-cut, and both terms are often used interchangeably. The village was home to a small community of people who lived simple lives, relying on the land for their livelihood. The villagers were known for their kindness, generosity, and strong sense of community. They worked together to cultivate the land, grow their own food, and raise their children. The village was a peaceful place, where everyone knew and looked out for each other.

However, as time passed, the village began to change. New technologies and innovations emerged, and the villagers found themselves adapting to a rapidly changing world. Some embraced the changes, while others resisted them. The village became more connected to the outside world, and the villagers began to interact with people from other places. The village was no longer isolated, and the villagers were
```
