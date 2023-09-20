# Llama Example For WASI-NN with GGML Backend

## Dependencies

Install the latest wasmedge with plugins (not released yet)

## Build

Compile the application to WebAssembly:

```bash
cargo build --target wasm32-wasi --release
```

The output WASM file will be at `target/wasm32-wasi/release/`.
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedge compile target/wasm32-wasi/release/wasmedge-ggml-llama-interactive.wasm aot.wasm
```

## Get Model

Download llama model:

```bash
curl -LO https://huggingface.co/Rabinovich/Llama-2-7B-Chat-GGUF/resolve/main/Llama-2-7B-Chat-q4_0.gguf
```

### Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model:

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:Llama-2-7B-Chat-q4_0.gguf \
  wasmedge-ggml-llama-aot.wasm default
```

After executing the command, you may need to wait a moment for the input prompt to appear.
You can enter your question once you see the `Question:` prompt:

```console
Question:
What's the capital of the United States?
Answer:
The capital of the United States is Washington, D.C. (District of Columbia).
Question:
What about France?
Answer:
The capital of France is Paris.
Question:
I have two apples, each costing 5 dollars. What is the total cost of these apples?
Answer:
The total cost of the two apples is $10.
Question:
What if I have 3 apples?
Answer:
The total cost of 3 apples would be 15 dollars. Each apple costs 5 dollars, so 3 apples would cost 3 x 5 = 15 dollars.
```
