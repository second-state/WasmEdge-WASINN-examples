# Llama Example For WasmEdge-Tensorflow plug-in

This package is a high-level Rust bindings for [WasmEdge-TensorFlow plug-in](https://wasmedge.org/docs/develop/rust/tensorflow) example of Mobilenet.

## Note

To build this example, you will need the nightly version of Rust.

## Dependencies

This crate depends on the `wasmedge_tensorflow_interface` in the `Cargo.toml`:

```toml
[dependencies]
wasmedge_tensorflow_interface = "0.3.0"
wasi-nn = "0.1.0"  # Ensure you use the latest version
thiserror = "1.0"
bytemuck = "1.13.1"
log = "0.4.19"
env_logger = "0.10.0"
anyhow = "1.0.79"
```

## Build

Compile the application to WebAssembly:

```bash
cd rust && cargo build --target=wasm32-wasip1 --release
```

The output WASM file will be at [`rust/target/wasm32-wasip1/release/wasmedge-tf-example-llama.wasm`](wasmedge-tf-example-llama.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedge compile rust/target/wasm32-wasip1/release/wasmedge-tf-example-llama.wasm wasmedge-tf-example-llama_aot.wasm
```

## Run

The frozen `tflite` model should be translated through `ai_edge_torch` and HuggingFace.

### Execute

Users should [install the WasmEdge with WasmEdge-TensorFlow and WasmEdge-Image plug-ins](https://wasmedge.org/docs/start/install#wasmedge-tensorflow-plug-in).

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasmedge_tensorflow wasmedge_image
```

Execute the WASM with the `wasmedge` with Tensorflow Lite supporting:

```bash
wasmedge --dir .:. wasmedge-tf-example-llama.wasm ./llama_1b_q8_ekv1280.tflite
```

You will get the output:

```console
Input the Chatbot:
Hello world
Hello world!
```
