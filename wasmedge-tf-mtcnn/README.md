# Mobilenet Example For WasmEdge-Tensorflow plug-in

This package is a high-level Rust bindings for [WasmEdge-TensorFlow plug-in](https://wasmedge.org/docs/develop/rust/tensorflow) example of Mobilenet.

## Dependencies

This crate depends on the `wasmedge_tensorflow_interface` in the `Cargo.toml`:

```toml
[dependencies]
wasmedge_tensorflow_interface = "0.3.0"
```

## Build

Compile the application to WebAssembly:

```bash
cd rust && cargo build --target=wasm32-wasip1 --release
```

The output WASM file will be at [`rust/target/wasm32-wasip1/release/wasmedge-tf-example-mtcnn.wasm`](wasmedge-tf-example-mtcnn.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedge compile rust/target/wasm32-wasip1/release/wasmedge-tf-example-mtcnn.wasm wasmedge-tf-example-mtcnn_aot.wasm
```

## Run

### Test data

The testing image is located at `./solvay.jpg`:

![solvay](solvay.jpg)

The frozen `pb` model is located at `./mtcnn.pb`.

### Execute

Users should [install the WasmEdge with WasmEdge-TensorFlow](https://wasmedge.org/docs/start/install#wasmedge-tensorflow-plug-in).

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasmedge_tensorflow
```

Execute the WASM with the `wasmedge` with Tensorflow Lite supporting:

```bash
wasmedge --dir .:. wasmedge-tf-example-mtcnn.wasm mtcnn.pb solvay.jpg out.jpg
```

You will get the output:

```console
Drawing box: 30 results ...
```

Please check the output `out.jpg` for the result.
