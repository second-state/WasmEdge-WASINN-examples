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

The output WASM file will be at [`rust/target/wasm32-wasip1/release/wasmedge-tf-example-mobilenet.wasm`](wasmedge-tf-example-mobilenet.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedge compile rust/target/wasm32-wasip1/release/wasmedge-tf-example-mobilenet.wasm wasmedge-tf-example-mobilenet_aot.wasm
```

## Run

### Test data

The testing image is located at `./grace_hopper.jpg`:

![grace_hopper](grace_hopper.jpg)

The frozen `pb` model is located at `./mobilenet_v2_1.4_224_frozen.pb`.

### Execute

Users should [install the WasmEdge with WasmEdge-TensorFlow and WasmEdge-Image plug-ins](https://wasmedge.org/docs/start/install#wasmedge-tensorflow-plug-in).

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasmedge_tensorflow wasmedge_image
```

Execute the WASM with the `wasmedge` with Tensorflow Lite supporting:

```bash
wasmedge --dir .:. wasmedge-tf-example-mobilenet.wasm mobilenet_v2_1.4_224_frozen.pb grace_hopper.jpg
```

You will get the output:

```console
653 : 0.3230184316635132
```

Which is index 653 (0-based index) with rate `0.33690106868743896`. The index 653 of label table (which is line 654 in `imagenet_slim_labels.txt`) is `military uniform`.
