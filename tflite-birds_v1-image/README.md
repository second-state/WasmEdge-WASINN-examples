# Mobilenet Example For WASI-NN with Tensorflow Lite Backend

This package is a high-level Rust bindings for [wasi-nn] example of Mobilenet with Tensorflow Lite backend.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.1.0"
```

> Note: After the `TENSORFLOWLITE` encoding added into the wasi-nn crate, we'll update this example to use the newer version.

## Build

Compile the application to WebAssembly:

```bash
cd rust/tflite-bird && cargo build --target=wasm32-wasi --release
```

The output WASM file will be at [`rust/tflite-bird/target/wasm32-wasi/release/wasmedge-wasinn-example-tflite-bird-image.wasm`](wasmedge-wasinn-example-tflite-bird-image.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec rust/tflite-bird/target/wasm32-wasi/release/wasmedge-wasinn-example-tflite-bird-image.wasm wasmedge-wasinn-example-tflite-bird-image.wasm
```

## Run

### Download fixture

Use the below script to download the testing image `input.jpg` and the `tflite` model:

```bash
./download_data.sh   
```

The testing image is located at `./bird.jpg`:

![banana](https://raw.githubusercontent.com/second-state/wasm-learning/master/rust/birds_v1/bird.jpg)

The `tflite` model is located at `./lite-model_aiy_vision_classifier_birds_V1_3.tflite`

### Generate Image Tensor

If you want to generate the [raw](birdx224x224x3.rgb) tensor, you can run:

```shell
cd rust/image-converter/ && cargo run ../../bird.jpg ../../birdx224x224x3.rgb
```

### Execute

Execute the WASM with the `wasmedge` with Tensorflow Lite supporting:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-tflite-bird-image.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
```

You will get the output:

```console
Read graph weights, size in bytes: 3561598
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 150528
Executed graph inference
   1.) [166](198)Aix galericulata
   2.) [158](2)Coccothraustes coccothraustes
   3.) [34](1)Gallus gallus domesticus
   4.) [778](1)Sitta europaea
   5.) [819](1)Anas platyrhynchos
```
