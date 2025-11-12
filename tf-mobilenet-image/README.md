# Mobilenet Example For WASI-NN with Tensorflow Backend

This package is a high-level Rust bindings for [wasi-nn] example of Mobilenet with Tensorflow backend.

[wasi-nn]: https://github.com/second-state/wasmedge-wasi-nn

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasmedge-wasi-nn = "0.2.1"
```

## Build

Compile the application to WebAssembly:

```bash
cd rust/tf-mobilenet && cargo build --target=wasm32-wasi --release
```

The output WASM file will be at [`rust/tf-mobilenet/target/wasm32-wasi/release/wasmedge-wasinn-example-tf-mobilenet-image.wasm`](wasmedge-wasinn-example-tf-mobilenet-image.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec rust/tf-mobilenet/target/wasm32-wasi/release/wasmedge-wasinn-example-tf-mobilenet-image.wasm run.wasm
```

## Run

### Download fixture

The testing image is located at `./PurpleGallinule.jpg` downloaded from [link](https://raw.githubusercontent.com/second-state/wasm-learning/master/rust/mobilenet_birds_tfhub/PurpleGallinule.jpg).

The `tf` model is located at `./saved_model.pb` downloaded from [link](https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1). Please note that, the model should be saved by tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format with no extra assets and variables.

### Generate Image Tensor

If you want to generate the [raw](birdx224x224x3.rgb) tensor, you can run:

```shell
cd rust/image-converter/ && cargo run ../../PurpleGallinule.jpg ../../birdX224X224X3.rgb && cd ../..
```

### Execute

Execute the WASM with the `wasmedge` with Tensorflow supporting:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-tf-mobilenet-image.wasm saved_model.pb PurpleGallinule.jpg
```

You will get the output:

```console
Read graph weights, size in bytes: 3561598
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 150528
Executed graph inference
   1.) [576](0.8951)Porphyrio martinicus
   2.) [427](0.0123)Cynanthus latirostris
   3.) [331](0.0064)Porphyrio poliocephalus
   4.) [798](0.0039)Porphyrio hochstetteri
   5.) [743](0.0033)Hemiphaga novaeseelandiae
```
