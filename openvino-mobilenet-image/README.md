# Mobilenet example for WASI-NN

This package is a high-level Rust bindings for [wasi-nn] example of Mobilenet.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.2.1"
```

## Build

Compile the application to WebAssembly:

```bash
cargo build --target=wasm32-wasi --release
```

The output WASM file will be at [`target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm`](wasmedge-wasinn-example-mobilenet-image.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm wasmedge-wasinn-example-mobilenet-image.wasm
```

## Run

First download the fixture files with the script:

```bash
./download_mobilenet.sh
```

it will also download a testing image `input.jpg`

![banana](https://github.com/bytecodealliance/wasi-nn/raw/main/rust/examples/images/1.jpg)

Users should [install the WasmEdge with WASI-NN OpenVINO backend plug-in](https://wasmedge.org/book/en/write_wasm/rust/wasinn.html#get-wasmedge-with-wasi-nn-plug-in-openvino-backend).

And execute the WASM with the `wasmedge` with OpenVINO supporting:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-mobilenet-image.wasm mobilenet.xml mobilenet.bin input.jpg
```

You will get the output:

```bash
Read graph XML, size in bytes: 143525
Read graph weights, size in bytes: 13956476
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](0.9789)banana
   2.) [940](0.0074)spaghetti squash
   3.) [951](0.0014)lemon
   4.) [969](0.0005)eggnog
   5.) [942](0.0005)butternut squash
```
