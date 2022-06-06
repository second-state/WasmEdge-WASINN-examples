# Mobilenet example for WASI-NN

This package is a high-level Rust bindings for [wasi-nn] example of Mobilenet.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.1.0"
```

## Build

Compile the application to WebAssembly:

```bash
cargo build --target=wasm32-wasi --release
```

The output WASM file will be at `target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet.wasm`.

## Run

First download the fixture files with the script:

```bash
./download_mobilenet.sh
```

And execute the WASM with the `wasmedge` with OpenVINO supporting:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-mobilenet.wasm mobilenet.xml mobilenet.bin tensor-1x224x224x3-f32.bgr
```

You will get the output:

```bash
Read graph XML, size in bytes: 143525
Read graph weights, size in bytes: 13956476
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [963](0.7113)pizza, pizza pie
   2.) [762](0.0707)restaurant, eating house, eating place, eatery
   3.) [909](0.0364)wok
   4.) [926](0.0155)hot pot, hotpot
   5.) [567](0.0153)frying pan, frypan, skillet
```
