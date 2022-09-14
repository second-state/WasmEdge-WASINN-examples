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

The output WASM file will be at `target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm`.
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm wasmedge-wasinn-example-mobilenet-image.wasm
```

## Run

First generate the fixture of the pre-trained mobilenet with the script:

```bash
./download_data.sh fixtures && cd fixtures
python -m pip install -r requirements.txt
# generate the model fixture
python generate_mobilenet.py
```

(or you can use the pre-generated fixture in `fixtures/mobilenet.pt`)

The above will download a testing image `input.jpg`
![](https://github.com/bytecodealliance/wasi-nn/raw/main/rust/images/1.jpg)
as well as a pre-trained mobilenet model, then convert the model into the torchscript model for C++.

And execute the WASM with the `wasmedge` with PyTorch supporting:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-mobilenet-image.wasm fixtures/mobilenet.pt input.jpg
```

You will get the output:

```console
Read torchscript binaries, size in bytes: 14376924
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](20.6681)banana
   2.) [940](12.1483)spaghetti squash
   3.) [951](11.5748)lemon
   4.) [950](10.4899)orange
   5.) [953](9.4834)pineapple, ananas
```
