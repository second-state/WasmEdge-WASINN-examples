# Mobilenet Example For WASI-NN with PyTorch Backend

This package is a high-level Rust bindings for [wasi-nn] example of Mobilenet with PyTorch backend.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.4.0"
```

## Build

Compile the application to WebAssembly:

```bash
cargo build --target=wasm32-wasi --release
```

The output WASM file will be at [`target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm`](wasmedge-wasinn-example-mobilenet-image.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm wasmedge-wasinn-example-mobilenet-image-aot.wasm
```

## Run

### Generate Model

First generate the fixture of the pre-trained mobilenet with the script:

```bash
pip3 install torch==1.8.2 torchvision==0.9.2 pillow --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
# generate the model fixture
python3 gen_mobilenet_model.py
```

(Or you can use the pre-generated one at [`mobilenet.pt`](mobilenet.pt))

### Test Image

The testing image `input.jpg` is downloaded from <https://github.com/bytecodealliance/wasi-nn/raw/main/rust/examples/images/1.jpg> with license Apache-2.0

### Generate Tensor

If you want to generate the [raw tensor](image-1x3x224x224.rgb), you can run:

```bash
python3 gen_tensor input.jpg image-1x3x224x224.rgb
```

### Execute

Users should [install the WasmEdge with WASI-NN PyTorch backend plug-in](https://wasmedge.org/book/en/write_wasm/rust/wasinn.html#get-wasmedge-with-wasi-nn-plug-in-pytorch-backend).

Execute the WASM with the `wasmedge` with PyTorch supporting:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-mobilenet-image.wasm mobilenet.pt input.jpg
```

You will get the output:

```console
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
