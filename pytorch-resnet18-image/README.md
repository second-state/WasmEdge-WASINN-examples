# Resnet18 Example For WASI-NN with PyTorch Backend

This package is a high-level Rust bindings for [wasi-nn] example of Resnet18 with PyTorch backend.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.6.0"
```

## Build

Compile the application to WebAssembly:

```bash
cargo build --target=wasm32-wasi --release
```

Because here, we will demonstrate two ways of using wasi-nn. So the output WASM files will be at [`target/wasm32-wasi/release/wasmedge-wasinn-example-resnet18-image.wasm`](wasmedge-wasinn-example-resnet18-image.wasm) and [`target/wasm32-wasi/release/wasmedge-wasinn-example-resnet18-image-named-model.wasm`](wasmedge-wasinn-example-resnet18-image-named-model.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-resnet18-image.wasm wasmedge-wasinn-example-resnet18-image-aot.wasm

wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-resnet18-image-named-model.wasm wasmedge-wasinn-example-resnet18-image-named-model-aot.wasm
```

## Run

### Generate Model

First generate the fixture of the pre-trained mobilenet with the script:

```bash
pip3 install torch==2.4.1 numpy pillow --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
# generate the model fixture
python3 gen_resnet18_model.py
```

(Or you can use the pre-generated one at [`resnet18.pt`](resnet18.pt))

### Test Image

The testing image `input.jpg` is downloaded from <https://github.com/bytecodealliance/wasi-nn/raw/main/rust/examples/images/1.jpg> with license Apache-2.0

### Generate Tensor

If you want to generate the [raw tensor](image-1x3x224x224.rgb), you can run:

```bash
python3 gen_tensor.py input.jpg image-1x3x224x224.rgb
```

### Execute

Users should [install the WasmEdge with WASI-NN PyTorch backend plug-in](https://wasmedge.org/docs/start/install#wasi-nn-plug-in-with-pytorch-backend).

Execute the WASM with the `wasmedge` with PyTorch supporting:

- Case 1:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-resnet18-image.wasm resnet18.pt input.jpg
```

You will get the output:

```console
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](18.0458)banana
   2.) [940](15.6954)spaghetti squash
   3.) [951](14.1337)lemon
   4.) [942](13.2925)butternut squash
   5.) [941](10.6792)acorn squash
```

- Case 2: Apply named model feature
> requirement wasi-nn >= 0.5.0 and WasmEdge-plugin-wasi_nn-(*) >= 0.13.4 and  
> --nn-preload argument format follow <name>:<encoding>:<target>:<model_path>

```bash
wasmedge --dir .:. --nn-preload demo:PyTorch:CPU:resnet18.pt wasmedge-wasinn-example-resnet18-image-named-model.wasm demo input.jpg
```

You will get the same output:

```console
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](18.0458)banana
   2.) [940](15.6954)spaghetti squash
   3.) [951](14.1337)lemon
   4.) [942](13.2925)butternut squash
   5.) [941](10.6792)acorn squash
```

## Run from AOTInductor

### Generate Model
PyTorch backend also support load from the AOTInductor (Shared Library). To compile the pytorch model, please follow the Pytorch official tutorial.

* https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html


Or you can use the pre-generated one at [`resnet18_pt2.so`](resnet18_pt2.so). However it may not suitable for your machine. it is suggested to use [`gen_resnet18_aoti`](gen_resnet18_aoti) recompile the model.

> Notice: The AOTInductor from pip will use old c++ abi interface, it is maybe incompatible with wasmedge release, you may need to install the libtorch **without c++11 abi** and rebuild the wasmedge with `-DWASMEDGE_USE_CXX11_ABI=OFF`.


```bash
## Build Wasmedge with cmake example
cmake -Bbuild -GNinja -DWASMEDGE_USE_CXX11_ABI=OFF -DWASMEDGE_PLUGIN_WASI_NN_BACKEND=PyTorch .
```

### Execute

To run the AOT Inductor, you need use `--nn-preload` with `PyTorchAOTI` interface and specify absolute path to load the shared library.

```bash
export LD_LIBRARY_PATH=/path_to_libtorch/lib
./wasmedge --dir .:. --nn-preload demo:PyTorchAOTI:CPU:/absolute_path_model/resnet18_pt2.so wasmedge-wasinn-example-resnet18-image-named-model.wasm demo input.jpg
```

```console
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](18.0458)banana
   2.) [940](15.6954)spaghetti squash
   3.) [951](14.1337)lemon
   4.) [942](13.2925)butternut squash
   5.) [941](10.6792)acorn squash
```
