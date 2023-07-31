# Mobilenet example with WasmEdge WASI-NN OpenVINO plugin

This example demonstrates how to use WasmEdge WASI-NN OpenVINO plugin to perform an inference task with Mobilenet model.

## Set up the environment

- Install `rustup` and `Rust`

  Go to the [official Rust webpage](https://www.rust-lang.org/tools/install) and follow the instructions to install `rustup` and `Rust`.

  > It is recommended to use Rust 1.68 or above in the stable channel.

  Then, add `wasm32-wasi` target to the Rustup toolchain:

  ```bash
  rustup target add wasm32-wasi
  ```

- Clone the example repo

  ```bash
  git clone https://github.com/second-state/WasmEdge-WASINN-examples.git
  ```

- Install OpenVINO

Please refer to [WasmEdge Docs](https://wasmedge.org/docs/contribute/source/plugin/wasi_nn) and [OpenVINO™](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_apt.html)(2023) for the installation process.

  ```bash
  bash WasmEdge-WASINN-examples/scripts/install_openvino.sh
  ldconfig
  ```

- Install WasmEdge with Wasi-NN OpenVINO plugin

  ```bash
  export CMAKE_BUILD_TYPE=Release
  export VERSION=0.13.2

  curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v $VERSION -p /usr/local --plugins wasi_nn-openvino
  ldconfig
  ```

## Build and run `openvino-mobilenet-raw` example

- Download `MobileNet` model file

  ```bash
  cd openvino-mobilenet-raw
  bash download_mobilenet.sh
  ```

- Build and run the example

  ```bash
  cd rust
  cargo build --target wasm32-wasi --release
  cd ..

  wasmedge --dir .:. ./rust/target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm mobilenet.xml mobilenet.bin input.jpg
  ```

  If the commands above run successfully, you will get the output:

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
