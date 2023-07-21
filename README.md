<div align="center">
  <h1>WasmEdge WASI-NN Examples</h1>

  <p><strong>High-level bindings for writing wasi-nn applications</strong></p>

  <p>
    <a href="https://github.com/second-state/WasmEdge-WASINN-examples/actions?query=workflow%3ACI++">
      <img src="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/build.yaml/badge.svg" alt="CI status"/>
    </a>
  </p>
</div>

### Introduction

This project provides the examples of high-level [wasi-nn] bindings on Rust programming language. Developers can refer to this project to write their machine learning application in a high-level language using the bindings, compile it to WebAssembly, and run it with a WebAssembly runtime that supports the [wasi-nn] proposal, such as [WasmEdge].

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes in the [wasi-nn] specification.

### Prerequisites

#### OpenVINO Installation

Developers should install the [OpenVINO] first before build and run WasmEdge with wasi-nn and the examples.
For this project, we use the version `2023.0.0`. Please refer to the [installation script](scripts/install_openvino.sh).

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list
sudo apt update
sudo apt-get -y install openvino
ldconfig
```
[OpenVINO]: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html

#### Rust Installation

Please refer to the [Rust Official Site](https://www.rust-lang.org/tools/install) for the Rust installation.
After the installation, developers should add the `wasm32-wasi` target.

```bash
rustup target add wasm32-wasi
```

#### Download the `wasi-nn` Rust Crate

In Rust, download the [crate from crates.io][crates.io] by adding `wasi-nn = "0.1"` as a Cargo dependency.

[crates.io]: https://crates.io/crates/wasi-nn

#### Build the WasmEdge with WASI-NN supporting

For running the examples, developers should [build WasmEdge from source](https://wasmedge.org/book/en/extend/build.html).
First developers should get the source:

```bash
git clone https://github.com/WasmEdge/WasmEdge.git
cd WasmEdge
```

And build with the `WASMEDGE_PLUGIN_WASI_NN_BACKEND` argument:

```bash
cmake -Bbuild -GNinja -WASMEDGE_PLUGIN_WASI_NN_BACKEND="OpenVINO" .
cmake --build build
# For the WASI-NN plugin, you should install this project.
cmake --install build
```

After the installation, developers can execute the `wasmedge` executable with WASI-NN plugin.

> Notice: If you didn't install the project, you should give the `WASMEDGE_PLUGIN_PATH` environment variable for specifying the WASI-NN plugin path (the built plugin is at `build/plugins/wasi_nn`).

### Examples

[Mobilenet](mobilenet)

### Related Links

- [WASI]
- [wasi-nn]
- [WasmEdge]
- [wasi-nn-guest](https://github.com/radu-matei/wasi-nn-guest)

[WasmEdge]: https://wasmedge.org/
[wasi-nn]: https://github.com/WebAssembly/wasi-nn
[WASI]: https://github.com/WebAssembly/WASI

### License

This project is licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for more details.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.
