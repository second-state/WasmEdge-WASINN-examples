<div align="center">
  <h1>WasmEdge WASI-NN Examples</h1>

  <p><strong>High-level bindings for writing wasi-nn applications</strong></p>

  <p>
    <a href="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/llama.yml/">
      <img src="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/llama.yml/badge.svg" alt="CI status - llama"/>
    </a>
    <a href="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/pytorch.yml/">
      <img src="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/pytorch.yml/badge.svg" alt="CI status - pytorch"/>
    </a>
    <a href="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/tflite.yml/">
      <img src="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/tflite.yml/badge.svg" alt="CI status - tflite"/>
    </a>
  </p>
</div>

### Introduction

This project provides the examples of high-level [wasi-nn] bindings and WasmEdge-TensorFlow plug-ins on Rust programming language. Developers can refer to this project to write their machine learning application in a high-level language using the bindings, compile it to WebAssembly, and run it with a WebAssembly runtime that supports the [wasi-nn] proposal, such as [WasmEdge].

### Prerequisites

#### OpenVINO Installation

Developers should install the [OpenVINO] first before build and run WasmEdge with wasi-nn and the examples.
For this project, we use the version `2023.0.0`. Please refer to [WasmEdge Docs](https://wasmedge.org/docs/contribute/source/plugin/wasi_nn) and [OpenVINO™](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_apt.html)(2023)

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

For building the WASM files from rust source, please refer to the [Rust Official Site](https://www.rust-lang.org/tools/install) for the Rust installation.
After the installation, developers should add the `wasm32-wasip1` target.

```bash
rustup target add wasm32-wasip1
```

#### Download the `wasi-nn` Rust Crate

In Rust, download the [crate from crates.io](https://crates.io/crates/wasi-nn) by adding `wasi-nn = "0.4.0"` as a Cargo dependency.

For using WasmEdge-TensorFlow plug-ins, please download the [crate from crates.io](https://crates.io/crates/wasmedge_tensorflow_interface) by adding `wasmedge_tensorflow_interface = "0.3.0"` as a Cargo dependency.

#### WasmEdge Installation

You can refer to [here to install WasmEdge](https://wasmedge.org/docs/start/install#install).

For the examples with different wasi-nn backends or using the WasmEdge-Tensorflow plug-ins, please install with plug-ins and their dependencies:

- [wasi-nn plug-in with OpenVINO backend](https://wasmedge.org/docs/start/install#wasi-nn-plug-in-with-openvino-backend)
- [wasi-nn plug-in with PyTorch backend](https://wasmedge.org/docs/start/install#wasi-nn-plug-in-with-pytorch-backend)
- [wasi-nn plug-in with TensorFlow-Lite backend](https://wasmedge.org/docs/start/install#wasi-nn-plug-in-with-tensorflow-lite-backend)
- [WasmEdge-Image plug-in](https://wasmedge.org/docs/start/install#wasmedge-image-plug-in)
- [WasmEdge-TensorFlow plug-in](https://wasmedge.org/docs/start/install#wasmedge-tensorflow-plug-in)
- [WasmEdge-TensorFlow-Lite plug-in](https://wasmedge.org/docs/start/install#wasmedge-tensorflow-lite-plug-in)

### Examples

[Mobilenet](mobilenet)

### Related Links

- [WASI]
- [wasi-nn]
- [WasmEdge]
- [WasmEdge-TensorFlow rust interface](https://crates.io/crates/wasmedge_tensorflow_interface)
- [wasi-nn-guest](https://github.com/radu-matei/wasi-nn-guest)

[WasmEdge]: https://wasmedge.org/
[wasi-nn]: https://github.com/WebAssembly/wasi-nn
[WASI]: https://github.com/WebAssembly/WASI

### License

This project is licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for more details.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.
