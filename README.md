<div align="center">
  <h1><code>wasi-nn</code></h1>

  <p><strong>High-level bindings for writing wasi-nn applications</strong></p>

  <p>
    <a href="https://github.com/second-state/WasmEdge-WASINN-examples/actions?query=workflow%3ACI++">
      <img src="https://github.com/second-state/WasmEdge-WASINN-examples/actions/workflows/main.yaml/badge.svg" alt="CI status"/>
    </a>
  </p>
</div>


### Introduction

This project provides high-level wasi-nn bindings for Rust. The basic idea: write
your machine learning application in a high-level language using these bindings, compile it to
WebAssembly, and run it in a WebAssembly runtime that supports the [wasi-nn] proposal, such as
[WasmEdge].

[WasmEdge]: https://wasmedge.org/
[wasi-nn]: https://github.com/WebAssembly/wasi-nn

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes
> in the [wasi-nn] specification.


### Use

 - In Rust, download the [crate from crates.io][crates.io] by adding `wasi-nn = "0.1"` as a Cargo
   dependency; more information in the [Rust README].

[crates.io]: https://crates.io/crates/wasi-nn
[Rust README]: rust/README.md

### Examples

This repository includes examples of using these bindings. See the [Rust example] to walk through an end-to-end image classification using an Mobilenet model.
Run them with:

 - `./build_mobilenet_base.sh <WASMEDGE RUNTIME PATH>` runs the [Rust example].

[Rust example]: rust/examples/classification-example

### Related Links

- [WASI]
- [wasi-nn]
- [WasmEdge]
- [wasi-nn-guest](https://github.com/radu-matei/wasi-nn-guest)

[WASI]: https://github.com/WebAssembly/WASI

### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
