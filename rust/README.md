# wasi-nn bindings for Rust

This package contains high-level Rust bindings for [wasi-nn] system calls. It is similar in purpose
to the [WASI bindings] but this package provides optional access to a system's machine learning
functionality from WebAssembly.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn
[WASI bindings]: https://github.com/bytecodealliance/wasi

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes
> in the wasi-nn specification.

### Use

1. Depend on this crate in your `Cargo.toml`:

    ```toml
    [dependencies]
    wasi-nn = "0.1.0"
    ```

2. Use the wasi-nn APIs in your application, for example:

    ```rust
    use wasi_nn;

    unsafe {
        wasi_nn::load(
            &[&xml.into_bytes(), &weights],
            wasi_nn::GRAPH_ENCODING_OPENVINO,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    }
    ```

3. Compile the application to WebAssembly:

    ```shell script
    cargo build --target=wasm32-wasi
    ```

4. Run the generated WebAssembly in a runtime supporting [wasi-nn], e.g. [WasmEdge].

[WasmEdge]: https://wasmedge.org/

### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: ../LICENSE

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
