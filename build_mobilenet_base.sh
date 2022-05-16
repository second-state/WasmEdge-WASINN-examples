#!/bin/bash

set -e

if [ -z $1 ]; then
    echo "Please specify wasmedge path"
else
    WASMEDGE=$1
    WASI_NN_DIR=$(dirname "$0" | xargs dirname)
    WASI_NN_DIR=$(realpath $WASI_NN_DIR)
    WASMEDGE=$(realpath $WASMEDGE)
    source /opt/intel/openvino_2021/bin/setupvars.sh

    FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet
    pushd $WASI_NN_DIR/rust/
    cargo build --release --target=wasm32-wasi
    mkdir -p $WASI_NN_DIR/rust/examples/mobilenet-base/build
    RUST_BUILD_DIR=$(realpath $WASI_NN_DIR/rust/examples/mobilenet-base/build/)
    pushd examples/mobilenet-base
    cargo build --release --target=wasm32-wasi
    cp target/wasm32-wasi/release/mobilenet-base-example.wasm $RUST_BUILD_DIR
    pushd build

    if [ ! -f $RUST_BUILD_DIR/mobilenet.bin ]; then
        wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.bin
    fi
    if [ ! -f $RUST_BUILD_DIR/mobilenet.xml ]; then
        wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.xml
    fi
    if [ ! -f $RUST_BUILD_DIR/tensor-1x224x224x3-f32.bgr ]; then
        wget --no-clobber $FIXTURE/tensor-1x224x224x3-f32.bgr --output-document=$RUST_BUILD_DIR/tensor-1x224x224x3-f32.bgr
    fi
    # Manually run .wasm
    echo "Running example with WasmEdge ${WASMEDGE}"
    $WASMEDGE --dir fixture:$RUST_BUILD_DIR --dir .:. mobilenet-base-example.wasm "fixture/mobilenet.xml" "fixture/mobilenet.bin" "fixture/tensor-1x224x224x3-f32.bgr"
fi
