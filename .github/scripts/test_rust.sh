rustup target add wasm32-wasi
rustup target add wasm32-unknown-unknown

source /opt/intel/openvino_2021/bin/setupvars.sh
ldconfig

./build_mobilenet_base.sh ./WasmEdge/build/tools/wasmedge/wasmedge
