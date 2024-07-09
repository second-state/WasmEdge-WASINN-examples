cargo build --target wasm32-wasi --release

wasmedge --dir .:.  ./target/wasm32-wasi/release/wasmedge-chattts.wasm