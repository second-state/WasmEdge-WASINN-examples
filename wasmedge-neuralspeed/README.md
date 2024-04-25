# Neural chat example with WasmEdge WASI-NN Neural Speed plugin
This example demonstrates how to use WasmEdge WASI-NN Neural Speed plugin to perform an inference task with Neural chat model.

## Install WasmeEdge with WASI-NN Neural Speed plugin

The Neural Speed backend relies on Neural Speed, we recommend the following commands to install Neural Speed.

``` bash
sudo apt update
sudo apt upgrade
sudo apt install python3-dev
wget https://raw.githubusercontent.com/intel/neural-speed/main/requirements.txt
pip install -r requirements.txt
pip install neural-speed
```

Then build and install WasmEdge from source:

``` bash
cd <path/to/your/wasmedge/source/folder>

cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="neuralspeed"
cmake --build build

# For the WASI-NN plugin, you should install this project.
cmake --install build
```

Then you will have an executable `wasmedge` runtime under `/usr/local/bin` and the WASI-NN with OpenVINO backend plug-in under `/usr/local/lib/wasmedge/libwasmedgePluginWasiNN.so` after installation.
## Model Download Link

In this example, we will use neural-chat-7b-v3-1.Q4_0 model in GGUF format.

``` bash
# Download model weight
wget https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_0.gguf
# Download tokenizer
wget https://huggingface.co/Intel/neural-chat-7b-v3-1/raw/main/tokenizer.json -O neural-chat-tokenizer.json
```

## Build wasm

Run the following command to build wasm, the output WASM file will be at `target/wasm32-wasi/release/`

```bash
cargo build --target wasm32-wasi --release
```

## Execute 

Execute the WASM with the `wasmedge` using nn-preload to load model. 

```bash
wasmedge --dir .:. \
  --nn-preload default:NeuralSpeed:AUTO:neural-chat-7b-v3-1.Q4_0.gguf \
  ./target/wasm32-wasi/release/wasmedge-neural-speed.wasm default

```

## Other 

You can change tokenizer_path to your tokenizer path.

``` rust
let tokenizer_name = "neural-chat-tokenizer.json";        
```

Prompt is the default model input.

``` rust
let prompt = "Once upon a time, there existed a little girl,";
```
If your model type not llama, you can set model_type parameter to load different model.

``` rust
let graph = GraphBuilder::new(GraphEncoding::NeuralSpeed, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&json!({"model_type": "mistral"}))
```