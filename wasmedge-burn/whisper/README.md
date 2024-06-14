# Whisper Example For WASI-NN with Burn Backend

The current version of the plugin has not been officially released yet, so it needs to be compiled manually. And here we are using two different branches for running in non-container and container environments.

## Non container

### Build plugin

The plugin can ==only support one model type== at a time.

```bash
git clone https://github.com/CaptainVincent/WasmEdge.git
cd WasmEdge
git checkout wasi_nn_rust

// For whisper model
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_BUILD_AOT_RUNTIME=OFF -DWASMEDGE_PLUGIN_WASI_NN_RUST_MODEL=whisper
cmake --build build

// Replace the path to your WasmEdge
export PATH=<Your WasmEdge Path>/build/tools/wasmedge:$PATH
export WASMEDGE_PLUGIN_PATH=<Your WasmEdge Path>/build/plugins/wasi_nn_rust
```
([More about how to build Wasmedge](https://wasmedge.org/docs/contribute/source/os/linux/))

### Execute

```bash
// Make sure you build your plugin with the Whisper model enabled then
cd WasmEdge-WASINN-examples/wasmedge-burn/whisper

// Untar model suite
tar -xvzf model.tar.gz

// Verify with CPU
wasmedge --dir .:. --nn-preload="default:Burn:CPU:tiny_en.mpk:tiny_en.cfg:tokenizer.json:en" wasmedge-wasinn-example-whisper.wasm audio16k.wav default

// Verify with GPU
wasmedge --dir .:. --nn-preload="default:Burn:GPU:tiny_en.mpk:tiny_en.cfg:tokenizer.json:en" wasmedge-wasinn-example-whisper.wasm audio16k.wav default
```

## Inside container

We could use the wasmedge shim as docker wasm runtime to run the example.

### Build the docker image include the wasm application only

```bash
cd WasmEdge-WASINN-examples
git checkout docker

cd wasmedge-burn/whisper
docker build . --platform wasi/wasm -t whisper
```

### Install the experimental version of Docker Desktop that supports WebGPU

Our plugin will be packaged as part of the runtime in Docker Desktop.

(Coming soon)

### Execute

```bash
cd WasmEdge-WASINN-examples/wasmedge-burn/whisper
tar -xvzf model.tar.gz

// Verify with CPU
docker run \
  --runtime=io.containerd.wasmedge.v1 \
  --platform=wasi/wasm \
  -v $(pwd):/resource \
  --env WASMEDGE_WASINN_PRELOAD=default:Burn:CPU:/resource/tiny_en.mpk:/resource/tiny_en.cfg:/resource/tokenizer.json:en \
  whisper:latest /resource/audio16k.wav default

// Verify with GPU
docker run \
  --runtime=io.containerd.wasmedge.v1 \
  --platform=wasi/wasm \
  -v $(pwd):/resource \
  --env WASMEDGE_WASINN_PRELOAD=default:Burn:GPU:/resource/tiny_en.mpk:/resource/tiny_en.cfg:/resource/tokenizer.json:en \
  whisper:latest /resource/audio16k.wav default
```

or 

You can directly use our prebuilt docker image, which already includes the model weights, configuration file, and other necessary files, so there's no need for additional extraction model.tar.gz.

```bash
cd WasmEdge-WASINN-examples/wasmedge-burn/whisper

docker pull secondstate/burn-whisper:latest

// Verify with CPU
docker run \
  --runtime=io.containerd.wasmedge.v1 \
  --platform=wasi/wasm \
  -v $(pwd):/resource \
  --env WASMEDGE_WASINN_PRELOAD=default:Burn:CPU:/tiny_en.mpk:/tiny_en.cfg:/tokenizer.json:en \
  whisper:latest /resource/audio16k.wav default

// Verify with GPU
docker run \
  --runtime=io.containerd.wasmedge.v1 \
  --platform=wasi/wasm \
  -v $(pwd):/resource \
  --env WASMEDGE_WASINN_PRELOAD=default:Burn:GPU:/tiny_en.mpk:/tiny_en.cfg:/tokenizer.json:en \
  whisper:latest /resource/audio16k.wav default
```