# Text to speech example with WasmEdge WASI-NN Piper plugin

This example demonstrates how to use WasmEdge WASI-NN Piper plugin to perform TTS.

## Build WasmEdge with WASI-NN Piper plugin

Build WasmEdge from source:

```bash
cd /path/to/wasmedge/source/folder

cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND=Piper
cmake --build build
```

Then you will have an executable `wasmedge` runtime at `build/tools/wasmedge/wasmedge` and the WASI-NN with Piper backend plug-in at `build/plugins/wasi_nn/libwasmedgePluginWasiNN.so`.

## Model Download Link

In this example, we will use the [en_US-lessac-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) model.

[MODEL CARD](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD):

```
# Model card for lessac (medium)

* Language: en_US (English, United States)
* Speakers: 1
* Quality: medium
* Samplerate: 22,050Hz

## Dataset

* URL: https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/
* License: https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html

## Training

Trained from scratch.

```

It has a model file [en_US-lessac-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx) and a config file [en_US-lessac-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json).

```bash
# Download model
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
# Download config
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

This model uses [eSpeak NG](https://github.com/rhasspy/espeak-ng) to convert text to phonemes, so we also need to download the required espeak-ng-data.

This will download and extract the espeak-ng-data directory to the current working directory:

```bash
curl -LO https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz piper/espeak-ng-data --strip-components=1
```

## Build wasm

Run the following command to build wasm, the output WASM file will be at `target/wasm32-wasi/release/`

```bash
cargo build --target wasm32-wasi --release
```

## Execute

Execute the WASM with the `wasmedge`.

```bash
WASMEDGE_PLUGIN_PATH=/path/to/parent/directory/of/libwasmedgePluginWasiNN.so /path/to/wasmedge --dir .:. /path/to/wasm
```

Example layout:

```
.
├── en_US-lessac-medium.onnx
├── en_US-lessac-medium.onnx.json
├── espeak-ng-data/
├── WasmEdge/build/
│    ├── plugins/wasi_nn/libwasmedgePluginWasiNN.so
│    └── tools/wasmedge/wasmedge
└── WasmEdge-WASINN-examples/wasmedge-piper/target/wasm32-wasi/release/wasmedge-piper.wasm
```

Then the command will be:

```bash
WASMEDGE_PLUGIN_PATH=WasmEdge/build/plugins/wasi_nn WasmEdge/build/tools/wasmedge/wasmedge --dir .:. WasmEdge-WASINN-examples/wasmedge-piper/target/wasm32-wasi/release/wasmedge-piper.wasm
```

The output `welcome.wav` is the synthesized audio.
