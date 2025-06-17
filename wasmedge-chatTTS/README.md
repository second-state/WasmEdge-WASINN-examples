# ChatTTS example with WasmEdge WASI-NN ChatTTS plugin
This example demonstrates how to use the WasmEdge WASI-NN ChatTTS plugin to generate speech from text. ChatTTS is a text-to-speech model designed specifically for dialogue scenarios such as LLM assistant. This example will use the WasmEdge WASI-NN ChatTTS plugin to run the ChatTTS to generate speech.

## Install WasmEdge with WASI-NN ChatTTS plugin
The ChatTTS backend relies on ChatTTS and Python library, we recommend the following commands to install the dependencies.
``` bash
sudo apt update
sudo apt upgrade
sudo apt install python3-dev
pip install chattts==0.1.1
```

Then build and install WasmEdge from source.

``` bash
cd <path/to/your/wasmedge/source/folder>

cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="chatTTS"
cmake --build build

# For the WASI-NN plugin, you should install this project.
cmake --install build
```

Then you will have an executable `wasmedge` runtime under `/usr/local/bin` and the WASI-NN with ChatTTS backend plug-in under `/usr/local/lib/wasmedge/libwasmedgePluginWasiNN.so` after installation.

## Build wasm

Run the following command to build wasm, the output WASM file will be at `target/wasm32-wasip1/release/`

```bash
cargo build --target wasm32-wasip1 --release
```

## Execute

Execute the WASM with the `wasmedge`.

``` bash
wasmedge --dir .:.  ./target/wasm32-wasip1/release/wasmedge-chattts.wasm
```

Then you will generate the `output1.wav` file. It is the wav file of the input text.

## Advanced Options

The `config_data` is used to adjust the configuration of the ChatTTS.
Supports the following options:
- `prompt`: Generate the special token in the text to synthesize.
- `spk_emb`: Sampled speaker (Using `random` for random speaker).
- `temperature`: Custom temperature.
- `top_k`: Top P decode.
- `top_p`: Top K decode.

``` rust
let config_data = serde_json::to_string(&json!({"prompt": "[oral_2][laugh_0][break_6]", "spk_emb": "random", "temperature": 0.5, "top_k": 0, "top_p": 0.9}))
        .unwrap()
        .as_bytes()
        .to_vec();
```
<table>
<tr>
<td>

[demo.webm](https://github.com/user-attachments/assets/377e0487-9107-41db-9c22-31962ce53f88)

</td>
</tr>
</table>
