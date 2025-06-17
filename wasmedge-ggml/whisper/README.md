# Basic Example For WASI-NN with Whisper Backend

This example is for a basic audio recognition with WASI-NN whisper backend in WasmEdge.
In current status, WasmEdge implement the Whisper backend of WASI-NN in only English. We'll extend more options in the future.

## Dependencies

This crate depends on the `wasmedge-wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasmedge-wasi-nn = "0.8.0"
```

## Build

Compile the application to WebAssembly:

```bash
cargo build --target=wasm32-wasip1 --release
```

The output WASM file will be at [`target/wasm32-wasip1/release/whisper-basic.wasm`](whisper-basic.wasm).
To speed up the processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedge compile target/wasm32-wasip1/release/whisper-basic.wasm whisper-basic_aot.wasm
```

## Run

### Test data

The testing audio is located at `./test.wav`.

Users should get the model by the guide from [whisper.cpp repository](https://github.com/ggerganov/whisper.cpp/tree/master/models):

```bash
curl -sSf https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/models/download-ggml-model.sh | bash -s -- base.en
```

The model will be stored at `./ggml-base.en.bin`.

### Input Audio

The WASI-NN whisper backend for WasmEdge currently supported 16kHz, 1 channel, and `pcm_s16le` format.

Users can convert their input audio as following `ffmpeg` command:

```bash
ffmpeg -i test.m4a -acodec pcm_s16le -ac 1 -ar 16000 test.wav
```

### Execute

> Note: This is prepared for `0.14.2` or later release in the future. Please build from source now.

Users should [install the WasmEdge with WASI-NN plug-in in Whisper backend](https://wasmedge.org/docs/start/install/#wasi-nn-plug-ins).

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-whisper
```

Execute the WASM with the `wasmedge` with WASI-NN plug-in:

```bash
wasmedge --dir .:. whisper-basic_aot.wasm ggml-base.en.bin test.wav
```

You will get recognized string from the audio file in the output:

```bash
Read model, size in bytes: 147964211
Loaded graph into wasi-nn with ID: Graph#0
Read input tensor, size in bytes: 141408
Recognized from audio:
[00:00:00.000 --> 00:00:04.300]  This is a test record for whisper.cpp
```
