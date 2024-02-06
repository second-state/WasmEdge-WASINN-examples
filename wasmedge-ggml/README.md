# Llama Example For WASI-NN with GGML Backend

[See it in action!](https://x.com/juntao/status/1705588244602114303)

## Requirement

### Install WasmEdge + WASI-NN ggml plugin

WASI-NN ggml plugin only supports Linux and macOS.

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.

# Assuming you use zsh (the default shell on macOS), you will need to run the following command
source $HOME/.zshenv
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

The supported matrix:

| OS    | Arch          | GPU        | Framework | Comments                                                                  |
| -     | -             | -          | -         | -                                                                         |
| Linux | amd64         | NVIDIA GPU | cuda 12.x | When CUDA is detected, we will install the cuda enabled plugin by default |
| Linux | amd64         | -          | -         | -                                                                         |
| Linux | arm64         | NVIDIA GPU | cuda 11.x | This is built on Jetson AGX Orin                                          |
| Linux | arm64         | -          | -         | -                                                                         |
| macOS | Intel         | -          | -         | Metal is not working very well on AMD GPU                                 |
| macOS | Apple Silicon | Mx         | Metal     | -                                                                         |

We are planing to support in the near future:

| OS    | Arch  | GPU     | Framework | Comments         |
| -     | -     | -       | -         | -                |
| Linux | amd64 | AMD GPU | OpenCL    | Work in progress |

This version is verified on the following platforms:
1. Nvidia Jetson AGX Orin 64GB developer kit
2. Intel i7-10700 + Nvidia GTX 1080 8G GPU
2. AWS EC2 `g5.xlarge` + Nvidia A10G 24G GPU + Amazon deep learning base Ubuntu 20.04

## Prepare WASM application

### (Recommend) Use the pre-built one bundled in this repo

We built a wasm of this example under the folder, check these files:
- `chatml/wasmedge-ggml-chatml.wasm`
- `embedding/wasmedge-ggml-llama-embedding.wasm`
- `llama/wasmedge-ggml-llama.wasm`
- `llama-stream/wasmedge-ggml-llama-stream.wasm`

### (Optional) Build from source

If you want to do some modifications, you can build from source.

Compile the application to WebAssembly:

```bash
cd llama
cargo build --target wasm32-wasi --release
```

The output WASM file will be at `target/wasm32-wasi/release/`.

```bash
cp target/wasm32-wasi/release/wasmedge-ggml-llama.wasm ./wasmedge-ggml-llama.wasm
```

## Get Model

In this example, we are going to use llama2 7b chat model in GGUF format.

Download llama model:

```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama.wasm default
```

### GPU acceleration

#### macOS

macOS will use the Metal framework by default. llama.cpp supports `n-gpu-layers` now, please make sure you set the `n-gpu-layers` to offload the tensor layers into GPU.

Modify the following code to ensure the tensor layers of the model is offloaded into GPU:

```rust
// llama2-7b-chat provides 35 GPU layers. So, we have to set a value that is large or equal to 35.
options.insert("n-gpu-layers", Value::from(35));
```

#### Linux + CUDA

Due to the various GPU hardware, it's hard to set a default value of `n-gpu-layers`.

Modify the following code to ensure the tensor layers of the model is offloaded into GPU:

```rust
// llama2-7b-chat provides 35 GPU layers. So, we have to set a value that is large or equal to 35.
options.insert("n-gpu-layers", Value::from(35));
```

Remember to re-generate the wasm file after changing the code.
After executing, you may need to wait a moment for the input prompt to appear.
You can enter your question once you see the `Question:` prompt:

```console
Question:
What's the capital of the United States?
Answer:
The capital of the United States is Washington, D.C. (District of Columbia).
Question:
What about France?
Answer:
The capital of France is Paris.
Question:
I have two apples, each costing 5 dollars. What is the total cost of these apples?
Answer:
The total cost of the two apples is $10.
Question:
What if I have 3 apples?
Answer:
The total cost of 3 apples would be 15 dollars. Each apple costs 5 dollars, so 3 apples would cost 3 x 5 = 15 dollars.
```

## Parameters

Supported parameters include:

- `enable-log`: Set it to true to enable logging.
- `enable-debug-log`: Set it to true to enable debug log.
- `stream-stdout`: Set it to true to print the inferred tokens to standard output.
- `ctx-size`: Set the context size, the same as the `--ctx-size` parameter in llama.cpp.
- `n-predict`: Set the number of tokens to predict, the same as the `--n-predict` parameter in llama.cpp.
- `n-gpu-layers`: Set the number of layers to store in VRAM, the same as the `--n-gpu-layers` parameter in llama.cpp.
- `reverse-prompt`: Set it to the token at which you want to halt the generation. Similar to the `--reverse-prompt` parameter in llama.cpp.
- `batch-size`: Set the batch size number for prompt processing, the same as the `--batch-size` parameter in llama.cpp.
- `temp`: Set the temperature for the generation, the same as the `--temp` parameter in llama.cpp.
- `repeat-penalty`: Set the repeat penalty for the generation, the same as the `--repeat-penalty` parameter in llama.cpp.
- `threads`: Set the number of threads for the inference, the same as the `--threads` parameter in llama.cpp.

(For more detailed instructions on usage or default values for the parameters, please refer to [WasmEdge](https://github.com/WasmEdge/WasmEdge/blob/master/plugins/wasi_nn/ggml.cpp).)

## Performance

Welcome to submit PR to upload the TPS <3

| WasmEdge | Hardware                        | OS                                     | Model                    | TPS    |
| -        | -                               | -                                      | -                        | -      |
| 0.13.5   | i7-10700 + NVIDIA GTX 1080 8G   | Ubuntu 22.04                           | llama-2-7b-chat.Q5\_K\_M |  31.73 |
| 0.13.5   | M2 Max 64GB                     | macOS 13.6                             | llama-2-7b-chat.Q5\_K\_M |  42.08 |
| 0.13.5   | M3 Max 64GB                     | macOS 14.1.1                           | llama-2-7b-chat.Q5\_K\_M |  49.39 |
| 0.13.5   | AWS g5.xlarge A10G 24G          | Amazon Deep Learning base Ubuntu 20.04 | llama-2-7b-chat.Q5\_K\_M |  71.48 |
| 0.13.5   | i7-13700K + NVIDIA RTX 4090 24G | Windows 11 WSL2 Ubuntu 22.04           | llama-2-7b-chat.Q5\_K\_M | 111.90 |

## Errors

- After running `apt update && apt install -y libopenblas-dev`, you may encounter the following error:

  ```bash
  ...
  E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
  E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
  ```

   This indicates that you are not logged in as `root`. Please try installing again using the `sudo` command:

  ```bash
  sudo apt update && sudo apt install -y libopenblas-dev
  ```

- After running the `wasmedge` command, you may receive the following error:

  ```bash
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  unknown option: nn-preload
  ```

  This suggests that your plugin installation was not successful. To resolve this issue, please attempt to install your desired plugin again.

## Details

### Metadata

Currently, the WASI-NN ggml plugin supports several ways to set the metadata for the inference.

1. From the graph builder

When constructing the graph, you can set the metadata by using the `config` method.

```rust
... wasi_nn::GraphBuilder::new(...).config(serde_json::to_string(&options).unwrap()) ...
```

2. From the input tensor

When setting input to the context, specify the index with 1 for the metadata.
This setting will overwrite the metadata set in the graph builder.
If you modify the `n-gpu-layers` parameter, the model will be reloaded.

```rust
context
    .set_input(
        1,
        wasi_nn::TensorType::U8,
        &[1],
        serde_json::to_string(&options).expect("Failed to serialize options").as_bytes().to_vec(),
    )
    .unwrap();
```

(For more detailed instructions on usage or default values for the parameters, please refer to [WasmEdge](https://github.com/WasmEdge/WasmEdge/blob/master/plugins/wasi_nn/ggml.cpp).)

### Embedding Usage

Once the `embedding` mode is on, the return output will be a JSON object in the following format:

```json
{
  "n_embedding": 4096,
  "embedding": [...]
}
```

### Token Usage

You can use `get_output()` with index 1 to get the token usage of input and output text.
The token usage is a JSON string with the following format:

```json
{
  "input_tokens": 78,
  "output_tokens": 31
}
```

Users should be aware of the context size as well as the number of tokens used to avoid exceeding the limit.
If the number of tokens exceeds the context size, the WASI-NN ggml plugin will return a RuntimeError.

## Credit

The WASI-NN ggml plugin embedded [`llama.cpp`](git://github.com/ggerganov/llama.cpp.git) as its backend.
For details about the version used, please refer to [WasmEdge/LICENSE.spdx](https://github.com/WasmEdge/WasmEdge/blob/master/LICENSE.spdx).
