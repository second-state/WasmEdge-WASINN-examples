# Llama Embedding Example For WASI-NN with GGML Backend

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

We built a wasm of this example under the folder, check `wasmedge-ggml-llama-embedding.wasm`

### (Optional) Build from source

If you want to do some modifications, you can build from source.

Compile the application to WebAssembly:

```bash
cargo build --target wasm32-wasi --release
```

The output WASM file will be at `target/wasm32-wasi/release/`.

```bash
cp target/wasm32-wasi/release/wasmedge-ggml-llama-embedding.wasm ./wasmedge-ggml-llama-embedding.wasm
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
  wasmedge-ggml-llama-embedding.wasm default
```

After executing the command, you may need to wait a moment for the input prompt to appear.
You can enter your question once you see the prompt:

```console
Prompt:
What's the capital of the United States?
Raw Embedding Output: {"n_embedding": 4096, "embedding": [2.129590273,-0.7558271289,1.354019523,1.840911746,-0.8185964823,...omitted...,-0.02710761502]}
Interact with Embedding:
N_Embd: 4096
Show the first 5 elements:
embd[0] = 2.129590273
embd[1] = -0.7558271289
embd[2] = 1.354019523
embd[3] = 1.840911746
embd[4] = -0.8185964823
```

## Parameters

Supported parameters include:

- `enable-log`: Set it to true to enable logging.
- `ctx-size`: Set the context size, the same as the `--ctx-size` parameter in llama.cpp.
- `batch-size`: Set the batch size number for prompt processing, the same as the `--batch-size` parameter in llama.cpp.
- `threads`: Set the number of threads for the inference, the same as the `--threads` parameter in llama.cpp.

(For more detailed instructions on usage or default values for the parameters, please refer to [WasmEdge](https://github.com/WasmEdge/WasmEdge/blob/master/plugins/wasi_nn/ggml.cpp).)

For convenience, these parameters could be set by adding the environmental variables in this example.
The environmental variables are handled by Rust. (Due to the limitation of environmental variables, beware of the `-` and `_` in the variable name.)

```bash
wasmedge --dir .:. \
  --env enable_log=false \
  --env ctx_size=512 \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama-embedding.wasm default
```

## Details

### Metadata

Currently, the WASI-NN ggml plugin supports several ways to set the metadata for the inference.

1. From the graph builder

When constructing the graph, you can set the metadata by using the `config` method.

```rust
... wasi_nn::GraphBuilder::new(...).config(options.to_string()) ...
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
        &options.to_string().as_bytes().to_vec(),
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
