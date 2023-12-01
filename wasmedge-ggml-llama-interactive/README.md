# Llama Example For WASI-NN with GGML Backend

[See it in action!](https://x.com/juntao/status/1705588244602114303)

## Requirement

### For macOS (apple silicon)

Install WasmEdge 0.13.5+WASI-NN ggml plugin(Metal enabled on Apple silicon) via the installer.

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.
# Assuming you use zsh (the default shell on macOS), you will need to run the following command
source $HOME/.zshenv
```

### For Ubuntu (>= 20.04)

#### CUDA enabled

The installer from WasmEdge 0.13.5 will detect cuda automatically.

If CUDA is detected, the installer will always attempt to install a CUDA-enabled version of the plugin.

Install WasmEdge 0.13.5+WASI-NN ggml plugin via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

This version is verified on the following platforms:
1. Nvidia Jetson AGX Orin 64GB developer kit
2. Intel i7-10700 + Nvidia GTX 1080 8G GPU
2. AWS EC2 `g5.xlarge` + Nvidia A10G 24G GPU + Amazon deep learning base Ubuntu 20.04

#### CPU only

If the CPU is the only available hardware on your machine, the installer will install the OpenBLAS version of the plugin instead.

You may need to install `libopenblas-dev` by `apt update && apt install -y libopenblas-dev`.

Install WasmEdge 0.13.5+WASI-NN ggml plugin via installer

```bash
apt update && apt install -y libopenblas-dev # You may need sudo if the user is not root.
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

### For General Linux

Install WasmEdge 0.13.5+WASI-NN ggml plugin via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Linux), you will need to run the following command
source $HOME/.bashrc
```

## Prepare WASM application

### (Recommend) Use the pre-built one bundled in this repo

We built a wasm of this example under the folder, check `wasmedge-ggml-llama-interactive.wasm`

### (Optional) Build from source

If you want to do some modifications, you can build from source.

Compile the application to WebAssembly:

```bash
cargo build --target wasm32-wasi --release
```

The output WASM file will be at `target/wasm32-wasi/release/`.

```bash
cp target/wasm32-wasi/release/wasmedge-ggml-llama-interactive.wasm ./wasmedge-ggml-llama-interactive.wasm
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
  wasmedge-ggml-llama-interactive.wasm default
```

### GPU acceleration

#### macOS

macOS will use the Metal framework by default. You don't have to specify the `n_gpu_layers` parameter.

#### Linux + CUDA

Due to the various GPU hardware, it's hard to set a default value of `n_gpu_layers`.

Please use the following command to ensure the tensor layers of the model is offloaded into GPU:

```
# llama2-7b-chat provides 35 GPU layers. So, we have to set a value that is large or equal to 35.
# If you use a larger model, this value may change.
wasmedge --dir .:. \
  --env n_gpu_layers=35 \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama-interactive.wasm default
```

After executing the command, you may need to wait a moment for the input prompt to appear.
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

Currently, we support setting llama options using `set_input` with index 1.
You can pass the JSON string as a `Vec<u8>` type to `set_input`.

Supported parameters include:

- `enable-log`: Set it to true to enable logging. (default: `false`)
- `stream-stdout`: Set it to true to print the inferred tokens to standard output. (default: `false`)
- `ctx-size`: Set the context size, the same as the `--ctx-size` parameter in llama.cpp. (default: `512`)
- `n-predict`: Set the number of tokens to predict, the same as the `--n-predict` parameter in llama.cpp. (default: `512`)
- `n-gpu-layers`: Set the number of layers to store in VRAM, the same as the `--n-gpu-layers` parameter in llama.cpp. When using Metal support in macOS, please set `n-gpu-layers` to `0` or do not set it for the default value. (default: `0`)
- `reverse-prompt`: Set it to the token at which you want to halt the generation. Similar to the `--reverse-prompt` parameter in llama.cpp. (default: `""`)
- `batch-size`: Set the batch size number for prompt processing, the same as the `--batch-size` parameter in llama.cpp. (default: `512`)

(For more detailed usage instructions regarding the parameters, please refer to [WasmEdge](https://github.com/WasmEdge/WasmEdge/blob/master/plugins/wasi_nn/ggml.cpp).)

For convenience, these parameters could be set by adding the environmental variables in this example.
The environmental variables are handled by Rust. (Due to the limitation of environmental variables, beware of the `-` and `_` in the variable name.)

```bash
wasmedge --dir .:. \
  --env stream_stdout=false \
  --env enable_log=false \
  --env ctx_size=512 \
  --env n_predict=512 \
  --env n_gpu_layers=0 \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama-interactive.wasm default
```

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

## Credit

The WASI-NN ggml plugin embedded [`llama.cpp`](git://github.com/ggerganov/llama.cpp.git) as its backend.
For details about the version used, please refer to [WasmEdge/LICENSE.spdx](https://github.com/WasmEdge/WasmEdge/blob/master/LICENSE.spdx).
