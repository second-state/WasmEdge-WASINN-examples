# Llama Example For WASI-NN with GGML Backend

[See it in action!](https://x.com/juntao/status/1705588244602114303)

## Requirement

### For macOS (apple silicon)

Install WasmEdge 0.13.4+WASI-NN ggml plugin(Metal enabled on apple silicon) via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After install the wasmedge, you have to activate the environment.
# Assuming you use zsh (the default shell on macOS), you will need to run the following command
source $HOME/.zshenv
```

### For Ubuntu (>= 20.04)

Because we enabled OpenBLAS on Ubuntu, you must install `libopenblas-dev` by `apt update && apt install -y libopenblas-dev`.

Install WasmEdge 0.13.4+WASI-NN ggml plugin(OpenBLAS enabled) via installer

```bash
apt update && apt install -y libopenblas-dev # You may need sudo if the user is not root.
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After install the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

### For General Linux

Install WasmEdge 0.13.4+WASI-NN ggml plugin via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After install the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
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

Execute the WASM with the `wasmedge` using the named model feature to preload large model:

```bash
wasmedge --dir .:. \
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

## Errors

- After running `apt update && apt install -y libopenblas-dev`, you may encountered the following error:

  ```bash
  ...
  E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
  E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
  ```

   This indicates that you are not logged in as `root`. Please try installing again using the `sudo` command:

  ```bash
  sudo apt update && sudo apt install -y libopenblas-dev
  ```

- After running the `wasmedge` command, you may received the following error:

  ```bash
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  unknown option: nn-preload
  ```

  This suggests that your plugin installation was not successful. To resolve this issue, please attempt to install your desired plugin again.

## Parameters

Currently, we support setting llama options using `set_input` with index 1.
You can pass the JSON string as a `Vec<u8>` type to `set_input`.

Supported parameters include:

- `enable-log`: Set it to true to enable logging.
- `stream-stdout`: Set it to true to print the inferred tokens to standard output.
- `ctx-size`: Set the context size, the same as the `--ctx-size` parameter in llama.cpp.
- `n-predict`: Set the number of tokens to predict, the same as the `--n-predict` parameter in llama.cpp.
- `n-gpu-layers`: Set the number of layers to store in VRAM, the same as the `--n-gpu-layers` parameter in llama.cpp.
- `reverse-prompt`: Set it to the token at which you want to halt the generation. Similar to the `--reverse-prompt` parameter in llama.cpp.
- `batch-size`: Set the number of batch size for prompt processing, the same as the `--batch-size` parameter in llama.cpp.

These parameters can be set by adding the following environment variables before the `wasmedge` command:

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama-interactive.wasm default
```

## Credit

The WASI-NN ggml plugin embedded [`llama.cpp`](git://github.com/ggerganov/llama.cpp.git@b1309) as its backend.
