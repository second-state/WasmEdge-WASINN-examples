# `通义千问`

## Execute - Tong Yi Qwen

### Model Download Link

```console
wget https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q2_k.gguf
```

### Execution Command

Please make sure you have the `qwen1_5-0_5b-chat-q2_k.gguf` file in the current directory.
If you want to enable GPU support, please set the `n_gpu_layers` environment variable.
You can also change the `ctx_size` to have a larger context window via `--env ctx_size=8192`. The default value is 1024.

```console
$ wasmedge --dir .:. \
  --env n_gpu_layers=10 \
  --nn-preload default:GGML:AUTO:qwen1_5-0_5b-chat-q2_k.gguf \
  wasmedge-ggml-qwen.wasm default

USER:
你好
ASSISTANT:
你好！有什么我能帮你的吗？
USER:
你是谁
ASSISTANT:
我是一个人工智能助手，我叫通义千问。有什么我可以帮助你的吗？
USER:
能帮助我写Rust代码吗？                                          
ASSISTANT:
当然可以！我可以帮助你使用Rust语言编写代码，帮助你理解和编写出高质量的代码。我可以帮你编写函数、函数和类，提供使用Python解释器编译和运行代码的建议，提供使用Node.js、Python或Java的代码示例，以及更多关于如何使用Python、Java或C++的代码示例。
```
