# Llava Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of the Llava model.

## Get Llava Model

In this example, we are going to use the pre-converted [llava-v1.5-7b](https://huggingface.co/mys/ggml_llava-v1.5-7b) model.

Download the llava v1.5 model:

```bash
curl -LO https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q5_k.gguf
curl -LO https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf
```

or use the llava v1.6 model:

```bash
curl -LO https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/vicuna-7b-q5_k.gguf
curl -LO https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/mmproj-vicuna7b-f16.gguf
```

## Prepare the Image

Download the image for the Llava model:

```bash
curl -LO https://llava-vl.github.io/static/images/monalisa.jpg
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

For GPU offloading, please adjust the `n-gpu-layers` options to the number of layers that you want to offload to the GPU.

```rust
options.insert("n-gpu-layers", Value::from(...));
```

In llava inference, we recommend to use the `ctx-size` at least `2048` when using llava-v1.5 and at least `4096` when using llava-v1.6 for better results.

```rust
options.insert("ctx-size", Value::from(4096));
```

## Execute (llava-1.5)

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

> [!NOTE]
> You may see some warnings stating `key clip.vision.* not found in file.` when using llava v1.5 models. These are expected and can be ignored.

```console
$ wasmedge --dir .:. \
  --env mmproj=mmproj-model-f16.gguf \
  --env image=monalisa.jpg \
  --nn-preload default:GGML:AUTO:ggml-model-q5_k.gguf \
  wasmedge-ggml-llava.wasm default

USER:
what is in this picture?
ASSISTANT:
The image features a painting or portrait of a woman with long hair, likely inspired by the famous Mona Lisa artwork. She appears to be smiling and is surrounded by an ocean view, giving the impression of being on a boat. The scene is painted in black and white, creating a classic and timeless look. Additionally, there are two birds present within this painting, one on the left side towards the top and another slightly higher up in the center.
USER:
Do you know who drew this painting?
ASSISTANT:
As a visual AI, I don't have knowledge about the artist who created this painting. However, it is inspired by the famous Mona Lisa artwork, which was painted by Leonardo da Vinci.
```

## Execute (llava-1.6)

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

> [!NOTE]
> For the llava-1.6 model, we will need to set the context size to at least 4096 to work.

```console
$ wasmedge --dir .:. \
  --env mmproj=mmproj-vicuna7b-f16.gguf \
  --env image=monalisa.jpg \
  --env ctx_size=4096 \
  --nn-preload default:GGML:AUTO:vicuna-7b-q5_k.gguf \
  wasmedge-ggml-llava.wasm default

USER:
what is in this picture?
ASSISTANT:
The image you've provided appears to be a portrait of the famous artwork known as the Mona Lisa, one of Leonardo da Vinci's most recognizable paintings. It features a woman with a subtly enigmatic smile and her gaze directed slightly off-center towards the viewer. The background is filled with elements of Renaissance landscape painting, including rolling hills, trees, waterways, and a distant city skyline, creating a serene yet intriguing atmosphere that complements the subject's ethereal presence.
USER:
Do you know who drew this painting?
ASSISTANT:
Yes, the painting was created by Leonardo da Vinci. It is one of his most celebrated works and is known for its enigmatic expression and the artist's attention to detail in both the subject's features and the landscape elements surrounding her.
```
