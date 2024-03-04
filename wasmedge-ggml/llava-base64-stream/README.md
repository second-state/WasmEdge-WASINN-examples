# Llava Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of the Llava model.
> Refer to the [wasmedge-ggml/llava/README.md](../llava/README.md) for downloading Llava models and execution commands.

This example is to demonstrate the usage of the Llava model inference with inline base64 encoded image. Here we hardcode the base64 encoded image in the source code.

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

> [!NOTE]
> You may see some warnings stating `key clip.vision.* not found in file.` when using llava v1.5 models. These are expected and can be ignored.

```console
$ wasmedge --dir .:. \
  --env mmproj=mmproj-model-f16.gguf \
  --nn-preload default:GGML:AUTO:ggml-model-q5_k.gguf \
  wasmedge-ggml-llava-base64-stream.wasm default

USER:
what is in this picture?
ASSISTANT:
The image showcases a bowl filled with an assortment of fresh berries, including several strawberries and blueberries. A person is standing close to the bowl, holding it in their hand or about to grab some fruit from it. The colorful fruit arrangement adds vibrancy to the scene.
USER:
please tell me a kind of fruit that is not in the picture
ASSISTANT:
There are no bananas in the picture.
```
