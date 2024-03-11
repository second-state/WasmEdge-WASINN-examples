
# Multiple Models Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example for the chaining the results between multiple models.

In this example, we will try asking the `Llava` model a question with a image, and then pass the answer to the `Llama2` model for further response. This example will demonstrate how to use WasmEdge WASI-NN plugin to link two or more models together.

## Get the Model

This example uses the `Llama2` model and `Llava` model. You can download the models from the following links:

```bash
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
curl -LO https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/vicuna-7b-q5_k.gguf
curl -LO https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/mmproj-vicuna7b-f16.gguf
```

## Parameters

> [!NOTE]
> Please check the parameters section of [wasmedge-ggml/README.md](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters) first.

Download the image for the Llava model:

```bash
curl -LO https://llava-vl.github.io/static/images/monalisa.jpg
```

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload the two large models:

```console
$ wasmedge --dir .:. \
  --env image=monalisa.jpg \
  --env mmproj=mmproj-vicuna7b-f16.gguf \
  --nn-preload llama2:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  --nn-preload llava:GGML:AUTO:vicuna-7b-q5_k.gguf \
  wasmedge-ggml-multimodel.wasm

USER:
describe this picture please
ASSISTANT (llava):
The image you've provided appears to be a painting of the Mona Lisa, one of Leonardo da Vinci's most famous works. It is a portrait of a woman with a serene and enigmatic expression, looking directly at the viewer. Her hair is styled in an updo, and she wears a dark dress that drapes elegantly around her shoulders. The background features a landscape with rolling hills and a river, which adds depth to the composition. The painting is renowned for its subtle changes in expression and the enigmatic smile on the subject's face, which has intrigued viewers for centuries.
ASSISTANT (llama2):
The image provided is a painting of the Mona Lisa, one of Leonardo da Vinci's most famous works, depicting a woman with a serene and enigmatic expression, styled in an updo with a dark dress draped elegantly around her shoulders, set against a landscape background with rolling hills and a river.
```
