# Mllama Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of the Mllama model.

## Get Mllama Model

In this example, we are going to use the pre-converted [llama3.2-vision-11b](https://ollama.com/library/llama3.2-vision) model.

For downloading the mllama model, please download and install [Ollama](https://ollama.com/) first.

After installing the `Ollama`, fetching the model by the command:

```bash
ollama pull llama3.2-vision
```

The model will in the `~/.ollama/models/blobs` directory. (Take `llama3.2-vision-11b` for example.)

* Model: `sha256-11f274007f093fefeec994a5dbbb33d0733a4feb87f7ab66dcd7c1069fef0068`
* Projector: `sha256-ece5e659647a20a5c28ab9eea1c12a1ad430bc0f2a27021d00ad103b3bf5206f`

## Prepare the Image

Download the image for the Mllama model:

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

## Build (WIP)

> Note: Currently, users should build from source.

1. Fetch WasmEdge and checkout the `wasi_nn_ggml_mllama` branch.

    ```bash
    git clone https://github.com/WasmEdge/WasmEdge.git
    cd WasmEdge
    git checkout wasi_nn_ggml_mllama
    ```

2. Build with WASI-NN GGML backend.

    ```bash
    cd <path/to/your/wasmedge/source/folder>
    cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="GGML"
    cmake --build build
    ```

## Execute (WIP)

Execute the WASM with the `wasmedge` using the named model feature to preload a large model:

> Note: Because of building from source, we take run in the build folder for example.

```console
$ cd <path/to/your/wasmedge/source/folder>
$ cd build/tools/wasmedge 
$ WASMEDGE_PLUGIN_PATH=../../plugins/wasi_nn ./wasmedge --dir .:. \
  --env mllamaproj=sha256-ece5e659647a20a5c28ab9eea1c12a1ad430bc0f2a27021d00ad103b3bf5206f \
  --nn-preload default:GGML:AUTO:sha256-11f274007f093fefeec994a5dbbb33d0733a4feb87f7ab66dcd7c1069fef0068 \
  wasmedge-ggml-mllama.wasm default

USER:
please describe this image
IMAGE_PATH: (press enter if you don't want to add image)
monalisa.jpg
ASSISTANT:
The image is a painting of a woman with long dark hair and a slight smile, wearing a Renaissance-style dress. The painting is likely a self-portrait of Leonardo da Vinci, as it is similar in style to his famous works such as the Mona Lisa. The woman's face is serene and calm, with a subtle hint of a smile playing on her lips. Her eyes are cast downward, giving the impression that she is lost in thought. Her hair is dark and flowing, cascading down her back in loose waves. She wears a Renaissance-style dress that is loose-fitting and elegant, with a high neckline and long sleeves. The dress is a deep shade of blue or purple, which complements the woman's skin tone nicely. The background of the painting is a soft, muted color that blends seamlessly into the woman's dress. There are no distinct features or objects in the background, which helps to focus the viewer's attention on the woman's face and figure. Overall, the painting is a beautiful example of Renaissance art, with its use of muted colors, elegant lines, and serene expression. It is likely that the woman depicted in the painting is a member of the artist's family or a wealthy patron, given the level of detail and craftsmanship that went into creating the piece.<|eot_id|>
```
