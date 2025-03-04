# `tts` - Text-to-Speech Example

## Model Download

```console
wget https://huggingface.co/second-state/OuteTTS-0.2-500M-GGUF/resolve/main/OuteTTS-0.2-500M-Q5_K_M.gguf
wget https://huggingface.co/second-state/OuteTTS-0.2-500M-GGUF/resolve/main/wavtokenizer-large-75-ggml-f16.gguf
```

## Speaker Profile Download

```console
wget https://raw.githubusercontent.com/edwko/OuteTTS/refs/heads/main/outetts/version/v1/default_speakers/en_male_1.json
```

> [!NOTE]
> The default speaker profile of the plugin is `en_female_1.json`.

### Execution

```console
$ wasmedge --dir .:. \
  --env tts=true \
  --env tts_output_file=output.wav \
  --env tts_speaker_file=en_male_1.json \
  --env model_vocoder=wavtokenizer-large-75-ggml-f16.gguf \
  --nn-preload default:GGML:AUTO:OuteTTS-0.2-500M-Q5_K_M.gguf \
  wasmedge-ggml-tts.wasm default 'Hello, world.'

Prompt:
Hello, world!
[INFO] llama_commit: "955a6c2d"
[INFO] llama_build_number: 4534
[INFO] Number of input tokens: 891
[INFO] Write output file to "output.wav"
```
