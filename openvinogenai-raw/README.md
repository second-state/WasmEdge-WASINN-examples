# Deepseek example with WasmEdge WASI-NN OpenVINO GenAI plugin

This example demonstrates how to use WasmEdge WASI-NN OpenVINO GenAI plugin to perform an inference task with Deepseek model.

## Set up the environment

- Install `rustup` and `Rust`

  Go to the [official Rust webpage](https://www.rust-lang.org/tools/install) and follow the instructions to install `rustup` and `Rust`.

  > It is recommended to use Rust 1.68 or above in the stable channel.

  Then, add `wasm32-wasip1` target to the Rustup toolchain:

  ```bash
  rustup target add wasm32-wasip1
  ```

- Clone the example repo

  ```bash
  git clone https://github.com/second-state/WasmEdge-WASINN-examples.git
  ```

- Install OpenVINO GenAI

  Please refer to [WasmEdge Docs](https://wasmedge.org/docs/contribute/source/plugin/wasi_nn) and [OpenVINO™ GenAI](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html) for the installation process.

  ```bash
  # ensure OpenVINO environment initialized
  source setupvars.sh
  ```

- Build WasmEdge with Wasi-NN OpenVINO GenAI plugin from source

  ```bash
  docker run -v $(pwd):/code -it --rm wasmedge/wasmedge:ubuntu-build-clang-plugins-deps bash
  cd /code
  cmake -Bbuild -GNinja -DWASMEDGE_PLUGIN_WASI_NN_BACKEND=openvinogenai . 
  ```

## Build and run `openvinogenai-deepseek-raw` example

- Download `Deepseek` model file ([huggingface](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B))

- Convert the model by optimum-intel

  ```bash
  python3 -m venv .venv
  . .venv/bin/activate

  pip install --upgrade --upgrade-strategy eager "optimum[openvino]"  

  optimum-cli export openvino --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 --sym DeepSeek-R1-Distill-Qwen-1.5B/INT4_compressed_weights
  ```

- Adjust the DeepSeek chat template

  OpenVINO GenAI may not accept the default `chat_template` in `openvino_tokienizer.xml`. Replace it with a valid template:

  ```xml
  <rt_info>
		<add_attention_mask value="True" />
		<add_prefix_space />
		<add_special_tokens value="True" />
		<bos_token_id value="151646" />
		<chat_template value="... /*replace this*/" />
  ```

  You can refer this information and use the template in ```llm_config.py``` : [Openvino: LLM reasoning with DeepSeek-R1 distilled models](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/deepseek-r1) 

- Build the example

  ```bash 
  cargo build --target wasm32-wasip1 --release 
  ```

- Run the example

  ```bash
  wasmedge ./target/wasm32-wasip1/release/openvinogenai-deepseek-raw.wasm path_to_model_xml_folder
  ```

  You will get the output:

  ```console
  Load graph ...done
  Init execution context ...done
  Set input tensor ...done
  Generating ...done
  Get the result ...Retrieve the output ...done
  The size of the output buffer is 285 bytes
  Output:  I'm a student, and I need to solve this problem: Given a function f(x) = x^3 + 3x^2 + 3x + 1, and a function g(x) = x^2 + 2x + 1. I need to find the number of real roots of f(x) and g(x). Also, I need to find the number of real roots of f(x) + g(x). Please explain step by step. I'm a
  done
  ```
