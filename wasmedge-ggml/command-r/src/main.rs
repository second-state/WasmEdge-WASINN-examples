use serde_json::json;
use serde_json::Value;
use std::env;
use std::io;
use wasmedge_wasi_nn::{
    self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};

fn read_input() -> String {
    loop {
        let mut answer = String::new();
        io::stdin()
            .read_line(&mut answer)
            .expect("Failed to read line");
        if !answer.is_empty() && answer != "\n" && answer != "\r\n" {
            return answer.trim().to_string();
        }
    }
}

fn get_options_from_env() -> Value {
    let mut options = json!({});
    if let Ok(val) = env::var("enable_log") {
        options["enable-log"] = serde_json::from_str(val.as_str())
            .expect("invalid value for enable-log option (true/false)")
    } else {
        options["enable-log"] = serde_json::from_str("false").unwrap()
    }
    if let Ok(val) = env::var("n_gpu_layers") {
        options["n-gpu-layers"] =
            serde_json::from_str(val.as_str()).expect("invalid ngl value (unsigned integer")
    } else {
        options["n-gpu-layers"] = serde_json::from_str("0").unwrap()
    }
    options["ctx-size"] = serde_json::from_str("1024").unwrap();

    options
}

fn set_data_to_context(context: &mut GraphExecutionContext, data: Vec<u8>) -> Result<(), Error> {
    context.set_input(0, TensorType::U8, &[data.len()], &data)
}

#[allow(dead_code)]
fn set_metadata_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), Error> {
    context.set_input(1, TensorType::U8, &[data.len()], &data)
}

fn get_data_from_context(context: &GraphExecutionContext, index: usize) -> String {
    // Preserve for 4096 tokens with average token length 6
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 6;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = context
        .get_output(index, &mut output_buffer)
        .expect("Failed to get output");
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    return String::from_utf8_lossy(&output_buffer[..output_size]).to_string();
}

fn get_output_from_context(context: &GraphExecutionContext) -> String {
    get_data_from_context(context, 0)
}

fn get_metadata_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&get_data_from_context(context, 1)).expect("Failed to get metadata")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];

    // Set options for the graph. Check our README for more details:
    // https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters
    let options = get_options_from_env();

    // Create graph and initialize context.
    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&options).expect("Failed to serialize options"))
        .build_from_cache(model_name)
        .expect("Failed to build graph");
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");

    // We also support setting the options via input tensor with index 1.
    // Uncomment the line below to run the example, Check our README for more details.
    // set_metadata_to_context(
    //     &mut context,
    //     serde_json::to_string(&options)
    //         .expect("Failed to serialize options")
    //         .as_bytes()
    //         .to_vec(),
    // )
    // .expect("Failed to set metadata");

    // If there is a third argument, use it as the prompt and enter non-interactive mode.
    // This is mainly for the CI workflow.
    if args.len() >= 3 {
        let prompt = &args[2];
        // Set the prompt.
        println!("Prompt:\n{}", prompt);
        let tensor_data = prompt.as_bytes().to_vec();
        context
            .set_input(0, TensorType::U8, &[tensor_data.len()], &tensor_data)
            .expect("Failed to set input");
        println!("Response:");

        // Get the number of input tokens and llama.cpp versions.
        let input_metadata = get_metadata_from_context(&context);
        println!("[INFO] llama_commit: {}", input_metadata["llama_commit"]);
        println!(
            "[INFO] llama_build_number: {}",
            input_metadata["llama_build_number"]
        );
        println!(
            "[INFO] Number of input tokens: {}",
            input_metadata["input_tokens"]
        );

        // Get the output.
        context.compute().expect("Failed to compute");
        let output = get_output_from_context(&context);
        println!("{}", output.trim());

        // Retrieve the output metadata.
        let metadata = get_metadata_from_context(&context);
        println!(
            "[INFO] Number of input tokens: {}",
            metadata["input_tokens"]
        );
        println!(
            "[INFO] Number of output tokens: {}",
            metadata["output_tokens"]
        );
        std::process::exit(0);
    }

    let mut saved_prompt = String::new();
    let system_tool_prompt = r#"
        # Safety Preamble
        The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

        # System Preamble
        ## Basic Rules
        You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

        # User Preamble
        ## Task and Context
        You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

        ## Available Tools
        Here is a list of tools that you have available to you:

        ```python
        def internet_search(query: str) -> List[Dict]:
            """Returns a list of relevant document snippets for a textual query retrieved from the internet

            Args:
                query (str): Query to search the internet with
            """
            pass
        ```

        ```python
        def directly_answer() -> List[Dict]:
            """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
            """
            pass
        ```
    "#;
    let system_instruction_prompt = r#"
        Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
        ```json
        [
            {
                "tool_name": title of the tool in the specification,
                "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
            }
        ]```
    "#;

    loop {
        println!("USER:");
        let input = read_input();
        //
        if saved_prompt.is_empty() {
            saved_prompt = format!(
                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>\
                <|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>\
                <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>\
                <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                system_tool_prompt, input, system_instruction_prompt
            );
        } else {
            saved_prompt = format!("{} <|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", saved_prompt, input);
        }

        // Set prompt to the input tensor.
        set_data_to_context(&mut context, saved_prompt.as_bytes().to_vec())
            .expect("Failed to set input");

        // Execute the inference.
        let mut reset_prompt = false;
        match context.compute() {
            Ok(_) => (),
            Err(Error::BackendError(BackendError::ContextFull)) => {
                println!("\n[INFO] Context full, we'll reset the context and continue.");
                reset_prompt = true;
            }
            Err(Error::BackendError(BackendError::PromptTooLong)) => {
                println!("\n[INFO] Prompt too long, we'll reset the context and continue.");
                reset_prompt = true;
            }
            Err(err) => {
                println!("\n[ERROR] {}", err);
            }
        }

        // Retrieve the output.
        let mut output = get_output_from_context(&context);
        println!("ASSISTANT:\n{}", output.trim());

        // Update the saved prompt.
        if reset_prompt {
            saved_prompt.clear();
        } else {
            output = output.trim().to_string();
            saved_prompt = format!("{} {} <|END_OF_TURN_TOKEN|>", saved_prompt, output);
        }
    }
}
