use serde_json::{json, Value};
use std::env;
use std::io::{self};
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
        options["enable-log"] = serde_json::from_str(val.as_str()).unwrap()
    }
    if let Ok(val) = env::var("ctx_size") {
        options["ctx-size"] = serde_json::from_str(val.as_str()).unwrap()
    }
    if let Ok(val) = env::var("batch_size") {
        options["batch-size"] = serde_json::from_str(val.as_str()).unwrap()
    }
    if let Ok(val) = env::var("threads") {
        options["threads"] = serde_json::from_str(val.as_str()).unwrap()
    }

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
    // Preserve for 4096 embedding size and each embedding number is length 20,
    // and add 128 bytes for other information such as "n_embedding" of other symbols.
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 20 + 128;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = context.get_output(index, &mut output_buffer).unwrap();
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    String::from_utf8_lossy(&output_buffer[..output_size]).to_string()
}

fn get_output_from_context(context: &GraphExecutionContext) -> String {
    get_data_from_context(context, 0)
}

fn get_metadata_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&get_data_from_context(context, 1)).unwrap()
}

fn get_embd_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&get_data_from_context(context, 0)).unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let mut options = get_options_from_env();
    options["embedding"] = serde_json::Value::Bool(true);

    // Create graph and initialize context.
    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(options.to_string())
        .build_from_cache(model_name)
        .expect("Create GraphBuilder Failed, please check the model name or options");
    let mut context = graph
        .init_execution_context()
        .expect("Init Context Failed, please check the model");

    // We also support setting the options via input tensor with index 1.
    // Uncomment the line below to run the example, Check our README for more details.
    //
    // set_metadata_to_context(&mut context, options.to_string().as_bytes().to_vec()).unwrap();

    // If there is a third argument, use it as the prompt and enter non-interactive mode.
    // Otherwise, enter interactive mode.
    if args.len() >= 3 {
        let prompt = &args[2];
        println!("Prompt:\n{}", prompt);
        let tensor_data = prompt.as_bytes().to_vec();
        context
            .set_input(0, TensorType::U8, &[tensor_data.len()], &tensor_data)
            .unwrap();
        println!("Raw Embedding Output:");
        context.compute().unwrap();
        let output = get_output_from_context(&context);
        println!("{}", output.trim());

        let embd = get_embd_from_context(&context);
        println!("Interact with Embedding:");
        let n_embd = embd["n_embedding"].as_u64().unwrap();
        println!("N_Embd: {}", n_embd);
        println!("Show the first 5 elements:");
        for idx in 0..5 {
            println!("embd[{}] = {}", idx, embd["embedding"][idx as usize]);
        }
        std::process::exit(0);
    }

    loop {
        println!("Prompt:");
        let input = read_input();

        // Set prompt to the input tensor.
        set_data_to_context(&mut context, input.as_bytes().to_vec()).unwrap();

        // Get the number of input tokens and llama.cpp versions.
        let input_metadata = get_metadata_from_context(&context);
        if let Some(true) = options["enable-log"].as_bool() {
            println!("[INFO] llama_commit: {}", input_metadata["llama_commit"]);
            println!(
                "[INFO] llama_build_number: {}",
                input_metadata["llama_build_number"]
            );
            println!(
                "[INFO] Number of input tokens: {}",
                input_metadata["input_tokens"]
            );
        }

        match context.compute() {
            Ok(_) => (),
            Err(Error::BackendError(BackendError::ContextFull)) => {
                println!("\n[INFO] Context full");
            }
            Err(Error::BackendError(BackendError::PromptTooLong)) => {
                println!("\n[INFO] Prompt too long");
            }
            Err(err) => {
                println!("\n[ERROR] {}", err);
            }
        }

        // Retrieve the output.
        let output = get_output_from_context(&context);

        println!("Raw Embedding Output: {}", output.trim());

        let embd = get_embd_from_context(&context);
        println!("Interact with Embedding:");
        let n_embd = embd["n_embedding"].as_u64().unwrap();
        println!("N_Embd: {}", n_embd);
        println!("Show the first 5 elements:");
        for idx in 0..5 {
            println!("embd[{}] = {}", idx, embd["embedding"][idx as usize]);
        }

        // Retrieve the output metadata.
        let metadata = get_metadata_from_context(&context);
        if let Some(true) = options["enable-log"].as_bool() {
            println!(
                "[INFO] Number of input tokens: {}",
                metadata["input_tokens"]
            );
            println!(
                "[INFO] Number of output tokens: {}",
                metadata["output_tokens"]
            );
        }
    }
}
