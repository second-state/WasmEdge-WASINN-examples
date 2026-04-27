use serde_json::Value;
use std::collections::HashMap;
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

fn read_image_path() -> String {
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .expect("");
    return answer.trim().to_string();
}

fn get_options_from_env() -> HashMap<&'static str, Value> {
    let mut options = HashMap::new();

    // Required parameters for mllama
    if let Ok(val) = env::var("mllamaproj") {
        options.insert("mllamaproj", Value::from(val.as_str()));
    } else {
        eprintln!("Failed to get mllamaproj model.");
        std::process::exit(1);
    }

    // Optional parameters
    if let Ok(val) = env::var("enable_log") {
        options.insert("enable-log", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("enable-log", Value::from(false));
    }
    if let Ok(val) = env::var("ctx_size") {
        options.insert("ctx-size", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("ctx-size", Value::from(2048));
    }
    if let Ok(val) = env::var("n_gpu_layers") {
        options.insert("n-gpu-layers", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("n-gpu-layers", Value::from(0));
    }
    options
}

fn set_data_to_context(context: &mut GraphExecutionContext, data: Vec<u8>) -> Result<(), Error> {
    context.set_input(0, TensorType::U8, &[1], &data)
}


fn set_metadata_to_context(context: &mut GraphExecutionContext, data: Vec<u8>) -> Result<(), Error> {
    context.set_input(1, TensorType::U8, &[1], &data)
}

fn get_data_from_context(context: &GraphExecutionContext, index: usize) -> String {
    // Preserve for 4096 tokens with average token length 6
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 6;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = context
        .get_output(index, &mut output_buffer)
        .expect("Failed to get output");
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    String::from_utf8_lossy(&output_buffer[..output_size]).to_string()
}

fn get_output_from_context(context: &GraphExecutionContext) -> String {
    get_data_from_context(context, 0)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];

    // Set options for the graph. Check our README for more details:
    // https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters
    let mut options = get_options_from_env();
    // Set the stream-stdout option to true to make the response more interactive.
    options.insert("stream-stdout", serde_json::from_str("true").unwrap());
    // You could also set the options manually like this:
    // options.insert("enable-log", Value::from(false));

    // Create graph and initialize context.
    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&options).expect("Failed to serialize options"))
        .build_from_cache(model_name)
        .expect("Failed to build graph");
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");

    // If there is a third argument, use it as the prompt and enter non-interactive mode.
    // This is mainly for the CI workflow.
    if args.len() >= 3 {
        let prompt = &args[2];
        println!("Prompt:\n{}", prompt);
        let tensor_data = prompt.as_bytes().to_vec();
        context
            .set_input(0, TensorType::U8, &[1], &tensor_data)
            .expect("Failed to set input");
        println!("Response:");
        context.compute().expect("Failed to compute");
        let output = get_output_from_context(&context);
        if let Some(true) = options["stream-stdout"].as_bool() {
            println!();
        } else {
            println!("{}", output.trim());
        }
        std::process::exit(0);
    }

    let image_placeholder = "<|image|>";

    loop {
        println!("USER:");
        let input = read_input();

        println!("IMAGE_PATH: (press enter if you don't want to add image)");
        let image_path = read_image_path();

        if !image_path.is_empty() {
            set_metadata_to_context(
                &mut context,
                format!("{{\"image\": \"{}\"}}", image_path).as_bytes().to_vec(),
            ).expect("Failed to set metadata");
        }

        let prompt: String;
        // mllama chat format is "<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        if !image_path.is_empty() {
            prompt = format!(
                "<|start_header_id|>user<|end_header_id|>{} {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                image_placeholder, input
            );
        } else {
            prompt = format!(
                "<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                input);
        }

        // Set prompt to the input tensor.
        set_data_to_context(&mut context, prompt.as_bytes().to_vec())
            .expect("Failed to set input");

        // Execute the inference.
        println!("ASSISTANT:");
        match context.compute() {
            Ok(_) => (),
            Err(Error::BackendError(BackendError::ContextFull)) => {
                println!("\n[INFO] Context full, we'll reset the context and continue.");
            }
            Err(Error::BackendError(BackendError::PromptTooLong)) => {
                println!("\n[INFO] Prompt too long, we'll reset the context and continue.");
            }
            Err(err) => {
                println!("\n[ERROR] {}", err);
            }
        }

        // Retrieve the output.
        let output = get_output_from_context(&context);
        if let Some(true) = options["stream-stdout"].as_bool() {
            println!();
        } else {
            println!("{}", output.trim());
        }
    }
}
