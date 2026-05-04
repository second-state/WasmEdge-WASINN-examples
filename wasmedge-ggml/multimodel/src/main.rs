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

    // Required parameters for llava
    if let Ok(val) = env::var("mmproj") {
        options["mmproj"] = Value::from(val.as_str());
    } else {
        eprintln!("Failed to get mmproj model.");
        std::process::exit(1);
    }
    if let Ok(val) = env::var("image") {
        options["image"] = Value::from(val.as_str());
    } else {
        eprintln!("Failed to get the target image.");
        std::process::exit(1);
    }

    // Optional parameters
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

#[allow(dead_code)]
fn get_metadata_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&get_data_from_context(context, 1)).expect("Failed to get metadata")
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Set options for the graph. Check our README for more details:
    // https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters
    let mut options = get_options_from_env();
    // We set the temperature to 0.1 for more consistent results.
    options["temp"] = Value::from(0.1);
    // Set the context size to 4096 tokens for the llava 1.6 model.
    options["ctx-size"] = Value::from(4096);

    // Create the llava model.
    let mut graphs = Vec::new();
    graphs.push(
        GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
            .config(serde_json::to_string(&options).expect("Failed to serialize options"))
            .build_from_cache("llava")
            .expect("Failed to build graph"),
    );

    // Remove unnecessary options for the llama2 model.
    options
        .as_object_mut()
        .expect("Failed to get jsons object")
        .remove("mmproj");
    options
        .as_object_mut()
        .expect("Failed to get json object")
        .remove("image");
    // Create the llama2 model.
    graphs.push(
        GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
            .config(serde_json::to_string(&options).expect("Failed to serialize options"))
            .build_from_cache("llama2")
            .expect("Failed to build graph"),
    );

    // Initilize the execution contexts.
    let mut contexts = Vec::new();
    contexts.push(
        graphs[0]
            .init_execution_context()
            .expect("Failed to init context"),
    );
    contexts.push(
        graphs[1]
            .init_execution_context()
            .expect("Failed to init context"),
    );

    let system_prompt = String::from("You are a helpful, respectful and honest assistant.");
    let mut input = String::from("");

    // If the user provides a prompt, use it.
    println!("USER:");
    if args.len() >= 2 {
        input += &args[1];
        println!("{}", input);
    } else {
        input = read_input();
    }

    // Llava inference.
    let image_placeholder = "<image>";
    let mut saved_prompt = format!(
        "{}\nUSER:{}\n{}\nASSISTANT:",
        system_prompt, image_placeholder, input
    );
    set_data_to_context(&mut contexts[0], saved_prompt.as_bytes().to_vec())
        .expect("Failed to set input");
    match contexts[0].compute() {
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

    // Retrieve the llava output.
    let mut output = get_output_from_context(&contexts[0]);
    println!("ASSISTANT (llava):\n{}", output.trim());

    // Llama2 inference.
    let llama2_prompt = "Summarize the following text in 1 sentence:";
    saved_prompt = format!(
        "[INST] <<SYS>> {} <</SYS>> {} {} [/INST]",
        system_prompt,
        llama2_prompt,
        output.trim()
    );
    set_data_to_context(&mut contexts[1], saved_prompt.as_bytes().to_vec())
        .expect("Failed to set input");
    match contexts[1].compute() {
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

    // Retrieve the llama2 output.
    output = get_output_from_context(&contexts[1]);
    println!("ASSISTANT (llama2):\n{}", output.trim());
}
