use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::io;
use wasmedge_wasi_nn::{
    self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};

const MULTIMODAL_IMAGE_MARKER: &str = "<image>";

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

fn get_options_from_env() -> HashMap<&'static str, Value> {
    let mut options = HashMap::new();

    let mmproj = env::var("mmproj").ok();
    let image = env::var("image").ok();

    match (mmproj, image) {
        (Some(mmproj), Some(image)) => {
            options.insert("mmproj", Value::from(mmproj));
            options.insert("image", Value::from(image));
        }
        (None, None) => {}
        (Some(_), None) => {
            eprintln!("The `image` environment variable is required when `mmproj` is set.");
            std::process::exit(1);
        }
        (None, Some(_)) => {
            eprintln!("The `mmproj` environment variable is required when `image` is set.");
            std::process::exit(1);
        }
    }

    // Optional parameters
    if let Ok(val) = env::var("enable_log") {
        options.insert("enable-log", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("enable-log", Value::from(false));
    }
    if let Ok(val) = env::var("enable_debug_log") {
        options.insert(
            "enable-debug-log",
            serde_json::from_str(val.as_str()).unwrap(),
        );
    } else {
        options.insert("enable-debug-log", Value::from(false));
    }
    if let Ok(val) = env::var("ctx_size") {
        options.insert("ctx-size", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("ctx-size", Value::from(4096));
    }
    if let Ok(val) = env::var("n_gpu_layers") {
        options.insert("n-gpu-layers", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("n-gpu-layers", Value::from(0));
    }
    options
}

fn set_data_to_context(context: &mut GraphExecutionContext, data: Vec<u8>) -> Result<(), Error> {
    context.set_input(0, TensorType::U8, &[data.len()], &data)
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

fn get_metadata_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&get_data_from_context(context, 1)).expect("Failed to get metadata")
}

fn get_system_prompt_from_env() -> String {
    env::var("system_prompt").unwrap_or_else(|_| "You are a helpful assistant.".to_string())
}

fn get_enable_thinking_from_env() -> bool {
    env::var("enable_thinking")
        .ok()
        .and_then(|v| serde_json::from_str::<bool>(&v).ok())
        .unwrap_or(true)
}

fn build_gemma4_system_turn(system_prompt: &str, enable_thinking: bool) -> String {
    if enable_thinking {
        format!("<|turn>system\n<|think|>{}<turn|>\n", system_prompt)
    } else {
        format!("<|turn>system\n{}<turn|>\n", system_prompt)
    }
}

fn build_gemma4_user_turn(user_content: &str) -> String {
    format!("<|turn>user\n{}<turn|>\n<|turn>model\n", user_content)
}

fn build_user_content(user_input: &str, multimodal_enabled: bool) -> String {
    if multimodal_enabled {
        format!("{}\n{}", MULTIMODAL_IMAGE_MARKER, user_input)
    } else {
        user_input.to_string()
    }
}

fn strip_gemma4_thoughts(text: &str) -> String {
    let mut cleaned = text.trim().to_string();
    let thought_start_tags = ["<|channel|>thought", "<|channel>thought"];
    let thought_end_tags = ["<|channel|>", "<channel|>"];

    loop {
        let Some(start) = thought_start_tags
            .iter()
            .filter_map(|tag| cleaned.find(tag))
            .min()
        else {
            break;
        };

        let search_start = start + 1;
        let Some(rel_end) = thought_end_tags
            .iter()
            .filter_map(|tag| cleaned[search_start..].find(tag).map(|idx| (idx, tag.len())))
            .min_by_key(|(idx, _)| *idx)
        else {
            cleaned.truncate(start);
            break;
        };

        let end = search_start + rel_end.0 + rel_end.1;
        cleaned.replace_range(start..end, "");
    }

    cleaned.trim().to_string()
}

fn strip_gemma4_turn_suffix(text: &str) -> String {
    text.trim()
        .strip_suffix("<turn|>")
        .unwrap_or(text.trim())
        .trim()
        .to_string()
}

fn parse_gemma4_output(text: &str) -> (String, String) {
    let without_thoughts = strip_gemma4_thoughts(text);
    let visible_output = strip_gemma4_turn_suffix(&without_thoughts);
    let prompt_output = if visible_output.is_empty() {
        String::new()
    } else {
        format!("{}<turn|>\n", visible_output)
    };

    (visible_output, prompt_output)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];

    // Set options for the graph. Check our README for more details:
    // https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters
    let options = get_options_from_env();
    // You could also set the options manually like this:

    // Create graph and initialize context.
    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&options).expect("Failed to serialize options"))
        .build_from_cache(model_name)
        .expect("Failed to build graph");
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");

    let system_prompt = get_system_prompt_from_env();
    let enable_thinking = get_enable_thinking_from_env();
    let system_turn = build_gemma4_system_turn(&system_prompt, enable_thinking);
    let multimodal_enabled = options.contains_key("mmproj") && options.contains_key("image");

    // Non-interactive mode
    if args.len() >= 3 {
        let user_input = &args[2];
        let prompt = format!(
            "{}{}",
            system_turn,
            build_gemma4_user_turn(&build_user_content(user_input, multimodal_enabled))
        );

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
        let (visible_output, _) = parse_gemma4_output(&output);
        println!("{}", visible_output);

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
        return;
    }

    let mut saved_prompt = system_turn.clone();

    loop {
        println!("USER:");
        let input = read_input();

        let user_turn = build_gemma4_user_turn(&build_user_content(&input, multimodal_enabled));
        saved_prompt.push_str(&user_turn);

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
                std::process::exit(1);
            }
        }

        // Retrieve the output.
        let output = get_output_from_context(&context);
        let (visible_output, prompt_output) = parse_gemma4_output(&output);
        println!("ASSISTANT:\n{}", visible_output);

        // Update the saved prompt.
        if reset_prompt {
            saved_prompt = system_turn.clone();
        } else {
            saved_prompt.push_str(&prompt_output);
        }
    }
}