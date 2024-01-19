use serde_json::{json, Value};
use std::env;
use std::io::{self, Write};
use wasi_nn::{self, GraphExecutionContext};

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
    match env::var("enable_log") {
        Ok(val) => options["enable-log"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    };
    match env::var("enable_debug_log") {
        Ok(val) => options["enable-debug-log"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    };
    match env::var("stream_stdout") {
        Ok(val) => options["stream-stdout"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    };
    match env::var("n_predict") {
        Ok(val) => options["n-predict"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("reverse_prompt") {
        Ok(val) => options["reverse-prompt"] = json!(val),
        _ => (),
    }
    match env::var("n_gpu_layers") {
        Ok(val) => options["n-gpu-layers"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("ctx_size") {
        Ok(val) => options["ctx-size"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("batch_size") {
        Ok(val) => options["batch-size"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("temp") {
        Ok(val) => options["temp"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("repeat_penalty") {
        Ok(val) => options["repeat-penalty"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("threads") {
        Ok(val) => options["threads"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }

    return options;
}

fn set_data_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), wasi_nn::Error> {
    context.set_input(0, wasi_nn::TensorType::U8, &[1], &data)
}

#[allow(dead_code)]
fn set_metadata_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), wasi_nn::Error> {
    context.set_input(1, wasi_nn::TensorType::U8, &[1], &data)
}

fn get_data_from_context(
    context: &GraphExecutionContext,
    index: usize,
    is_compute_single: bool,
) -> String {
    // Preserve for 4096 tokens with average token length 6
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 6;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = if is_compute_single {
        context
            .get_output_single(index, &mut output_buffer)
            .unwrap()
    } else {
        context.get_output(index, &mut output_buffer).unwrap()
    };
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    return String::from_utf8_lossy(&output_buffer[..output_size]).to_string();
}

fn get_output_from_context(context: &GraphExecutionContext, is_compute_single: bool) -> String {
    return get_data_from_context(context, 0, is_compute_single);
}

fn get_metadata_from_context(context: &GraphExecutionContext) -> Value {
    return serde_json::from_str(&get_data_from_context(context, 1, false)).unwrap();
}

fn update_saved_prompt(
    saved_prompt: String,
    input: &String,
    output: &String,
    prompt_format: &String,
) -> String {
    let system_prompt = String::from("You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe." );
    let new_prompt: String;

    if prompt_format == "llama" {
        if saved_prompt == "" {
            new_prompt = format!(
                "[INST] <<SYS>> {} <</SYS>> {} [/INST]",
                system_prompt, input
            );
        } else {
            if output == "" {
                new_prompt = format!("{} [INST] {} [/INST]", saved_prompt, input);
            } else {
                new_prompt = format!("{} {}", saved_prompt, output);
            }
        }
    } else if prompt_format == "chatml" {
        if saved_prompt == "" {
            new_prompt = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", system_prompt, input);
        } else {
            if output == "" {
                new_prompt = format!(
                    "{}<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    saved_prompt, input
                );
            } else {
                new_prompt = format!("{}{}<|im_end|>\n", saved_prompt, output);
            }
        }
    } else {
        println!(
            "[ERROR] prompt_format must be either `llama` or `chatml` (`{}` is invalid).",
            prompt_format
        );
        std::process::exit(1);
    }

    return new_prompt;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let options = get_options_from_env();

    // We suuport both compute() and compute_single().
    // compute() will compute the entire paragraph until the end of sequence, and return the entire output.
    // compute_single() will compute one token at a time, and return the output of the last token.
    let mut is_compute_single = false;
    match env::var("compute_single") {
        Ok(val) => is_compute_single = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }

    // Check streaming related options.
    if is_compute_single && options["stream-stdout"].as_bool().unwrap() {
        println!("[ERROR] compute_single and stream_stdout cannot be enabled at the same time.");
        std::process::exit(1);
    }

    // We support both llama and chatml prompt format.
    let mut prompt_format = String::from("llama");
    match env::var("prompt_format") {
        Ok(val) => prompt_format = val,
        _ => (),
    }
    prompt_format.make_ascii_lowercase();
    if prompt_format != "llama" && prompt_format != "chatml" {
        println!(
            "[ERROR] prompt_format must be either `llama` or `chatml` (`{}` is invalid).",
            prompt_format
        );
        std::process::exit(1);
    }

    // Create graph and initialize context.
    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .config(options.to_string())
            .build_from_cache(model_name)
            .unwrap();
    let mut context = graph.init_execution_context().unwrap();

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
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .unwrap();
        println!("Response:");
        context.compute().unwrap();
        let output = get_output_from_context(&context, false);
        println!("{}", output.trim());
        std::process::exit(0);
    }

    let mut saved_prompt = String::new();

    loop {
        println!("Question:");
        let input = read_input();
        saved_prompt = update_saved_prompt(saved_prompt, &input, &String::new(), &prompt_format);

        // Set prompt to the input tensor.
        set_data_to_context(&mut context, saved_prompt.as_bytes().to_vec()).unwrap();

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

        println!("Answer:");

        let mut output = String::new();
        let mut reset_prompt = false;
        if is_compute_single {
            // Streaming: compute one token at a time, and get the token using the get_output_single().
            loop {
                match context.compute_single() {
                    Ok(_) => (),
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::EndOfSequence)) => {
                        break;
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                        println!("\n[INFO] Context full, we'll reset the context and continue.");
                        reset_prompt = true;
                        break;
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                        println!("\n[INFO] Prompt too long, we'll reset the context and continue.");
                        reset_prompt = true;
                        break;
                    }
                    Err(err) => {
                        println!("\n[ERROR] {}", err);
                        break;
                    }
                }

                // Retrieve the single output token and print it.
                let token = get_output_from_context(&context, is_compute_single);
                print!("{}", token);
                io::stdout().flush().unwrap();
                output += &token;
            }
            println!("");
        } else {
            // Blocking: execute the inference.
            match context.compute() {
                Ok(_) => (),
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                    println!("\n[INFO] Context full, we'll reset the context and continue.");
                    reset_prompt = true;
                }
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                    println!("\n[INFO] Prompt too long, we'll reset the context and continue.");
                    reset_prompt = true;
                }
                Err(err) => {
                    println!("\n[ERROR] {}", err);
                }
            }

            // Retrieve the output.
            output = get_output_from_context(&context, is_compute_single);

            // Skip the output if is streaming.
            if let Some(true) = options["stream-stdout"].as_bool() {
                println!("");
            } else {
                println!("{}", output.trim());
            }
        }

        // Update the saved prompt.
        if reset_prompt {
            saved_prompt.clear();
        } else {
            output = output.trim().to_string();
            saved_prompt = update_saved_prompt(saved_prompt, &input, &output, &prompt_format);
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

        // Delete the context in compute_single mode.
        if is_compute_single {
            context.fini_single().unwrap();
        }
    }
}
