use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use wasmedge_wasi_nn::{
    self, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};

fn get_options_from_env() -> HashMap<&'static str, Value> {
    let mut options = HashMap::new();

    // Required parameters for tts
    options.insert("tts", Value::from(true));
    if let Ok(val) = env::var("tts_output_file") {
        options.insert("tts-output-file", Value::from(val.as_str()));
    } else {
        eprintln!("Failed to get output file name.");
        std::process::exit(1);
    }
    if let Ok(val) = env::var("model_vocoder") {
        options.insert("model-vocoder", Value::from(val.as_str()));
    } else {
        eprintln!("Failed to get vocoder model.");
        std::process::exit(1);
    }
    // Speaker profile is optional.
    if let Ok(val) = env::var("tts_speaker_file") {
        options.insert("tts-speaker-file", Value::from(val.as_str()));
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
    if let Ok(val) = env::var("n_predict") {
        options.insert("n-predict", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("n-predict", Value::from(4096));
    }
    if let Ok(val) = env::var("ctx_size") {
        options.insert("ctx-size", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("ctx-size", Value::from(8192));
    }
    if let Ok(val) = env::var("batch_size") {
        options.insert("batch-size", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("batch-size", Value::from(8192));
    }
    if let Ok(val) = env::var("ubatch_size") {
        options.insert("ubatch-size", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("ubatch-size", Value::from(8192));
    }
    if let Ok(val) = env::var("n_gpu_layers") {
        options.insert("n-gpu-layers", serde_json::from_str(val.as_str()).unwrap());
    } else {
        options.insert("n-gpu-layers", Value::from(100));
    }
    if let Ok(val) = env::var("seed") {
        options.insert("seed", serde_json::from_str(val.as_str()).unwrap());
    }
    if let Ok(val) = env::var("temp") {
        options.insert("temp", serde_json::from_str(val.as_str()).unwrap());
    }
    options
}

#[allow(dead_code)]
fn set_data_to_context(context: &mut GraphExecutionContext, data: Vec<u8>) -> Result<(), Error> {
    context.set_input(0, TensorType::U8, &[data.len()], &data)
}

fn get_data_from_context(context: &GraphExecutionContext, index: usize) -> Vec<u8> {
    // Use 1MB as the maximum output buffer size for audio output.
    const MAX_OUTPUT_BUFFER_SIZE: usize = 1024 * 1024;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = context
        .get_output(index, &mut output_buffer)
        .expect("Failed to get output");
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    output_buffer[..output_size].to_vec()
}

fn get_metadata_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&String::from_utf8_lossy(&get_data_from_context(context, 1)).to_string())
        .expect("Failed to get metadata")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <nn-preload-model> <prompt>", args[0]);
        return;
    }
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

    // If there is a third argument, use it as the prompt and enter non-interactive mode.
    // This is mainly for the CI workflow.
    let prompt = &args[2];
    // Set the prompt.
    println!("Prompt:\n{}", prompt);
    let tensor_data = prompt.as_bytes().to_vec();
    context
        .set_input(0, TensorType::U8, &[tensor_data.len()], &tensor_data)
        .expect("Failed to set input");

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

    context.compute().expect("Failed to compute");
    println!(
        "[INFO] Plugin writes output to file {}",
        options["tts-output-file"]
    );

    // Write output buffer to file, should be the same as the output file in the options.
    let output_filename = "output-buffer.wav";
    let output_bytes = get_data_from_context(&context, 0);
    let mut output_file = File::create(output_filename).expect("Failed to create output file");
    output_file
        .write_all(&output_bytes)
        .expect("Failed to write output file");
    println!("[INFO] Write output buffer to file {}", output_filename);

    return;
}
