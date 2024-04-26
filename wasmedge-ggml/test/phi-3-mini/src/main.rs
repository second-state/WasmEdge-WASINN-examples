use serde_json::json;
use serde_json::Value;
use std::env;
use wasmedge_wasi_nn::{
    self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};

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

    options
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];

    // Set options for the graph. Check our README for more details:
    // https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#parameters
    let options = get_options_from_env();

    // This is mainly for the CI workflow. Only support the prompt from the command line.
    if args.len() < 3 {
        println!("Usage: {} <model_name> <prompt>", args[0]);
        std::process::exit(1);
    }
    let prompt = &args[2];

    // Create graph and initialize context.
    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&options).expect("Failed to serialize options"))
        .build_from_cache(model_name)
        .expect("Failed to build graph");
    println!("Graph {} loaded.", graph);
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");

    // Set the prompt.
    println!("Prompt:\n{}", prompt);
    let tensor_data = prompt.as_bytes().to_vec();
    context
        .set_input(0, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");
    println!("Response:");

    // Execute the inference.
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
            std::process::exit(1);
        }
    }

    // Retrieve the output.
    let output = get_output_from_context(&context);
    println!("{}", output.trim());

    // Unload.
    graph.unload().expect("Failed to unload graph");
}
