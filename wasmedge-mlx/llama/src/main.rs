use serde_json::json;
use std::env;
use wasmedge_wasi_nn::{
    self, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext, TensorType,
};
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
    let tokenizer_path = "./tokenizer.json";
    let prompt = "Once upon a time, there existed a little girl,";
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let graph = GraphBuilder::new(GraphEncoding::Mlx, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&json!({"is_quantized":false, "group_size": 64, "q_bits": 4,"model_type": "tiny_llama_1.1B_chat_v1.0", "tokenizer":tokenizer_path, "max_token":100})).expect("Failed to serialize options"))
        .build_from_cache(model_name)
        .expect("Failed to build graph");
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");
    let tensor_data = prompt.as_bytes().to_vec();
    context
        .set_input(0, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");
    context.compute().expect("Failed to compute");
    let output = get_output_from_context(&context);

    println!("{}", output.trim());
}
