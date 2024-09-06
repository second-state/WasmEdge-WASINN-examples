use tokenizers::tokenizer::Tokenizer;
use serde_json::json;
use wasmedge_wasi_nn::{
    self, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};
use std::env;
fn get_data_from_context(context: &GraphExecutionContext, index: usize) -> Vec<u8> {
    // Preserve for 4096 tokens with average token length 8
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 8;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let _ = context
        .get_output(index, &mut output_buffer)
        .expect("Failed to get output");

    return output_buffer;
}

fn get_output_from_context(context: &GraphExecutionContext) -> Vec<u8> {
    get_data_from_context(context, 0)
}
fn main() {
        let tokenizer_path = "tokenizer.json";
        let prompt = "Once upon a time, there existed a little girl,";

        let graph = GraphBuilder::new(GraphEncoding::Mlx, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&json!({"tokenizer":tokenizer_path})).expect("Failed to serialize options"))
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
        let output_bytes = get_output_from_context(&context);

        println!("{}", output.trim());

}