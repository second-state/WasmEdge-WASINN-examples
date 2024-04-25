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
        let tokenizer_path = "neural-chat-tokenizer.json";
        let prompt = "Once upon a time, there existed a little girl,";
        let args: Vec<String> = env::args().collect();
        let model_name: &str = &args[1];
        let tokenizer:Tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let inputs = encoding.get_ids();
        let mut tensor_data: Vec<u8> = Vec::with_capacity(inputs.len() * 8);

        for &val in inputs {
            let mut bytes = u64::from(val).to_be_bytes();
            bytes.reverse();
            tensor_data.extend_from_slice(&bytes);
        }
        let graph = GraphBuilder::new(GraphEncoding::NeuralSpeed, ExecutionTarget::AUTO)
        .config(serde_json::to_string(&json!({"model_type": "mistral"})).expect("Failed to serialize options"))
        .build_from_cache(model_name)
        .expect("Failed to build graph");
        let mut context = graph
            .init_execution_context()
            .expect("Failed to init context");
        context
            .set_input(0, TensorType::U8, &[1], &tensor_data)
            .expect("Failed to set input");
        context.compute().expect("Failed to compute");
        let output_bytes = get_output_from_context(&context);
        let output_id:Vec<u32> = output_bytes
        .chunks(8)
        .map(|chunk| {
            chunk
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &byte)| acc + ((byte as u64) << (i * 8))) as u32
        })
        .collect();
        let output = tokenizer.decode(&output_id, true).unwrap();
        println!("{}", output);
        context.fini_single().expect("Failed to free resource");

}