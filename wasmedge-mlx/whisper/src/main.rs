use std::env;
use wasmedge_wasi_nn::{
    self, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext, TensorType,
};

use serde_json::Value;
use std::fs::File;
use std::io::{self, BufReader};

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

fn read_json(path: &str) -> io::Result<Value> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let v = serde_json::from_reader(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(v)
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let model_dir = &args[2];
    let audio = &args[3];
    let config = read_json(&format!("{}/config.json", model_dir)).unwrap();
    let graph = GraphBuilder::new(GraphEncoding::Mlx, ExecutionTarget::AUTO)
        .config(config.to_string())
        .build_from_cache(model_name)
        .expect("Failed to build graph");
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");
    let tensor_data = audio.as_bytes().to_vec();
    context
        .set_input(0, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");
    context.compute().expect("Failed to compute");
    let output = get_output_from_context(&context);

    println!("{}", output.trim());
}
