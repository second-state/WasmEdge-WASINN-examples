use std::env;
use wasi_nn;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let prompt: &str = &args[2];

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::CPU)
            .build_from_cache(model_name)
            .unwrap();
    println!("Loaded model into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    println!("Created wasi-nn execution context with ID: {:?}", context);

    let tensor_data = prompt.as_bytes().to_vec();
    println!("Read input tensor, size in bytes: {}", tensor_data.len());
    context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .unwrap();

    // Execute the inference.
    context.compute().unwrap();
    println!("Executed model inference");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; 1000];
    context.get_output(0, &mut output_buffer).unwrap();
    let output = String::from_utf8(output_buffer.clone()).unwrap();
    println!("Output: {}", output);
}
