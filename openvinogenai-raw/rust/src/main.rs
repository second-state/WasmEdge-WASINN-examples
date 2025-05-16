use std::env;
use std::fs;
use std::io::{self, Write};
use wasmedge_wasi_nn;
use wasmedge_wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding, TensorType};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_type: &str = "LLMPipeline";
    let model_xml_path: &str = &args[1];
    let plugin_config: &str = "";

    print!("Load graph ...");
    let graph = GraphBuilder::new(GraphEncoding::OpenvinoGenAI, ExecutionTarget::CPU)
        .build_from_bytes([model_type, model_xml_path, plugin_config])?;
    println!("done");

    print!("Init execution context ...");
    let mut context = graph.init_execution_context()?;
    println!("done");

    print!("Set input tensor ...");
    let input_dims = vec![1];
    let tensor_data = "Hello, how are you?".as_bytes().to_vec();
    context.set_input(0, TensorType::U8, &input_dims, tensor_data)?;
    println!("done");

    print!("Generating ...");
    context.compute()?;
    println!("done");

    print!("Get the result ...");
    print!("Retrieve the output ...");
    // Copy output to abuffer.
    let mut output_buffer = vec![0u8; 1001];
    let size_in_bytes = context.get_output(0, &mut output_buffer)?;
    println!("done");
    println!("The size of the output buffer is {} bytes", size_in_bytes);

    let string_output = String::from_utf8(output_buffer.iter().map(|&c| c as u8).collect()).unwrap();
    println!("Output: {}", string_output);
    println!("done");
    io::stdout().flush().unwrap();

    Ok(())
}
