use std::env;
use std::fs;
use std::error::Error;
use wasmedge_wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget, TensorType};

pub fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    let wav_name: &str = &args[2];

    let model_bin = fs::read(model_bin_name)?;
    println!("Read model, size in bytes: {}", model_bin.len());

    let graph = GraphBuilder::new(GraphEncoding::Whisper, ExecutionTarget::CPU).build_from_bytes(&[&model_bin])?;
    let mut ctx = graph.init_execution_context()?;
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    // Load the raw pcm tensor.
    let wav_buf = fs::read(wav_name)?;
    println!("Read input tensor, size in bytes: {}", wav_buf.len());

    // Set input.
    ctx.set_input(0, TensorType::F32, &[1, wav_buf.len()], &wav_buf)?;

    // Execute the inference.
    ctx.compute()?;

    // Retrieve the output.
    let mut output_buffer = vec![0u8; 2048];
    _ = ctx.get_output(0, &mut output_buffer)?;

    println!("Recognized from audio: \n{}", String::from_utf8(output_buffer).unwrap());
    
    Ok(())
}
