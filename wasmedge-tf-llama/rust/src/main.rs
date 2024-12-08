use std::env;
use std::fs::File;
use std::io::Read;

use wasi_nn;
use bytemuck::{bytes_of, bytes_of_mut, cast_slice, cast_slice_mut};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_file>", args[0]);
        std::process::exit(1);
    }
    let model_file = &args[1];

    // Read the TFLite model from file
    let mut file_mod = File::open(model_file).unwrap();
    let mut mod_buf = Vec::new();
    file_mod.read_to_end(&mut mod_buf).unwrap();

    // Create model segments (just one segment here)
    let model_segments = &[&mod_buf[..]];

    // Load the model
    let graph = match unsafe {
        wasi_nn::load(
            model_segments,
            wasi_nn::GraphEncoding::TensorflowLite,
            wasi_nn::ExecutionTarget::CPU,
        )
    } {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load graph: {:?}", e);
            return;
        }
    };

    // Create an execution context
    let ctx = match unsafe { wasi_nn::init_execution_context(graph) } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to init execution context: {:?}", e);
            return;
        }
    };

    // Example input data:
    // tokens: shape [1, 10], i32
    let tokens = [1i32, 2, 3, 4, 0, 0, 0, 0, 0, 0];
    let tokens_dims = &[1u32, 10u32];
    let tokens_tensor = wasi_nn::Tensor {
        dimensions: tokens_dims,
        r#type: wasi_nn::TensorType::I32,
        // cast_slice converts &[i32] -> &[u8] without copying
        data: cast_slice(&tokens),
    };

    // input_pos: shape [10], i32
    let input_pos: Vec<i32> = (0..10).collect();
    let input_pos_dims = &[10u32];
    let input_pos_tensor = wasi_nn::Tensor {
        dimensions: input_pos_dims,
        r#type: wasi_nn::TensorType::I32,
        data: cast_slice(&input_pos),
    };

    // kv: This depends on your model. For demonstration:
    // Let's say for LLaMA 1B (example), KV shape might be: [20, 2, 1, 16, 10, 64]
    // dtype is often fp32. Let's compute the size:
    // 20 * 2 * 1 * 16 * 10 * 64 = 20 * 2 * 16 * 10 * 64 = a large number
    // We'll guess a shape:
    let kv_dims = &[20u32, 2u32, 1u32, 16u32, 10u32, 64u32];
    let kv_size = 20 * 2 * 1 * 16 * 10 * 64;
    let kv_data = vec![0f32; kv_size];
    let kv_tensor = wasi_nn::Tensor {
        dimensions: kv_dims,
        r#type: wasi_nn::TensorType::F32,
        data: cast_slice(&kv_data),
    };

    // Set inputs
    if let Err(e) = unsafe { wasi_nn::set_input(ctx, 0, tokens_tensor) } {
        eprintln!("Failed to set tokens input: {:?}", e);
        return;
    }

    if let Err(e) = unsafe { wasi_nn::set_input(ctx, 1, input_pos_tensor) } {
        eprintln!("Failed to set input_pos input: {:?}", e);
        return;
    }

    if let Err(e) = unsafe { wasi_nn::set_input(ctx, 2, kv_tensor) } {
        eprintln!("Failed to set kv input: {:?}", e);
        return;
    }

    // Run inference
    if let Err(e) = unsafe { wasi_nn::compute(ctx) } {
        eprintln!("Failed to compute: {:?}", e);
        return;
    }

    // Get output:
    // You need to know the output size and type. Let's say the output is logits: shape [1, vocab_size]
    // Suppose vocab_size = 32000 for LLaMA. Then output buffer = 32000 floats
    let output_size = 32000;
    let mut output_buffer = vec![0f32; output_size];
    let output_bytes = cast_slice_mut(&mut output_buffer);

    // get_output expects (ctx, index, buffer, length)
    // length in bytes of the buffer:
    let byte_len = output_bytes.len() as u32;

    if let Err(e) = unsafe { wasi_nn::get_output(ctx, 0, output_bytes.as_mut_ptr(), byte_len) } {
        eprintln!("Failed to get output: {:?}", e);
        return;
    }

    // Now output_buffer should contain your output logits
    println!("Model output (first 10 logits): {:?}", &output_buffer[..10]);
}