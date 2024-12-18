use std::env;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::process;
use std::io::Read;
use thiserror::Error;
use wasi_nn;
use bytemuck::{cast_slice, cast_slice_mut};
use std::collections::HashMap;

// Define a custom error type to handle multiple error sources
#[derive(Error, Debug)]
pub enum ChatbotError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WASI NN Error: {0}")]
    WasiNn(String),

    #[error("Other Error: {0}")]
    Other(String),
}

impl From<wasi_nn::Error> for ChatbotError {
    fn from(error: wasi_nn::Error) -> Self {
        ChatbotError::WasiNn(error.to_string())
    }
}

type Result<T> = std::result::Result<T, ChatbotError>;

// Simple tokenizer with a fixed vocabulary
fn tokenize(input: &str) -> Vec<i32> {
    let vocab: HashMap<&str, i32> = [
        ("hello", 1),
        ("world", 2),
        ("this", 3),
        ("is", 4),
        ("a", 5),
        ("test", 6),
        // Add more tokens as needed
    ]
    .iter()
    .cloned()
    .collect();

    input
        .split_whitespace()
        .map(|word| *vocab.get(word).unwrap_or(&0)) // Use 0 for unknown tokens
        .collect()
}

// Simple detokenizer with a fixed vocabulary
fn detokenize(tokens: &[f32]) -> String {
    let vocab_reverse: HashMap<i32, &str> = [
        (1, "hello"),
        (2, "world"),
        (3, "this"),
        (4, "is"),
        (5, "a"),
        (6, "test"),
        // Add more tokens as needed
    ]
    .iter()
    .cloned()
    .collect();

    tokens
        .iter()
        .map(|&token| *vocab_reverse.get(&(token as i32)).unwrap_or(&""))
        .collect::<Vec<&str>>()
        .join(" ")
}

// Function to load the TFLite model
fn load_model(model_path: &str) -> Result<wasi_nn::Graph> {
    let mut file = File::open(model_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let model_segments = &[&buffer[..]];
    Ok(unsafe { wasi_nn::load(model_segments, 4, wasi_nn::EXECUTION_TARGET_CPU)? })
}

// Function to initialize the execution context
fn init_context(graph: wasi_nn::Graph) -> Result<wasi_nn::GraphExecutionContext> {
    Ok(unsafe { wasi_nn::init_execution_context(graph)? })
}

fn main() -> Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_file>", args[0]);
        process::exit(1);
    }
    let model_file = &args[1];

    // Load the model
    let graph = load_model(model_file)?;

    // Initialize execution context
    let ctx = init_context(graph)?;

    let mut stdout = io::stdout();
    let stdin = io::stdin();

    // Initialize KV cache data
    let mut kv_data = vec![0f32; 20 * 2 * 1 * 16 * 10 * 64]; // Adjust based on your model's requirements

    println!("Chatbot is ready! Type your messages below:");

    for line in stdin.lock().lines() {
        let user_input = line?;
        if user_input.trim().is_empty() {
            continue;
        }
        if user_input.to_lowercase() == "exit" {
            break;
        }

        // Tokenize input
        let tokens = tokenize(&user_input);
        let tokens_dims = &[1u32, tokens.len() as u32];
        let tokens_tensor = wasi_nn::Tensor {
            dimensions: tokens_dims,
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: cast_slice(&tokens),
        };

        // Create input_pos tensor
        let input_pos: Vec<i32> = (0..tokens.len() as i32).collect();
        let input_pos_dims = &[input_pos.len() as u32];
        let input_pos_tensor = wasi_nn::Tensor {
            dimensions: input_pos_dims,
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: cast_slice(&input_pos),
        };

        // Create kv tensor
        let kv_dims = &[20u32, 2u32, 1u32, 16u32, 10u32, 64u32];
        let kv_tensor = wasi_nn::Tensor {
            dimensions: kv_dims,
            r#type: wasi_nn::TENSOR_TYPE_F32,
            data: cast_slice(&kv_data),
        };

        // Set inputs
        unsafe {
            wasi_nn::set_input(ctx, 0, tokens_tensor)?;
            wasi_nn::set_input(ctx, 1, input_pos_tensor)?;
            wasi_nn::set_input(ctx, 2, kv_tensor)?;
        }

        // Run inference
        run_inference(&ctx)?;

        // Get output
        let output = get_model_output(&ctx, 0, 32000)?;

        // Detokenize output
        let response = detokenize(&output);

        // Display response
        writeln!(stdout, "Bot: {}", response)?;
    }

    println!("Chatbot session ended.");
    Ok(())
}

// Function to run inference
fn run_inference(ctx: &wasi_nn::GraphExecutionContext) -> Result<()> {
    unsafe { Ok(wasi_nn::compute(*ctx)?) }
}

// Function to get model output
fn get_model_output(
    ctx: &wasi_nn::GraphExecutionContext,
    index: u32,
    size: usize,
) -> Result<Vec<f32>> {
    let mut buffer = vec![0f32; size];
    let buffer_ptr = cast_slice_mut(&mut buffer).as_mut_ptr();
    let byte_len = (size * std::mem::size_of::<f32>()) as u32;
    unsafe { wasi_nn::get_output(*ctx, index, buffer_ptr, byte_len)? };
    Ok(buffer)
}
