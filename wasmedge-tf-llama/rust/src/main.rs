#![feature(str_as_str)]
use bytemuck::{cast_slice, cast_slice_mut};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::io::{self, BufRead, Write};
use std::process;
use thiserror::Error;
use wasi_nn;

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

// Tokenizer Struct
struct Tokenizer {
    vocab: HashMap<String, i32>,
    vocab_reverse: HashMap<i32, String>,
    next_id: i32,
}

impl Tokenizer {
    fn new(initial_vocab: Vec<(&str, i32)>) -> Self {
        let mut vocab = HashMap::new();
        let mut vocab_reverse = HashMap::new();
        let mut next_id = 1;

        for (word, id) in initial_vocab {
            vocab.insert(word.to_string(), id);
            vocab_reverse.insert(id, word.to_string());
            if id >= next_id {
                next_id = id + 1;
            }
        }

        // Add special tokens
        vocab.insert("<UNK>".to_string(), 0);
        vocab_reverse.insert(0, "<UNK>".to_string());
        vocab.insert("<PAD>".to_string(), -1);
        vocab_reverse.insert(-1, "<PAD>".to_string());

        Tokenizer {
            vocab,
            vocab_reverse,
            next_id,
        }
    }

    fn tokenize(&mut self, input: &str) -> Vec<i32> {
        input
            .split_whitespace()
            .map(|word| {
                self.vocab.get(word).cloned().unwrap_or_else(|| {
                    let id = self.next_id;
                    self.vocab.insert(word.to_string(), id);
                    self.vocab_reverse.insert(id, word.to_string());
                    self.next_id += 1;
                    id
                })
            })
            .collect()
    }

    fn tokenize_with_fixed_length(&mut self, input: &str, max_length: usize) -> Vec<i32> {
        let mut tokens = self.tokenize(input);

        if tokens.len() > max_length {
            tokens.truncate(max_length);
        } else if tokens.len() < max_length {
            tokens.extend(vec![-1; max_length - tokens.len()]); // Assuming -1 is the <PAD> token
        }

        tokens
    }

    fn detokenize(&self, tokens: &[i32]) -> String {
        tokens
            .iter()
            .map(|&token| self.vocab_reverse.get(&token).map_or("<UNK>", |v| v).as_str())
            .collect::<Vec<&str>>()
            .join(" ")
    }
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
    println!("Chatbot is ready! Type your messages below:");

    for line in stdin.lock().lines() {
        let user_input = line?;
        if user_input.trim().is_empty() {
            continue;
        }
        if user_input.to_lowercase() == "exit" {
            break;
        }
        // Initialize tokenizer
        let initial_vocab = vec![
            ("hello", 1),
            ("world", 2),
            ("this", 3),
            ("is", 4),
            ("a", 5),
            ("test", 6),
            ("<PAD>", -1),
        ];

        let mut tokenizer = Tokenizer::new(initial_vocab);
        // let user_input = "hello world this is a test with more words";

        // Tokenize with fixed length
        let max_length = 655360;
        let tokens = tokenizer.tokenize_with_fixed_length(&user_input, max_length);
        let tokens_dims = &[1u32, max_length as u32];
        let tokens_tensor = wasi_nn::Tensor {
            dimensions: tokens_dims,
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: cast_slice(&tokens),
        };

        // Create input_pos tensor
        let input_pos: Vec<i32> = (0..max_length as i32).collect();
        let input_pos_dims = &[1u32, max_length as u32];
        let input_pos_tensor = wasi_nn::Tensor {
            dimensions: input_pos_dims,
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: cast_slice(&input_pos),
        };

        // Create kv tensor (ensure kv_data has the correct size)
        let kv_data = vec![0.0_f32; max_length]; // Example initialization
        let kv_dims = &[32u32, 2u32, 1u32, 16u32, 10u32, 64u32];
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
        let output = get_model_output(&ctx, 0, 655360)?;

        // Detokenize output
        let response = tokenizer.detokenize(&output.as_slice());

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
) -> Result<Vec<i32>> {
    let mut buffer = vec![0i32; size];
    let buffer_ptr = cast_slice_mut(&mut buffer).as_mut_ptr();
    let byte_len = (size * std::mem::size_of::<i32>()) as u32;
    unsafe { wasi_nn::get_output(*ctx, index, buffer_ptr, byte_len)? };
    Ok(buffer)
}
