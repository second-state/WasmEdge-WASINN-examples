use rust_processor::auto::processing_auto::AutoProcessor;
use rust_processor::gemma3::detokenizer::decode;
use rust_processor::processor_utils::prepare_inputs;
use rust_processor::NDTensorI32;
use std::env;
use wasmedge_wasi_nn::{
    self, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext, TensorType,
};

use serde_json::Value;
use std::fs::File;
use std::io::{self, BufReader};

fn get_data_from_context(context: &GraphExecutionContext, index: usize) -> NDTensorI32 {
    // Preserve for 4096 tokens with average token length 6
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 6;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = context
        .get_output(index, &mut output_buffer)
        .expect("Failed to get output");
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    return NDTensorI32::from_bytes(&output_buffer[..output_size]).unwrap();
}

fn get_output_from_context(context: &GraphExecutionContext) -> NDTensorI32 {
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
    // prompt: "What is this icon?";
    // image: "wasmedge-runtime-logo.png";
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let model_dir = &args[2];
    let config = read_json(&format!("{}/config.json", model_dir)).unwrap();
    let prompt = "<bos><start_of_turn>user\
    What is this icon?<start_of_image><end_of_turn>\
    <start_of_turn>model";
    let image_path = "wasmedge-runtime-logo.png";
    println!("create processor: {}", model_dir);
    let mut processor = match AutoProcessor::from_pretrained(model_dir) {
        Ok(processor) => match processor {
            rust_processor::auto::processing_auto::AutoProcessorType::Gemma3(processor) => {
                processor
            }
            _ => {
                eprintln!("Error loading processor: not a Gemma3Processor");
                return;
            }
        },
        Err(e) => {
            eprintln!("Error loading processor: {}", e);
            return;
        }
    };
    println!("processor created");
    let image_token_index = config["image_token_index"].as_u64().unwrap_or(262144) as u32;
    let model_inputs = prepare_inputs(
        &mut processor,
        &[image_path], // Use single image array
        prompt,
        image_token_index,
        Some((896, 896)), // Use 896x896 as image size
    );
    let graph = GraphBuilder::new(GraphEncoding::Mlx, ExecutionTarget::AUTO)
        .config(config.to_string())
        .build_from_cache(model_name)
        .expect("Failed to build graph");

    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");

    let tensor_data = model_inputs["input_ids"].to_bytes();
    context
        .set_input(0, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");
    let tensor_data = model_inputs["pixel_values"].to_bytes();
    context
        .set_input(1, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");
    let tensor_data = model_inputs["mask"].to_bytes();
    context
        .set_input(2, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");

    context.compute().expect("Failed to compute");
    let tokens = get_output_from_context(&context);
    let output = decode(
        &tokens
            .data
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>(),
        &processor,
        true,
    );
    println!("{}", output.trim());
}
