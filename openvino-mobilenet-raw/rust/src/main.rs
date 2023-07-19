use std::env;
use std::fs;
use wasi_nn;
mod imagenet_classes;

use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding, TensorType};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_xml_path: &str = &args[1];
    let model_bin_path: &str = &args[2];
    let tensor_name: &str = &args[3];

    print!("Load graph ...");
    let graph = GraphBuilder::new(GraphEncoding::Openvino, ExecutionTarget::CPU)
        .build_from_files([model_xml_path, model_bin_path])?;
    println!("done");

    print!("Init execution context ...");
    let mut context = graph.init_execution_context()?;
    println!("done");

    // Load a tensor that precisely matches the graph input tensor (see
    // `fixture/frozen_inference_graph.xml`).
    print!("Set input tensor ...");
    let input_dims = vec![1, 3, 224, 224];
    let tensor_data = fs::read(tensor_name).unwrap();
    context.set_input(0, TensorType::F32, &input_dims, tensor_data)?;
    println!("done");

    print!("Perform graph inference ...");
    context.compute()?;
    println!("done");

    print!("Retrieve the output ...");
    // Copy output to abuffer.
    let mut output_buffer = vec![0f32; 1001];
    let size_in_bytes = context.get_output(0, &mut output_buffer)?;
    println!("done");
    println!("The size of the output buffer is {} bytes", size_in_bytes);

    let results = sort_results(&output_buffer);
    for i in 0..5 {
        println!(
            "   {}.) [{}]({:.4}){}",
            i + 1,
            results[i].0,
            results[i].1,
            imagenet_classes::IMAGENET_CLASSES[results[i].0]
        );
    }

    Ok(())
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
