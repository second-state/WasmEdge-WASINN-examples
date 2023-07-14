use std::env;
use std::fs;
use wasi_nn;
mod imagenet_classes;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let model_xml_name: &str = &args[1];
    let model_bin_name: &str = &args[2];
    let tensor_name: &str = &args[3];

    let xml = fs::read_to_string(model_xml_name).unwrap();
    println!("Read graph XML, size in bytes: {}", xml.len());

    let weights = fs::read(model_bin_name).unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let graph = wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Openvino,
        wasi_nn::ExecutionTarget::CPU,
    )
    .build_from_bytes(&[xml.into_bytes(), weights])
    .unwrap();
    println!("Loaded graph into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    println!("Created wasi-nn execution context with ID: {:?}", context);

    // Load a tensor that precisely matches the graph input tensor (see
    // `fixture/frozen_inference_graph.xml`).
    let tensor_data = fs::read(tensor_name).unwrap();
    println!("Read input tensor, size in bytes: {}", tensor_data.len());
    // for i in 0..10{
    //     println!("tensor -> {}", tensor_data[i]);
    // }
    context
        .set_input(0, wasi_nn::TensorType::F32, &[1, 3, 224, 224], &tensor_data)
        .unwrap();
    // Execute the inference.
    context.compute().unwrap();
    println!("Executed graph inference");
    // Retrieve the output.
    let mut output_buffer = vec![0f32; 1001];
    context.get_output(0, &mut output_buffer).unwrap();

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
    let ground_truth_result = [963, 762, 909, 926, 567];
    // let ground_truth_pred = [0.7113048, 0.0707076, 0.036355935, 0.015456136, 0.015344063];
    for i in 0..ground_truth_result.len() {
        assert_eq!(results[i].0, ground_truth_result[i]);
    }
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
