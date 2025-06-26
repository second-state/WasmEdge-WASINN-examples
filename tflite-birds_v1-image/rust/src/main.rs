use image::io::Reader;
use image::DynamicImage;
use std::env;
use std::error::Error;
use std::fs;
use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding, TensorType};
mod imagenet_classes;

pub fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    let image_name: &str = &args[2];

    let weights = fs::read(model_bin_name)?;
    println!("Read graph weights, size in bytes: {}", weights.len());

    let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU)
        .build_from_bytes(&[&weights])?;
    let mut ctx = graph.init_execution_context()?;
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    // Load a tensor that precisely matches the graph input tensor (see
    let tensor_data = image_to_tensor(image_name.to_string(), 224, 224);
    println!("Read input tensor, size in bytes: {}", tensor_data.len());

    // Pass tensor data into the TFLite runtime
    ctx.set_input(0, TensorType::U8, &[1, 224, 224, 3], &tensor_data)?;

    // Execute the inference.
    ctx.compute()?;

    // Retrieve the output.
    let mut output_buffer = vec![0u8; imagenet_classes::AIY_BIRDS_V1.len()];
    _ = ctx.get_output(0, &mut output_buffer)?;

    // Sort the result with the highest probability result first
    let results = sort_results(&output_buffer);
    for i in 0..5 {
        println!(
            "   {}.) [{}]({:.4}){}",
            i + 1,
            results[i].0,
            results[i].1,
            imagenet_classes::AIY_BIRDS_V1[results[i].0]
        );
    }

    Ok(())
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[u8]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// Take the image located at 'path', open it, resize it to height x width. The resulting RGB
// pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = Reader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img = dyn_img.to_rgb8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    return raw_u8_arr.to_vec();
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, u8);
