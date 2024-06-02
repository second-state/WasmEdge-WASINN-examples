use image;
use std::fs::File;
use std::io::Read;
use squeezenet_burn::model::label::LABELS;

pub fn main() {
    let img_path = std::env::args().nth(1).expect("No image path provided");
    let model_name = std::env::args().nth(2).expect("No model name provided");

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Burn, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(&model_name)
            .expect("Failed to build graph");

    println!("Loaded graph into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    println!("Created wasi-nn execution context with ID: {:?}", context);

    let tensor_data = image_to_tensor(img_path, 224, 224);
    context
        .set_input(0, wasi_nn::TensorType::F32, &[1, 3, 224, 224], &tensor_data)
        .unwrap();

    context.compute().unwrap();
    println!("Executed graph inference");

    let mut output_buffer = vec![0f32; 1000];
    context.get_output(0, &mut output_buffer).unwrap();

    top_5_classes(output_buffer)
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<f32> {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();
    // Resize the image
    let resized =
        image::imageops::resize(&img, height, width, ::image::imageops::FilterType::Triangle);
    let mut flat_img: Vec<f32> = Vec::new();
    // Normize the image and rearrange the tensor data order
    for rgb in resized.pixels() {
        flat_img.push((rgb[0] as f32 / 255. - 0.485) / 0.229);
    }
    for rgb in resized.pixels() {
        flat_img.push((rgb[1] as f32 / 255. - 0.456) / 0.224);
    }
    for rgb in resized.pixels() {
        flat_img.push((rgb[2] as f32 / 255. - 0.406) / 0.225);
    }
    return flat_img;
}

#[derive(Debug)]
pub struct InferenceResult {
    index: usize,
    probability: f32,
    label: String,
}

fn top_5_classes(probabilities: Vec<f32>) {
    // Convert the probabilities into a vector of (index, probability)
    let mut probabilities: Vec<_> = probabilities.iter().enumerate().collect();

    // Sort the probabilities in descending order
    probabilities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    // Take the top 5 probabilities
    probabilities.truncate(5);

    // Convert the probabilities into InferenceResult
    let result: Vec<InferenceResult> = probabilities
        .into_iter()
        .map(|(index, probability)| InferenceResult {
            index,
            probability: *probability,
            label: LABELS[index].to_string(),
        })
        .collect();
    println!("Top 5 classes: {:?}", result);
}
