use image::{self, GenericImage, RgbImage};
use std::env;
use std::fs::File;
use std::io::Read;
use wasi_nn;

use crate::yolo_classes::YOLO_CLASSES;

mod yolo_classes;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    let image_name: &str = &args[2];
    println!("model_bin_name {}", model_bin_name);
    println!("image_name {}", image_name);

    let graph = wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Pytorch,
        wasi_nn::ExecutionTarget::CPU,
    )
    .build_from_files([model_bin_name])
    .unwrap();

    println!("Loaded graph into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    println!("Created wasi-nn execution context with ID: {:?}", context);

    // Load a tensor that precisely matches the graph input tensor dimensions
    // Graph expects 1,3,640,640 dimensional tensor

    let tensor_data = pre_process_image(image_name.to_string())
        .into_iter()
        .flatten()
        .collect::<Vec<Vec<f32>>>()
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();

    println!("Read input tensor, size in bytes: {}", tensor_data.len());

    context
        .set_input(
            0,
            wasi_nn::TensorType::F32,
            // Input
            &[1, 3, SIZE, SIZE],
            &tensor_data,
        )
        .unwrap();

    // // Execute the inference.
    context.compute().unwrap();
    println!("Executed graph inference");

    // Retrieve the output.
    // YOLO's output Tensor has a dimension of 1*84*8400
    // 1 : Batch Size
    // 84 : x,y,w,h, P1, P2, P3, P4,
    // 8400 : number of total objects YoloV8 can detect per frame

    let mut output_buffer = vec![0f32; 1 * 84 * 8400];
    context.get_output(0, &mut output_buffer).unwrap();

    let results = post_process_results(&output_buffer);
    for result in results {
        if result.probability > 0.5 {
            println!("{:?}", result);
        }
    }
}

// Function to normalize and resize image for YOLO
const SIZE: usize = 640;
const SIZE_U32: u32 = 640;
type Channel = Vec<Vec<f32>>;

fn pre_process_image(path: String) -> [Channel; 3] {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();

    let (height, width);
    if img.width() > img.height() {
        // height is the shorter length
        height = SIZE_U32 * img.height() / img.width();
        width = SIZE_U32;
    } else {
        // width is the shorter length
        width = SIZE_U32 * img.width() / img.height();
        height = SIZE_U32;
    }

    let resized =
        image::imageops::resize(&img, width, height, ::image::imageops::FilterType::Triangle);

    // We need the image to fit the 640 x 640 size,
    // and we want to keep the aspect ratio of the original image
    // So we fill the remaining pixels with black,
    let mut resized_640x640 = RgbImage::new(SIZE_U32, SIZE_U32);
    resized_640x640.copy_from(&resized, 0, 0).unwrap();

    // Split intoChannels
    let mut red: Channel = vec![vec![0.0; SIZE]; SIZE];
    let mut blue: Channel = vec![vec![0.0; SIZE]; SIZE];
    let mut green: Channel = vec![vec![0.0; SIZE]; SIZE];

    for (_, pixel) in resized_640x640.enumerate_rows() {
        for (x, y, rgb) in pixel {
            let x = x as usize;
            let y = y as usize;

            red[y][x] = rgb.0[0] as f32 / 255.0;
            green[y][x] = rgb.0[1] as f32 / 255.0;
            blue[y][x] = rgb.0[2] as f32 / 255.0;
        }
    }

    let final_tensor: [Vec<Vec<f32>>; 3];
    final_tensor = [red, green, blue];

    final_tensor
}

// Function to process output tensor from YOLO
fn post_process_results(buffer: &[f32]) -> Vec<InferenceResult> {
    // Output buffer is in columar format
    // 84 rows x 8400 columns as a single Vec of f32
    let mut columns = Vec::new();
    for col_slice in buffer.chunks_exact(8400) {
        let col_vec = col_slice.to_vec();
        columns.push(col_vec);
    }

    let rows = transpose(columns);

    // Row Format is
    // [x,y,w,h,p1,p2,p3...p80]
    // where:
    // x,y are the pixel locations of the top left corner of the bounding box,
    // w,h are the width and height of bounding box,
    // p1,p2..p80, are the class probabilities.
    let mut results = Vec::new();

    for row in rows {
        let x = row[0].round() as u32;
        let y = row[1].round() as u32;
        let width = row[2].round() as u32;
        let height = row[3].round() as u32;

        // Get maximum likeliehood for each detection
        // Iterator of only class probabilities
        let mut prob_iter = row.clone().into_iter().skip(4);
        let max = prob_iter.clone().reduce(|a, b| a.max(b)).unwrap();
        let index = prob_iter.position(|element| element == max).unwrap();
        let class = YOLO_CLASSES.get(index).unwrap().to_string();

        results.push(InferenceResult {
            x,
            y,
            width,
            height,
            probability: max,
            class,
        });
    }
    results
}

// Output data is in the form 84 Rows x 8400 columns
fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}
// Struct to return:
// Location Top left pixel of bounding box,
// Width + Height of bounding box
// Probability and Class Name
#[derive(Debug)]
struct InferenceResult {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    class: String,
    probability: f32,
}
