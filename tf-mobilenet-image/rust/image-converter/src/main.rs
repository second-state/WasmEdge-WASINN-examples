use std::env;
use std::fs;
use std::fs::File;
use std::io::Read;

fn main() {
    let args: Vec<String> = env::args().collect();
    let image_name: &str = &args[1];
    let output_name: &str = &args[2];

    let tensor_data = image_to_tensor(image_name.to_string(), 224, 224);
    println!(
        "Read image with {} bytes to write {}",
        tensor_data.len(),
        output_name
    );
    fs::write(output_name.to_string(), tensor_data).expect("Failed to write tensor")
}

fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&img, height, width, ::image::imageops::FilterType::Triangle);
    let mut flat_img: Vec<f32> = Vec::new();
    for rgb in resized.pixels() {
        flat_img.push(rgb[0] as f32 / 255.0);
        flat_img.push(rgb[1] as f32 / 255.0);
        flat_img.push(rgb[2] as f32 / 255.0);
    }
    let bytes_required = flat_img.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for c in 0..3 {
        for i in 0..(flat_img.len() / 3) {
            // Read the number as a f32 and break it into u8 bytes
            let u8_f32: f32 = flat_img[i * 3 + c] as f32;
            let u8_bytes = u8_f32.to_ne_bytes();

            for j in 0..4 {
                u8_f32_arr[((i * 3 + c) * 4) + j] = u8_bytes[j];
            }
        }
    }
    return u8_f32_arr;
}
