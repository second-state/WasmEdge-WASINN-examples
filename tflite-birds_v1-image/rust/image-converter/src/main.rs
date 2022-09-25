use image::io::Reader;
use image::DynamicImage;
use std::env;
use std::fs;

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
    let pixels = Reader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img = dyn_img.to_rgb8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    let u8_f32_arr = raw_u8_arr.to_vec();
    return u8_f32_arr;
}
