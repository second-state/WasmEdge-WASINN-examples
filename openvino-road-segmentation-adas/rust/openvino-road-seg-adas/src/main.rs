use image::{io::Reader, DynamicImage};
use std::env;
use wasmedge_nn::nn::{ctx::WasiNnCtx, Dtype, ExecutionTarget, GraphEncoding, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_xml_name: &str = &args[1];
    let model_bin_name: &str = &args[2];
    let image_name: &str = &args[3];

    // load image
    let tensor_data = image_to_tensor(image_name.to_string(), 512, 896);
    println!("Load input tensor, size in bytes: {}", tensor_data.len());
    let tensor = Tensor {
        dimensions: &[1, 3, 512, 896],
        r#type: Dtype::F32.into(),
        data: &tensor_data,
    };

    let mut ctx = WasiNnCtx::new()?;

    println!("Load model files ...");
    let graph_id = ctx.load(
        model_xml_name,
        model_bin_name,
        GraphEncoding::Openvino,
        ExecutionTarget::CPU,
    )?;

    println!("initialize the execution context ...");
    let exec_context_id = ctx.init_execution_context(graph_id)?;

    println!("Set input tensor ...");
    ctx.set_input(exec_context_id, 0, tensor)?;

    println!("Do inference ...");
    ctx.compute(exec_context_id)?;

    println!("Extract result ...");
    let mut out_buffer = vec![0u8; 1 * 4 * 512 * 896 * 4];
    ctx.get_output(exec_context_id, 0, out_buffer.as_mut_slice())?;

    println!("Dump result ...");
    dump(
        "wasinn-openvino-inference-output-1x4x512x896xf32.tensor",
        out_buffer.as_slice(),
    )?;

    Ok(())
}

/// Dump data to the specified binary file.
fn dump<T>(
    path: impl AsRef<std::path::Path>,
    buffer: impl AsRef<[T]>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\tdump tensor to {:?}", path.as_ref());

    let dst_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(
            buffer.as_ref().as_ptr() as *const u8,
            buffer.as_ref().len() * std::mem::size_of::<T>(),
        )
    };
    println!("\tThe size of bytes: {}", dst_slice.len());

    std::fs::write(path, &dst_slice).expect("failed to write file");

    Ok(())
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = Reader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img = dyn_img.to_bgr8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    // Create an array to hold the f32 value of those pixels
    let bytes_required = raw_u8_arr.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for i in 0..raw_u8_arr.len() {
        // Read the number as a f32 and break it into u8 bytes
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let u8_bytes = u8_f32.to_ne_bytes();

        for j in 0..4 {
            u8_f32_arr[(i * 4) + j] = u8_bytes[j];
        }
    }
    return u8_f32_arr;
}
