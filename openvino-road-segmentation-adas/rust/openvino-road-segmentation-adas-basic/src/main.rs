//! Build wasi-nn wasm
//!
//! First, run `cargo build --target=wasm32-wasi --release`;
//!
//! Then, go to the `./target/wasm32-wasi/release` directory, you can find `rust-road-segmentation-adas.wasm` file.
//!
//! In the environment where WasmEdge is deployed, run `wasmedge --dir .:. <path to rust-road-segmentation-adas.wasm> ../model/road-segmentation-adas-0001.xml ../model/road-segmentation-adas-0001.bin <path to input tensor>`. After the inference, the result tensor `wasinn-openvino-inference-output-1x4x512x896xf32.tensor` will be returned. You can use opencv or other tools to visualize the inference result.

use std::{convert::TryInto, env, fs};
use wasi_nn as nn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_xml_name: &str = &args[1];
    let model_bin_name: &str = &args[2];
    let tensor_name: &str = &args[3];

    let xml = fs::read_to_string(model_xml_name).unwrap();
    let xml_bytes = xml.into_bytes();
    println!("Load graph XML, size in bytes: {}", xml_bytes.len());

    let weights = fs::read(model_bin_name).unwrap();
    println!("Load graph weights, size in bytes: {}", weights.len());

    let tensor_data = fs::read(tensor_name).unwrap();
    println!("Load input tensor, size in bytes: {}", tensor_data.len());
    let tensor = nn::Tensor {
        dimensions: &[1, 3, 512, 896],
        r#type: nn::TENSOR_TYPE_F32,
        data: &tensor_data,
    };

    // do inference
    let output_buffer = infer(xml_bytes.as_slice(), weights.as_slice(), tensor)?;

    // dump result
    dump(
        "wasinn-openvino-inference-output-1x4x512x896xf32.tensor",
        output_buffer.as_slice(),
    )?;

    Ok(())
}

/// Do inference
fn infer(
    xml_bytes: impl AsRef<[u8]>,
    weights: impl AsRef<[u8]>,
    in_tensor: nn::Tensor,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let graph = unsafe {
        wasi_nn::load(
            &[xml_bytes.as_ref(), weights.as_ref()],
            wasi_nn::GRAPH_ENCODING_OPENVINO,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
    println!("Created wasi-nn execution context with ID: {}", context);

    unsafe {
        wasi_nn::set_input(context, 0, in_tensor).unwrap();
    }
    // Execute the inference.
    unsafe {
        wasi_nn::compute(context).unwrap();
    }
    println!("Executed graph inference");

    // Retrieve the output.
    let mut output_buffer = vec![0f32; 1 * 4 * 512 * 896];
    let bytes_written = unsafe {
        wasi_nn::get_output(
            context,
            0,
            &mut output_buffer[..] as *mut [f32] as *mut u8,
            (output_buffer.len() * 4).try_into().unwrap(),
        )
        .unwrap()
    };

    println!("bytes_written: {:?}", bytes_written);

    Ok(output_buffer)
}

/// Dump data to the specified binary file.
fn dump(
    path: impl AsRef<std::path::Path>,
    buffer: impl AsRef<[f32]>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("dump tensor to {:?}", path.as_ref());

    let dst_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(
            buffer.as_ref().as_ptr() as *const u8,
            buffer.as_ref().len() * 4,
        )
    };
    println!("\tThe size of bytes: {}", dst_slice.len());

    std::fs::write(path, &dst_slice).expect("failed to write file");

    Ok(())
}

/// Distinguish the precision of each pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// Each pixel is an 8-bit value.
    U8,
    /// Each pixel is a 32-bit floating point value.
    FP32,
}
impl Dtype {
    /// Return the number of bytes occupied by the precision.
    pub fn bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::FP32 => 4,
        }
    }
}

/// Define the dimensions and pixel precision of an image.
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub struct Dimensions {
    batch: i32,
    height: i32,
    width: i32,
    channels: i32,
    dtype: Dtype,
}
impl Dimensions {
    /// Construct a new dimensions object.
    pub fn new(batch: i32, height: i32, width: i32, channels: i32, dtype: Dtype) -> Self {
        Self {
            batch,
            height,
            width,
            channels,
            dtype,
        }
    }

    /// Return the number of bytes that the dimensions should occupy.
    pub fn bytes(&self) -> usize {
        self.height as usize * self.width as usize * self.channels as usize * self.dtype.bytes()
    }
}
