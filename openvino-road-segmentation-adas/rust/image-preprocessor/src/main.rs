//! run `cargo run -- --image <path-to-your-image> --dims <target tensor dimensions> --tensor <filename saving tensor>`
//!
//! For example,
//!
//! `cargo run -- --image ../image/empty_road_mapillary.jpg --dims 512x896x3xfp32 --tensor wasinn-openvino-inference-input-512x896x3xf32-bgr.tensor`
//!
//! Or, you can build the project first by running `cargo build --release`;
//! then, go to the `./target/release/` directory, there you can find `im2tensor`;
//! run `im2tensor --image ../image/empty_road_mapillary.jpg --dims 512x896x3xfp32 --tensor  wasinn-openvino-inference-input-512x896x3xf32-bgr.tensor`

use anyhow::{anyhow, Result};
use opencv::{
    core::{Scalar_, Size_},
    prelude::*,
};
use std::{num::ParseIntError, slice, str::FromStr};
use structopt::StructOpt;
use thiserror::Error;

fn main() -> Result<()> {
    let options = Options::from_args();

    let path = options.input.as_path();
    let filename = &path
        .to_str()
        .ok_or(anyhow!("Unable to stringify the path."))?;

    let dimensions = Dimensions::from_str(&options.dimensions)?;

    // load image
    println!("load image ...");
    let src: Mat = opencv::imgcodecs::imread(filename, opencv::imgcodecs::IMREAD_COLOR)?;
    println!("\tThe input image has size = {:?}, channels = {}, type = {}, total size = {}, item size (bytes) = {}", src.size().unwrap(), src.channels(), src.typ(), src.total() as i32 * src.channels(), src.elem_size1());

    // Resize the src image to a specific dimension
    let dst = resize(&src, dimensions)?;

    // dump the preprocess image as &[u8]
    dump(options.output, dst)?;

    Ok(())
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "im2tensor",
    about = "Decode and resize images into valid tensors for wasi-nn OpenVINO backend."
)]
struct Options {
    /// Image file.
    #[structopt(name = "INPUT FILE", long = "--image", parse(from_os_str))]
    input: std::path::PathBuf,

    /// File the generated tensor dumps to.
    #[structopt(name = "OUTPUT FILE", long = "--tensor", parse(from_os_str))]
    output: std::path::PathBuf,

    /// The dimensions of the output file as "[height]x[width]x[channels]x[precision]"; e.g. 300x300x3xfp32.
    #[structopt(name = "OUTPUT DIMENSIONS", long = "--dims")]
    dimensions: String,
}

/// Resize the source image to the specified dimensions
fn resize(src: &Mat, dimensions: Dimensions) -> Result<Mat> {
    println!("convert image to wasi-nn OpenVINO tensor ...");

    let mut resized = Mat::new_rows_cols_with_default(
        dimensions.height,
        dimensions.width,
        dimensions.as_type(),
        Scalar_::all(0.0),
    )?;
    println!("\tBefore resizing, the `resize` image has size = {:?}, channels = {}, type = {}, total size = {}, item size (bytes) = {}", resized.size().unwrap(), resized.channels(), resized.typ(), dimensions.bytes(), resized.elem_size1());

    // Resize the `src` Mat into the `dst` Mat using bilinear interpolation
    let dst_size: Size_<i32> = resized.size()?;
    opencv::imgproc::resize(
        &src,
        &mut resized,
        dst_size,
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    println!("\tAfter resizing, the `resize` image has size = {:?}, channels = {}, type = {}, total size = {}, item size (bytes) = {}", resized.size()?, resized.channels(), resized.typ(), dimensions.bytes(), resized.elem_size1());

    // Because `imgproc::resize` can alter the depth/precision of our destination image, we convert the `resized` image
    // to the appropriate `Precision`.
    let mut dst: Mat = Mat::new_rows_cols_with_default(
        dimensions.height,
        dimensions.width,
        dimensions.as_type(),
        Scalar_::all(0.0),
    )?;
    // The alpha/beta values are the defaults from C++.
    resized.convert_to(&mut dst, dimensions.as_type(), 1.0, 0.0)?;
    println!("\tAfter conversion, the `dst` image has size = {:?}, channels = {}, type = {}, total size = {}, item size (bytes) = {}", dst.size().unwrap(), dst.channels(), dst.typ(), dimensions.bytes(), dst.elem_size1());

    Ok(dst)
}

/// Dump data to the specified binary file.
fn dump(path: impl AsRef<std::path::Path>, mat: Mat) -> Result<()> {
    println!("dump tensor to {:?}", path.as_ref());

    let size = mat.size()?;
    let len = size.height * size.width * mat.channels() * mat.elem_size1() as i32;
    let dst_slice: &[u8] = unsafe { slice::from_raw_parts(mat.data() as *const u8, len as usize) };
    println!("\tThe size of bytes: {}", dst_slice.len());

    std::fs::write(path, &dst_slice).expect("failed to write file");

    Ok(())
}

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("{0}")]
    OpencvError(String),
    #[error("{0}")]
    ParseError(String),
}
impl From<opencv::Error> for ConversionError {
    fn from(e: opencv::Error) -> Self {
        Self::OpencvError(e.message)
    }
}
impl From<ParseIntError> for ConversionError {
    fn from(e: ParseIntError) -> Self {
        Self::ParseError(format!("parsing error: {}", e.to_string()))
    }
}

/// Distinguish the precision of each pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Each pixel is an 8-bit value.
    U8,
    /// Each pixel is a 32-bit floating point value.
    FP32,
}
impl Precision {
    /// Return the number of bytes occupied by the precision.
    pub fn bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::FP32 => 4,
        }
    }
}
impl FromStr for Precision {
    type Err = ConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "u8" => Ok(Self::U8),
            "fp32" => Ok(Self::FP32),
            _ => Err(ConversionError::ParseError(format!(
                "unrecognized precision: {}",
                s
            ))),
        }
    }
}

/// Define the dimensions and pixel precision of an image.
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub struct Dimensions {
    height: i32,
    width: i32,
    channels: i32,
    precision: Precision,
}
impl Dimensions {
    /// Construct a new dimensions object.
    pub fn new(height: i32, width: i32, channels: i32, precision: Precision) -> Self {
        Self {
            height,
            width,
            channels,
            precision,
        }
    }

    /// Return the number of bytes that the dimensions should occupy.
    pub fn bytes(&self) -> usize {
        self.height as usize * self.width as usize * self.channels as usize * self.precision.bytes()
    }

    /// See https://docs.opencv.org/2.4/modules/core/doc/basic_structures.html for a description of the various OpenCV
    /// primitive types.
    fn as_type(&self) -> i32 {
        use Precision::*;
        match (self.precision, self.channels) {
            (FP32, 3) => opencv::core::CV_32FC3,
            (U8, 3) => opencv::core::CV_8UC3,
            _ => unimplemented!(),
        }
    }
}
impl FromStr for Dimensions {
    type Err = ConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.trim().split('x').collect();
        if parts.len() != 4 {
            return Err(ConversionError::ParseError("Not enough parts in dimension string; should be [height]x[width]x[channels]x[precision]".to_string()));
        }
        let height = i32::from_str(parts[0])?;
        let width = i32::from_str(parts[1])?;
        let channels = i32::from_str(parts[2])?;
        let precision = Precision::from_str(parts[3])?;
        Ok(Self {
            height,
            width,
            channels,
            precision,
        })
    }
}
