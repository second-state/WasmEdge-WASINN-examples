use std::env;
use wasmedge_nn::{
    cv::image_to_bytes,
    nn::{ctx::WasiNnCtx, Dtype, ExecutionTarget, GraphEncoding, Tensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_xml_name: &str = &args[1];
    let model_bin_name: &str = &args[2];
    let image_name: &str = &args[3];

    // load image and convert it to tensor
    println!("Load image file and convert it into tensor ...");
    let bytes = image_to_bytes(image_name.to_string(), 512, 896, Dtype::F32)?;
    let tensor = Tensor {
        dimensions: &[1, 3, 512, 896],
        r#type: Dtype::F32.into(),
        data: bytes.as_slice(),
    };
    // create wasi-nn context
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
