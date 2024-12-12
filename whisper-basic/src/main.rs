use serde_json::json;
use std::env;
use std::error::Error;
use std::fs;
use wasmedge_wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding, TensorType};

const MAX_BUFFER_SIZE: usize = 2usize.pow(14) * 15 + 128;

pub fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    // let wav_name: &str = &args[2];

    let wav_names = vec![
        "output_000.wav",
        "output_001.wav",
        "output_002.wav",
        "output_003.wav",
        "output_004.wav",
        "output_005.wav",
        "output_006.wav",
    ];

    let model_bin = fs::read(model_bin_name)?;
    // println!("Read model, size in bytes: {}", model_bin.len());

    println!("\n");

    for wav_name in wav_names.iter() {
        println!("Processing: {}", wav_name);

        // create config
        let config = json!({
            "max-len": 1,
            "split-on-word": true
        });
        let config = serde_json::to_string(&config)?;
        // println!("Config: {}", &config);
        let config_bytes = config.as_bytes().to_vec();

        let graph = GraphBuilder::new(GraphEncoding::Whisper, ExecutionTarget::CPU)
            .build_from_bytes(&[&model_bin])?;

        let mut ctx = graph.init_execution_context()?;
        // println!("Loaded graph into wasi-nn with ID: {}", graph);

        // set metadata
        ctx.set_input(1, wasmedge_wasi_nn::TensorType::U8, &[1], &config_bytes)?;
        // println!("Set config");

        // Load the raw pcm tensor.
        let wav_buf = fs::read(wav_name)?;
        // println!("Read input tensor, size in bytes: {}", wav_buf.len());

        // Set input.
        ctx.set_input(0, TensorType::F32, &[1, wav_buf.len()], &wav_buf)?;

        // Execute the inference.
        ctx.compute()?;

        // Retrieve the output.
        let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
        let size = ctx.get_output(0, &mut output_buffer)?;
        unsafe {
            output_buffer.set_len(size);
        }

        println!(
            "Recognized from audio: \n{}",
            String::from_utf8(output_buffer).unwrap()
        );

        // * explicitly drop ctx and graph to release memory
        drop(ctx);
        drop(graph);

        println!("--------------------------------");
        let sec = 30;
        println!("Sleeping {} seconds for observing the memory usage", sec);
        std::thread::sleep(std::time::Duration::from_secs(sec));
        println!("\n\n");
    }

    println!("All {} wav files processed", wav_names.len());
    let sec = 40;
    println!("Sleeping {} seconds for observing the memory usage", sec);
    std::thread::sleep(std::time::Duration::from_secs(sec));

    Ok(())
}
