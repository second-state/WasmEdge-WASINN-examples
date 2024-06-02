use hound::{self, SampleFormat};
use std::process;

pub fn main() {
    let wav_file = std::env::args().nth(1).expect("No wav file name provided");
    let model_name = std::env::args().nth(2).expect("No model name provided");

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Burn, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(&model_name)
            .expect("Failed to build graph");

    println!("Loaded graph into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    println!("Created wasi-nn execution context with ID: {:?}", context);

    println!("Loading waveform...");
    let (waveform, sample_rate) = match load_audio_waveform(&wav_file) {
        Ok((w, sr)) => (w, sr),
        Err(e) => {
            eprintln!("Failed to load audio file: {}", e);
            process::exit(1);
        }
    };
    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");

    context
        .set_input(0, wasi_nn::TensorType::F32, &[1, waveform.len()], &waveform)
        .unwrap();

    context.compute().unwrap();
    println!("Executed audio to text converter.");

    let mut output_buffer = vec![0u8; 100];
    context.get_output(0, &mut output_buffer).unwrap();

    match String::from_utf8(output_buffer) {
        Ok(s) => println!("Text: {}", s),
        Err(e) => println!("Error: {}", e),
    }
}

fn load_audio_waveform(filename: &str) -> hound::Result<(Vec<f32>, usize)> {
    let reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    // let duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    // let bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");
    assert_eq!(channels, 1, "The audio must be single-channel.");

    let max_int_val = 2_u32.pow(spec.bits_per_sample as u32 - 1) - 1;

    let floats = match sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect::<hound::Result<_>>()?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()?,
    };

    return Ok((floats, sample_rate));
}
