use hound;
use serde_json::json;
use wasmedge_wasi_nn::{
    self, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext, TensorType,
};

fn get_data_from_context(context: &GraphExecutionContext, index: usize, limit: usize) -> Vec<u8> {
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 4096;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let _ = context
        .get_output(index, &mut output_buffer)
        .expect("Failed to get output");

    return output_buffer[..limit].to_vec();
}

fn main() {
    let prompt = "It is [uv_break] test sentence [laugh] for chat T T S";
    let tensor_data = prompt.as_bytes().to_vec();
    let config_data = serde_json::to_string(&json!({"prompt": "[oral_2][laugh_0][break_6]", "spk_emb": "random", "temperature": 0.5, "top_k": 0, "top_p": 0.9}))
        .unwrap()
        .as_bytes()
        .to_vec();
    let empty_vec: Vec<Vec<u8>> = Vec::new();
    let graph = GraphBuilder::new(GraphEncoding::ChatTTS, ExecutionTarget::CPU)
        .build_from_bytes(empty_vec)
        .expect("Failed to build graph");
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");
    context
        .set_input(0, TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");
    context
        .set_input(1, TensorType::U8, &[1], &config_data)
        .expect("Failed to set input");
    context.compute().expect("Failed to compute");
    let bytes_written = get_data_from_context(&context, 1, 4);
    let bytes_written = usize::from_le_bytes(bytes_written.as_slice().try_into().unwrap());
    println!("Byte: {}", bytes_written);
    let output_bytes = get_data_from_context(&context, 0, bytes_written);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output1.wav", spec).unwrap();
    let samples: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    for sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    graph.unload().expect("Failed to free resource");
}
