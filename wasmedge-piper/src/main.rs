fn main() {
    // create graph with the config
    let config = serde_json::json!({
        "model": "en_US-lessac-medium.onnx", // path to .onnx voice file, required
        "config": "en_US-lessac-medium.onnx.json", // path to JSON voice config file, optional, default is model path + .json
        "espeak_data": "espeak-ng-data", // path to espeak-ng data directory, required for espeak phonemes
    });
    let graph = wasmedge_wasi_nn::GraphBuilder::new(
        wasmedge_wasi_nn::GraphEncoding::Piper,
        wasmedge_wasi_nn::ExecutionTarget::CPU,
    )
    .build_from_bytes([config.to_string()])
    .unwrap();

    let mut context = graph.init_execution_context().unwrap();

    // set the input text
    let text = "Welcome to the world of speech synthesis!";
    context
        .set_input(
            0,
            wasmedge_wasi_nn::TensorType::U8,
            &[text.len()],
            text.as_bytes(),
        )
        .unwrap();

    // synthesize the audio
    context.compute().unwrap();

    // retrieve the output, output is wav by default
    let mut out_buffer = vec![0u8; 1 << 20];
    let size = context.get_output(0, &mut out_buffer).unwrap();

    std::fs::write("welcome.wav", &out_buffer[..size]).unwrap();
}
