use serde_json::json;
use std::env;
use std::io;
use wasi_nn;

fn read_input() -> String {
    loop {
        let mut answer = String::new();
        io::stdin()
            .read_line(&mut answer)
            .ok()
            .expect("Failed to read line");
        if !answer.is_empty() && answer != "\n" && answer != "\r\n" {
            return answer;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let stream_stdout: bool = env::var("stream_stdout")
        .unwrap_or("false".to_string())
        .trim()
        .parse()
        .unwrap();
    let enable_log: bool = env::var("enable_log")
        .unwrap_or("false".to_string())
        .trim()
        .parse()
        .unwrap();
    let ctx_size: i32 = env::var("ctx_size")
        .unwrap_or("512".to_string())
        .trim()
        .parse()
        .unwrap();
    let n_predict: i32 = env::var("n_predict")
        .unwrap_or("512".to_string())
        .trim()
        .parse()
        .unwrap();
    let n_gpu_layers: i32 = env::var("n_gpu_layers")
        .unwrap_or("0".to_string())
        .trim()
        .parse()
        .unwrap();

    let model_name: &str = &args[1];

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(model_name)
            .unwrap();
    let mut context = graph.init_execution_context().unwrap();

    let system_prompt = String::from("<<SYS>>You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>");
    let mut saved_prompt = String::new();

    // Set options to input with index 1
    let options = json!({
        "stream-stdout": stream_stdout,
        "enable-log": enable_log,
        "ctx-size": ctx_size,
        "n-predict": n_predict,
        "n-gpu-layers": n_gpu_layers,
    });
    context
        .set_input(
            1,
            wasi_nn::TensorType::U8,
            &[1],
            &options.to_string().as_bytes().to_vec(),
        )
        .unwrap();

    // If there is a third argument, use it as the prompt and enter non-interactive mode.
    // Otherwise, enter interactive mode.
    if args.len() >= 3 {
        let prompt = &args[2];
        println!("Prompt:\n{}", prompt);
        let tensor_data = prompt.as_bytes().to_vec();
        context
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .unwrap();
        println!("Response:");
        context.compute().unwrap();
        let max_output_size = 4096 * 6;
        let mut output_buffer = vec![0u8; max_output_size];
        let mut output_size = context.get_output(0, &mut output_buffer).unwrap();
        output_size = std::cmp::min(max_output_size, output_size);
        let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();
        println!("{}", output.trim());
    } else {
        loop {
            println!("Question:");
            let input = read_input();
            if saved_prompt == "" {
                saved_prompt = format!("[INST] {} {} [/INST]", system_prompt, input.trim());
            } else {
                saved_prompt = format!("{} [INST] {} [/INST]", saved_prompt, input.trim());
            }

            // Set prompt to the input tensor.
            let tensor_data = saved_prompt.as_bytes().to_vec();
            context
                .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
                .unwrap();

            // Execute the inference.
            println!("Answer:");
            context.compute().unwrap();

            // Retrieve the output.
            let max_output_size = 4096 * 6;
            let mut output_buffer = vec![0u8; max_output_size];
            let mut output_size = context.get_output(0, &mut output_buffer).unwrap();
            output_size = std::cmp::min(max_output_size, output_size);
            let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();
            if !stream_stdout {
                print!("{}", output.trim());
            }
            println!("");

            saved_prompt = format!("{} {} ", saved_prompt, output.trim());
        }
    }
}
