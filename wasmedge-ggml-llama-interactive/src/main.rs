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
    let model_name: &str = &args[1];

    let mut options = json!({});
    match env::var("enable_log") {
        Ok(val) => options["enable-log"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    };
    match env::var("stream_stdout") {
        Ok(val) => options["stream-stdout"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    };
    match env::var("n_predict") {
        Ok(val) => options["n-preidct"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("reverse_prompt") {
        Ok(val) => options["reverse-prompt"] = json!(val),
        _ => (),
    }
    match env::var("n_gpu_layers") {
        Ok(val) => options["n-gpu-layers"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("ctx_size") {
        Ok(val) => options["ctx-size"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }
    match env::var("batch_size") {
        Ok(val) => options["batch-size"] = serde_json::from_str(val.as_str()).unwrap(),
        _ => (),
    }

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .config(options.to_string())
            .build_from_cache(model_name)
            .unwrap();
    let mut context = graph.init_execution_context().unwrap();

    let system_prompt = String::from("<<SYS>>You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>");
    let mut saved_prompt = String::new();

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
            if let Some(is_stream) = options["stream-stdout"].as_bool() {
                if !is_stream {
                    print!("{}", output.trim());
                }
            } else {
                print!("{}", output.trim());
            }
            println!("");

            saved_prompt = format!("{} {} ", saved_prompt, output.trim());
        }
    }
}
