use std::env;
use wasmedge_wasi_nn::{self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding};

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];

    // Create graph and initialize context.
    let graph =
        GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO).build_from_cache(model_name);

    // Check graph
    match graph {
        Err(Error::BackendError(BackendError::ModelNotFound)) => {
            println!("Model not found");
        }
        Err(_) => {
            panic!("Should be model not found");
        }
        Ok(_) => {
            panic!("Should be model not found");
        }
    }
}
