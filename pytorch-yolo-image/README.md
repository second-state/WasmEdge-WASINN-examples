# YOLOV8 detection Example For WASI-NN with PyTorch Backend

Code for a working example of object detection using the yolov8 model. 

### Requirements 

`pip3 install torchvision ultralytics`  

To get the Pytorch YOLOV8 model please use the `get_model.py` script provided. 

Alternatively: pre-trained weights can be downloaded directly from github and will need to be converted to torchscript
https://github.com/ultralytics/assets/releases

### Build + Run

To build (from the `/rust` directory):  
`cargo build --release --target wasm32-wasip1`

To compile the WASM AOT (Ahead of Time) - this results in a much more performant binary  
`
wasmedge compile ./target/wasm32-wasip1/release/wasmedge-wasinn-example-yolo-image.wasm ./target/wasm32-wasip1/release/wasmedge-wasinn-example-yolo-image.wasm
`

To Run (from root `/` directory of the example):  
`
wasmedge --dir .:. ./rust/target/wasm32-wasip1/release/wasmedge-wasinn-example-yolo-image.wasm ./yolov8n.torchscript ./input.jpg
`

> **Note**
> Non-Maximum supression has not been applied to the output, therefore there will be many results with very close pixel bounds