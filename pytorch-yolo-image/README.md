# YOLOV8 detection Example For WASI-NN with PyTorch Backend

Code for a working example of object detection using the yolov8 model. 

To build (from the `/rust` directory):  
`cargo build --release --target wasm32-wasi`

To compile the WASM AOT (Ahead of Time) - this results in a much more performant binary  
```bash
wasmedge compile ./target/wasm32-wasi/release/wasmedge-wasinn-example-yolo-image.wasm ./target/wasm32-wasi/release/wasmedge-wasinn-example-yolo-image.wasm
```


To Run (from root `/` directory of the example):  
```bash
wasmedge --dir .:. ./rust/target/wasm32-wasi/release/wasmedge-wasinn-example-yolo-image.wasm ./yolov8n.torchscript ./dog.png
```  


> **Note**
> Non-Maximum supression has not been applied to the output, therefore there will be many results with very close pixel bounds

To get the Pytorch YOLOV8 model we can use the `get_model.py` script provided, 
This requires the `ultralytics` python package which can be installed via

`pip install ultralytics`  

Alternatively: pre-trained weights can be downloaded directly from github
https://github.com/ultralytics/assets/releases

> **Note**
> The Pytorch `.pt` files downloaded from github will have to be converted to torchscript
