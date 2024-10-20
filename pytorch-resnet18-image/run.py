import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_so_path = os.path.join(os.getcwd(), "resnet18_pt2.so")

model = torch._export.aot_load(model_so_path, device)
example_inputs = (torch.randn(1, 3, 224, 224, device=device),)

with torch.inference_mode():
    output = model(example_inputs)
    print(output)
