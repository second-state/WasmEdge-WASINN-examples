# More detail please follow the link below
# https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html

import os
import torch
from torchvision.models import ResNet18_Weights, resnet18

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with torch.inference_mode():

    # Specify the generated shared library path
    aot_compile_options = {
            "aot_inductor.output_path": os.path.join(os.getcwd(), "resnet18_pt2.so"),
    }
    if torch.cuda.is_available():
        device = "cuda"
        aot_compile_options.update({"max_autotune": True})
    else:
        device = "cpu"

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

    # min=2 is not a bug and is explained in the 0/1 Specialization Problem
    batch_dim = torch.export.Dim("batch", min=2, max=32)
    exported_program = torch.export.export(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
    )
    so_path = torch._inductor.aot_compile(
        exported_program.module(),
        example_inputs,
        # Specify the generated shared library path
        options=aot_compile_options
    )
