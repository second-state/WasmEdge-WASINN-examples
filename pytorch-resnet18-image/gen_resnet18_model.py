import os
import torch
from torch import jit

with torch.no_grad():
    fake_input = torch.rand(1, 3, 224, 224)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    out1 = model(fake_input).squeeze()

    sm = torch.jit.script(model)
    if not os.path.exists("resnet18.pt"):
        sm.save("resnet18.pt")
    load_sm = jit.load("resnet18.pt")
    out2 = load_sm(fake_input).squeeze()

    print(out1[:5], out2[:5])
