
# adapted from https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

import torch
import struct
# Download an example image from the pytorch website
url, filename = (
    "https://github.com/bytecodealliance/wasi-nn/raw/main/rust/images/1.jpg", "../input.jpg")
# import urllib
# try:
#     urllib.URLopener().retrieve(url, filename)
# except:
#     urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
model = torch.hub.load('pytorch/vision:v0.10.0',
                       'mobilenet_v2', pretrained=True)
model.eval()

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
print(input_image.mode)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)
with open("image-1-3-244-244.rgb", 'wb') as f:
    order_data = input_batch.reshape(-1)
    for d in order_data:
        d = d.item()
        f.write(struct.pack('f', d))


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(top5_catid[i], categories[top5_catid[i]], top5_prob[i].item())
