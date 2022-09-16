import sys
import struct
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    input_image = Image.open(sys.argv[1])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    with open(sys.argv[2], 'wb') as f:
        order_data = input_batch.reshape(-1)
        for d in order_data:
            d = d.item()
            f.write(struct.pack('f', d))
