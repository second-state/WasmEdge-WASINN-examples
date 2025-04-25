
from transformers import AutoProcessor
import mlx.core as mx
from PIL import Image, ImageOps
import sys


def encode(processor, image, prompts):
    model_inputs = {}
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    images = [image]
    inputs = processor(
        text=prompts, images=images, padding=True, return_tensors="mlx"
    )
    if "images" in inputs:
        inputs["pixel_values"] = inputs["images"]
        inputs.pop("images")

    if isinstance(inputs["pixel_values"], list):
        pixel_values = inputs["pixel_values"]
    else:
        pixel_values = mx.array(inputs["pixel_values"])

    model_inputs["pixel_values"] = pixel_values
    model_inputs["attention_mask"] = (
        mx.array(inputs["attention_mask"]
                 ) if "attention_mask" in inputs else None
    )
    # Convert inputs to model_inputs with mx.array if present
    for key, value in inputs.items():
        if key not in model_inputs and not isinstance(value, (str, list)):
            model_inputs[key] = mx.array(value)
    mx.save("input_ids.npy", model_inputs["input_ids"])
    mx.save("pixel_values.npy", model_inputs["pixel_values"])
    mx.save("mask.npy", model_inputs["attention_mask"])


if __name__ == "__main__":
    model_path, image, prompts = sys.argv[1:]
    processor = AutoProcessor.from_pretrained(model_path)
    formatted_prompt = f"<bos><start_of_turn>user\n\
        {prompts}<start_of_image><end_of_turn>\n\
            <start_of_turn>model"
    encode(processor, image, formatted_prompt)
