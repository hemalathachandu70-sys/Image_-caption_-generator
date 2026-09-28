import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--num-beams", type=int, default=3)
    args = p.parse_args()

    dev = device()
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if dev.type != "cpu" else torch.float32
    ).to(dev).eval()

    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(
        dev, dtype=torch.float16 if dev.type != "cpu" else torch.float32
    )
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    print(processor.decode(ids[0], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
