# ...existing code...
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        tokenizer_name: str,
        image_transform=None,
        max_length: int = 64,
        image_filenames: Optional[List[str]] = None,
    ) -> None:
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Keep transform for compatibility, but default to None and avoid applying it
        self.image_transform = image_transform
        self.max_length = max_length

        image_to_captions: Dict[str, List[str]] = {}
        # Parse caption file robustly (Flickr token file format: "image.jpg#0 <tab> caption" or similar)
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # try tab-separated first
                if "\t" in line:
                    left, caption = line.split("\t", 1)
                else:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        left, caption = parts
                    else:
                        continue
                # left is like "1000268201_693b08cb0e.jpg#0"
                image_name = left.split("#")[0]
                caption = caption.strip()
                image_to_captions.setdefault(image_name, []).append(caption)

        self.entries: List[Dict[str, str]] = []
        candidates = image_filenames or list(image_to_captions.keys())
        for image_name in candidates:
            if image_name in image_to_captions:
                # Use first caption (or you can randomize/select all)
                for cap in image_to_captions[image_name]:
                    self.entries.append({"image": image_name, "caption": cap})

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.entries[idx]
        image_path = os.path.join(self.images_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        # Do NOT apply transforms; let the image processor handle resizing/normalization
        tokenized = self.tokenizer(
            item["caption"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "pixel_values": image,  # raw PIL image
            "labels": tokenized["input_ids"],
        }


@dataclass
class CaptionDataCollator:
    processor: Any  # feature extractor / image processor for vision
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [f["pixel_values"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        batch = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )
        # replace pad token ids in labels with label_pad_token_id for loss masking
        if self.tokenizer.pad_token_id is None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = self.tokenizer.pad_token_id
        labels_tensor = batch["input_ids"].clone()
        labels_tensor[labels_tensor == pad_id] = self.label_pad_token_id
        batch["labels"] = labels_tensor
        batch["pixel_values"] = pixel_values
        return batch
# ...existing code...
