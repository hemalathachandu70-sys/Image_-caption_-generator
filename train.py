# ...existing code...
import os
import argparse
import time
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments,
)

from data_utils import Flickr8kDataset, CaptionDataCollator

# Default paths (your dataset)
DEFAULT_IMAGES_DIR = r"C:\Users\prart\OneDrive\Desktop\mini project\Flickr8k_Dataset\Flicker8k_Dataset"
DEFAULT_TEXT_DIR = r"C:\Users\prart\OneDrive\Desktop\mini project\Flickr8k_text"
DEFAULT_TRAIN_LIST = os.path.join(DEFAULT_TEXT_DIR, "Flickr_8k.trainImages.txt")
DEFAULT_TOKEN_FILE = os.path.join(DEFAULT_TEXT_DIR, "Flickr8k.token.txt")

MODEL_ENCODER = "google/vit-base-patch16-224-in21k"
MODEL_DECODER = "gpt2"


def read_image_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train Vision->Caption model on Flickr8k")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--output-dir", default="outputs_flickr8k_v1", help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze vision encoder weights to speed up training")
    parser.add_argument("--subset", type=int, default=0, help="Use only first N images (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = select_device()
    print(f"Using device: {device}")

    # Prepare model + tokenizer + image processor
    print("Loading model and tokenizers...")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_ENCODER, MODEL_DECODER)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DECODER)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ENCODER)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = args.max_length
    model.config.num_beams = 4

    if args.freeze_encoder:
        print("Freezing encoder parameters...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Data
    train_images = read_image_list(args.train_list)
    if args.subset and args.subset > 0:
        train_images = train_images[: args.subset]
        print(f"Using subset of {len(train_images)} images for quick test")

    train_dataset = Flickr8kDataset(
        images_dir=args.images_dir,
        captions_file=args.token_file,
        tokenizer_name=MODEL_DECODER,
        image_transform=None,
        max_length=args.max_length,
        image_filenames=train_images,
    )
    data_collator = CaptionDataCollator(processor=image_processor, tokenizer=tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=(device.type == "cuda"),
        bf16=False,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.2f} minutes")

    # Save final model / tokenizer / processor
    print("Saving model to", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print("Saved.")

if __name__ == "__main__":
    main()
# ...existing code...
