# ...existing code...
import os
import argparse
import subprocess
import time
import tempfile
import shutil
import sys
from typing import List

import torch
from PIL import Image, ImageTk
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
)

# optional GUI / TTS imports handled at runtime
try:
    import tkinter as tk
    from tkinter import filedialog
    tk_available = True
except Exception:
    tk_available = False

try:
    import pyttsx3
    tts_available = True
except Exception:
    tts_available = False

# default dataset image folder (your provided absolute path)
IMAGES_DIR = r"C:\Users\prart\OneDrive\Desktop\mini project\Flickr8k_Dataset\Flicker8k_Dataset"
# safe default pretrained captioning model if you don't have outputs/
DEFAULT_MODEL = "nlpconnect/vit-gpt2-image-captioning"


def list_images(path: str) -> List[str]:
    if os.path.isdir(path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts]
        files.sort()
        return files
    if os.path.isfile(path):
        return [path]
    return []


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str):
    """
    Load model/tokenizer/processor. Set tokenizer/model config to avoid attention-mask warnings.
    """
    model = None
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
    except Exception:
        # fallback to public checkpoint if local 'outputs' missing
        if model_path == "outputs" and not os.path.isdir(os.path.join(os.path.dirname(__file__), "outputs")):
            print("Local 'outputs' not found or failed to load. Attempting to load the public checkpoint...")
            model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_MODEL)
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            image_processor = AutoImageProcessor.from_pretrained(DEFAULT_MODEL)
        else:
            raise

    # Ensure tokenizer has pad token and model config set to reduce warnings
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "cls_token_id", None) or tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token

    return model, tokenizer, image_processor


def generate(model, tokenizer, image_processor, device: torch.device, image_path: str, max_length: int = 64, num_beams: int = 4) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    generate_kwargs = {"pixel_values": pixel_values, "max_length": max_length, "num_beams": num_beams}
    if "pixel_mask" in inputs:
        generate_kwargs["encoder_attention_mask"] = inputs["pixel_mask"].to(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def pick_image_via_dialog(initialdir: str = IMAGES_DIR) -> str:
    if not tk_available:
        raise RuntimeError("tkinter not available. Provide an image path as the first argument instead.")
    root = tk.Tk()
    root.withdraw()
    opts = {"initialdir": initialdir, "filetypes": [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]}
    filename = filedialog.askopenfilename(**opts)
    root.destroy()
    return filename


def speak_text(text: str, volume: float = 1.0) -> None:
    """
    Reliable Windows playback: write WAV via pyttsx3 then play with winsound.
    If pyttsx3 missing or fails, try PowerShell SAPI fallback.
    Prints diagnostics.
    """
    try:
        v = max(0.0, min(1.0, float(volume)))
    except Exception:
        v = 1.0

    # On Windows: prefer WAV + winsound playback for reliable audible output in VSCode terminal
    if os.name == "nt":
        # Try to create WAV via pyttsx3
        if tts_available:
            tmp_dir = None
            try:
                tmp_dir = tempfile.mkdtemp(prefix="tts_")
                wav_path = os.path.join(tmp_dir, "tts_output.wav")
                engine = pyttsx3.init(driverName="sapi5")
                try:
                    engine.setProperty("volume", v)
                except Exception:
                    pass
                engine.save_to_file(text, wav_path)
                engine.runAndWait()
                try:
                    import winsound
                    winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    print("TTS: played WAV via winsound.")
                    return
                except Exception as e:
                    print("winsound playback failed:", repr(e))
                finally:
                    # try to cleanup file
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
            except Exception as e:
                print("pyttsx3 save_to_file failed:", repr(e))
            finally:
                if tmp_dir:
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass
        # PowerShell fallback
        try:
            vol_percent = int(v * 100)
            safe = text.replace('"', '\\"')
            ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Volume = {vol_percent}; $s.Speak(\"{safe}\")'
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
            if proc.returncode == 0:
                print("TTS: spoken with PowerShell SAPI")
                return
            else:
                print("PowerShell TTS failed, stderr:", proc.stderr.strip())
        except Exception as e:
            print("PowerShell TTS fallback exception:", repr(e))
        print("TTS failed on Windows.")
        return

    # Non-Windows: try direct pyttsx3 speak, else fallback to engine.runAndWait
    if tts_available:
        try:
            engine = pyttsx3.init()
            try:
                engine.setProperty("volume", v)
            except Exception:
                pass
            engine.say(text)
            engine.runAndWait()
            print("TTS: spoken with pyttsx3 (non-Windows).")
            return
        except Exception as e:
            print("pyttsx3 non-Windows direct speak failed:", repr(e))

    # Final fallback: try system call via say / espeak if available
    try:
        if shutil.which("say"):
            subprocess.run(["say", text], check=False)
            print("TTS: used 'say' command")
            return
        if shutil.which("espeak"):
            subprocess.run(["espeak", text], check=False)
            print("TTS: used 'espeak' command")
            return
    except Exception as e:
        print("System TTS fallback failed:", repr(e))

    print("No usable TTS method found.")


def show_image_window_and_caption(image_path: str, caption_text: str | None, speak: bool, volume: float) -> None:
    """
    Open an external window showing the image and a caption label below it.
    If caption_text is None, shows 'Generating...' until updated externally.
    This function blocks until the window is closed.
    """
    if not tk_available:
        # fallback: print path and caption
        print(f"[Image: {image_path}]")
        if caption_text is None:
            print("Caption: (generating...)")
        else:
            print("Caption:\n  " + caption_text)
        if speak and caption_text:
            speak_text(caption_text, volume=volume)
        return

    root = tk.Tk()
    root.title(os.path.basename(image_path))

    # image frame
    frm = tk.Frame(root)
    frm.pack(fill="both", expand=True)

    try:
        img = Image.open(image_path)
        max_w, max_h = 900, 700
        w, h = img.size
        scale = min(1.0, max_w / w, max_h / h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl_img = tk.Label(frm, image=tk_img)
        lbl_img.image = tk_img  # keep reference
        lbl_img.pack(side="top", padx=8, pady=8)
    except Exception as e:
        lbl_img = tk.Label(frm, text=f"Failed to load image: {e}")
        lbl_img.pack(side="top", padx=8, pady=8)

    # caption label
    caption_var = tk.StringVar(value="Generating..." if caption_text is None else caption_text)
    lbl_caption = tk.Label(frm, textvariable=caption_var, wraplength=880, justify="center", font=("Arial", 14))
    lbl_caption.pack(side="top", padx=12, pady=(0, 12))

    # controls frame
    ctrl = tk.Frame(root)
    ctrl.pack(side="bottom", fill="x", pady=6)
    btn_close = tk.Button(ctrl, text="Close", command=root.destroy)
    btn_close.pack(side="right", padx=8)

    # speak button (manual) and auto-speak flag
    def on_speak_now():
        val = caption_var.get()
        if val and val != "Generating...":
            speak_text(val, volume=volume)

    btn_speak = tk.Button(ctrl, text="Speak", command=on_speak_now)
    btn_speak.pack(side="right", padx=8)

    # allow external update function to set caption and optionally auto speak
    def update_caption(new_caption: str, do_speak: bool):
        caption_var.set(new_caption)
        if do_speak:
            # speak in separate process to avoid blocking UI
            root.after(50, lambda: speak_text(new_caption, volume=volume))

    # attach to root so caller can access update_caption via attribute
    root.update_caption = update_caption  # type: ignore

    # center window
    root.update_idletasks()
    w_root = root.winfo_width()
    h_root = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w_root // 2)
    y = (root.winfo_screenheight() // 2) - (h_root // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show image (external window), then generate caption below it and optionally speak.")
    parser.add_argument("path", nargs="?", type=str, default=None, help="Image file or folder. If omitted, file picker opens (tkinter).")
    parser.add_argument("--model", type=str, default="outputs", help="Local model dir or HF id (default 'outputs' then fallback to public).")
    parser.add_argument("--max-length", type=int, default=32, help="Max tokens for caption (lower = faster).")
    parser.add_argument("--num-beams", type=int, default=1, help="Beams for generation (1 = fastest).")
    parser.add_argument("--speak", action="store_true", help="Speak caption after generation.")
    parser.add_argument("--volume", type=float, default=1.0, help="TTS volume 0.0-1.0.")
    parser.add_argument("--no-gui", action="store_true", help="Do not show GUI window (print instead).")
    args = parser.parse_args()

    device = select_device()
    print(f"Using device: {device}")

    path = args.path
    if path is None:
        try:
            path = pick_image_via_dialog()
        except RuntimeError as e:
            print(f"{e}\nYou can run: python infer.py \"C:\\path\\to\\image.jpg\"")
            return
        if not path:
            print("No file selected. Exiting.")
            return

    images = list_images(path)
    if not images:
        print(f"No images found at {path}")
        return

    try:
        model, tokenizer, image_processor = load_model(args.model)
    except Exception as e:
        print("Failed to load model:", repr(e))
        return

    model.to(device)
    model.eval()

    for img_path in images:
        print("\n" + "=" * 10 + f" {os.path.basename(img_path)} " + "=" * 10 + "\n")

        # create GUI window first (shows image and "Generating..."), unless user disabled GUI
        root_window = None
        if not args.no_gui and tk_available:
            # show image and generating label in a window; it will block until closed.
            # To allow updating the caption after generation, we create the window, then run generation
            # and call root.update_caption(...) from this thread (tkinter supports this before mainloop returns).
            # Implementation: open window in separate process of code path: we open window, but we need to return a handle to update.
            # Simpler approach: open window, then schedule generation via after so image displays immediately.
            def open_window_and_generate():
                # open window with no caption (it shows "Generating...")
                win = tk.Tk()
                win.title(os.path.basename(img_path))
                try:
                    img = Image.open(img_path)
                    max_w, max_h = 900, 700
                    w, h = img.size
                    scale = min(1.0, max_w / w, max_h / h)
                    if scale < 1.0:
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img)
                    lbl_img = tk.Label(win, image=tk_img)
                    lbl_img.image = tk_img
                    lbl_img.pack(side="top", padx=8, pady=8)
                except Exception as e:
                    tk.Label(win, text=f"Failed to load image: {e}").pack(side="top", padx=8, pady=8)

                caption_var = tk.StringVar(value="Generating...")
                lbl_caption = tk.Label(win, textvariable=caption_var, wraplength=880, justify="center", font=("Arial", 14))
                lbl_caption.pack(side="top", padx=12, pady=(0, 12))

                def do_close():
                    win.destroy()

                ctrl = tk.Frame(win)
                ctrl.pack(side="bottom", fill="x", pady=6)
                btn_close = tk.Button(ctrl, text="Close", command=do_close)
                btn_close.pack(side="right", padx=8)

                def on_speak_now():
                    val = caption_var.get()
                    if val and val != "Generating...":
                        speak_text(val, volume=args.volume)

                btn_speak = tk.Button(ctrl, text="Speak", command=on_speak_now)
                btn_speak.pack(side="right", padx=8)

                # schedule generation shortly so image shows immediately
                def generate_and_update():
                    try:
                        cap = generate(model, tokenizer, image_processor, device, img_path, max_length=args.max_length, num_beams=args.num_beams)
                        caption_var.set(cap)
                        if args.speak:
                            # speak after a small delay to allow caption label update
                            win.after(100, lambda: speak_text(cap, volume=args.volume))
                    except Exception as e:
                        caption_var.set(f"Generation failed: {e}")

                win.after(100, generate_and_update)
                # center window
                win.update_idletasks()
                w_root = win.winfo_width()
                h_root = win.winfo_height()
                x = (win.winfo_screenwidth() // 2) - (w_root // 2)
                y = (win.winfo_screenheight() // 2) - (h_root // 2)
                win.geometry(f"+{x}+{y}")
                win.mainloop()

            # run the GUI window (blocking until user closes) — this shows image first, then caption appears
            open_window_and_generate()
        else:
            # no GUI: show path, then generate, print caption and speak
            print(f"[Image: {img_path}]")
            try:
                cap = generate(model, tokenizer, image_processor, device, img_path, max_length=args.max_length, num_beams=args.num_beams)
                print("Caption:\n  " + cap + "\n")
                if args.speak:
                    speak_text(cap, volume=args.volume)
            except Exception as e:
                print(f"Failed to generate caption for {img_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
# ...existing code...
