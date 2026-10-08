#!/usr/bin/env python3
"""Load a portable checkpoint and recognize an image using CPU, MPS, or CUDA."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from data import Vocabulary, image_to_tensor
from model import ModelConfig, SVTRv2CTC


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this environment; choose --device cpu")
    return torch.device(requested)


def load_recognizer(model_dir: Path, device: torch.device):
    # Prefer the overall grid winner, then a trial's validation-best model.
    # Always use the corresponding config/vocabulary bundle, never a mix.
    checkpoints = sorted(model_dir.glob("model_*_BEST_OVERALL.pth"))
    suffix = "_OVERALL"
    if not checkpoints:
        checkpoints = sorted(model_dir.glob("model_*_BEST.pth"))
        suffix = ""
    if checkpoints:
        if len(checkpoints) != 1:
            raise ValueError(f"Multiple selected checkpoints in {model_dir}; use one run's directory")
        checkpoint_path = checkpoints[0]
        # New bundles retain the split number in both weights and companion filenames.
        split_match = re.match(r"model_split_(\d+)_", checkpoint_path.name)
        if split_match:
            suffix = f"_split_{split_match[1]}{suffix}"
    else:
        # Retain support for checkpoints written before parameterized filenames.
        suffix = "_BEST" if (model_dir / "best_BEST.safetensors").is_file() else ""
        checkpoint_path = model_dir / f"best{suffix}.safetensors"
    config = ModelConfig(**json.loads((model_dir / f"config{suffix}.json").read_text(encoding="utf-8")))
    vocabulary = Vocabulary.load(model_dir / f"vocab{suffix}.json")
    if config.num_classes != len(vocabulary):
        raise ValueError("Model and vocabulary class counts differ")
    model = SVTRv2CTC(config)
    # Load on CPU first. The weights are not tied to the training CUDA device.
    # Parameterized .pth files contain safetensors, not torch.save/pickle data.
    weights = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(weights, strict=True)
    model = model.float().to(device).eval()
    return model, vocabulary, config


@torch.inference_mode()
def predict_image(model: SVTRv2CTC, vocabulary: Vocabulary, path: Path, device: torch.device) -> str:
    size = (model.config.image_width, model.config.image_height)
    image = image_to_tensor(path, size).unsqueeze(0).to(device)
    ids = model(image).argmax(dim=-1)[0].cpu().tolist()
    return vocabulary.decode(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default="./output-v2/split_01/trial_0001")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    model, vocabulary, _ = load_recognizer(args.model_dir, device)
    text = predict_image(model, vocabulary, args.image, device)
    print(json.dumps({"image": str(args.image), "device": str(device), "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
