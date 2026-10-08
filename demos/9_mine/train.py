#!/usr/bin/env python3
"""Train an SVTRv2 CTC recognizer from the Gaussian-blur manifest dataset."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import shutil
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import (TextImageDataset, Vocabulary, canonical_text, collate_samples,
                  load_manifest, make_splits, minimum_ctc_steps)
from model import ModelConfig, SVTRv2CTC


# SVTRv2 reports AdamW LR=6.5e-4 and weight_decay=0.05. Include lower rates
# and weaker decay for this CTC-only task. Batches are conservative for MPS;
# edit this dictionary for the available memory, rather than adding loss terms.
# Reference: https://arxiv.org/html/2411.15858v2#S4.SS1
param_grid: Dict[str, List[Any]] = {
    "learning_rate": [1e-4, 3.25e-4, 6.5e-4],
    "batch_size": [2, 4],
    "weight_decay": [0.02, 0.05],
}

# Edit to request fewer split rounds; values outside 1..5 are rejected.
SPLIT_ROUNDS = 2

# Lower CER wins; remaining metrics break exact ties in this fixed order.
SELECTION_POLICY = {"metric": "test_cer", "mode": "min",
                    "tie_breakers": ["test_wer:min", "test_exact_match:max", "test_loss:min", "trial:ascending"]}


def write_json(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def artifact_path(directory: Path, filename: str, split_round: int | None = None) -> Path:
    """Keep round identity in filenames, including after copying a winner."""
    path = Path(filename)
    suffix = f"_split_{split_round:02d}" if split_round is not None else ""
    return directory / f"{path.stem}{suffix}{path.suffix}"


def write_split_files(directory: Path, rows: list[dict], splits: dict[str, list[int]],
                      seed: int, train_eval_samples: int, split_round: int | None = None) -> list[int]:
    """Record membership and the fixed training probe without moving image files."""
    payload = {name: [rows[index]["sample_id"] for index in indices]
               for name, indices in splits.items()}
    eval_indices = random.Random(seed + 1).sample(
        splits["train"], min(train_eval_samples, len(splits["train"])))
    payload["train_eval"] = [rows[index]["sample_id"] for index in eval_indices]
    write_json(artifact_path(directory, "splits.json", split_round), payload)
    text_path = artifact_path(directory, "splits.txt", split_round)
    temporary = text_path.with_name(text_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["bucket", "sample_id", "after_path", "before_path"])
        for name in ("train", "val", "test"):
            for index in splits[name]:
                row = rows[index]
                writer.writerow(["validation" if name == "val" else name,
                                 row["sample_id"], row["after_path"], row.get("before_path", "")])
    temporary.replace(text_path)
    return eval_indices


def save_weights(model: nn.Module, path: Path, metadata: dict | None = None):
    # Keep floating-point parameters in FP32, including after mixed-precision training.
    tensors = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.float()
        tensors[name] = tensor.contiguous().clone()
    temporary = path.with_name(path.name + ".tmp")
    header = {"format": "pt", "architecture": "svtrv2_ctc_frm_v1"}
    if metadata is not None:
        # Safetensors headers accept strings; retain types in a JSON payload.
        header["training"] = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    # The .pth extension is a naming convention; the payload remains safetensors.
    save_file(tensors, str(temporary), metadata=header)
    temporary.replace(path)
    if metadata is not None:
        write_json(path.with_suffix(".json"), metadata)


@dataclass
class EarlyStopping:
    patience: int = 5
    min_epochs: int = 5
    warmup_epochs: int = 3
    min_delta: float = 1e-4
    best_loss: float = math.inf
    best_epoch: int = 0
    significant_best: float = math.inf
    bad_epochs: int = 0
    training_loss_at_best: float = math.inf
    overfitting_suspected: bool = False

    def update(self, value: float, epoch: int, training_loss: float) -> tuple[bool, bool]:
        """Select by validation loss; compare eval-mode training loss for overfitting."""
        if not math.isfinite(value) or not math.isfinite(training_loss):
            raise FloatingPointError("Validation or training-evaluation loss is not finite")
        new_best = value < self.best_loss
        if new_best:
            self.best_loss, self.best_epoch = value, epoch
            self.training_loss_at_best = training_loss
        # A widening gap alone is not proof of overfitting. Flag the specific
        # pattern of worse validation but better training than the best epoch.
        self.overfitting_suspected = (value > self.best_loss + self.min_delta
                                     and training_loss < self.training_loss_at_best - self.min_delta)
        if epoch <= self.warmup_epochs:
            self.significant_best = min(self.significant_best, value)
            self.bad_epochs = 0
        elif value < self.significant_best - self.min_delta:
            self.significant_best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        stop = epoch >= self.min_epochs and epoch > self.warmup_epochs and self.bad_epochs >= self.patience
        return new_best, stop


def edit_distance(reference, prediction) -> int:
    """Levenshtein distance for a character string or a list of words."""
    previous = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, pred_item in enumerate(prediction, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_item != pred_item)))
        previous = current
    return previous[-1]


def ctc_losses(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """Per-example, target-length-normalized CTC loss, computed in float32."""
    # PyTorch 2.10 has no native MPS CTC kernel. Keep the model on MPS,
    # but compute log-softmax and CTC on CPU without detaching the autograd graph.
    device = torch.device("cpu") if logits.device.type == "mps" else logits.device
    log_probs = logits.to(device=device, dtype=torch.float32).log_softmax(dim=-1).transpose(0, 1).contiguous()
    batch_size, time_steps, _ = logits.shape
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    target_lengths = batch["target_lengths"]  # CPU lengths for native CTC.
    # int64 targets select the regular PyTorch CTC path, not the restricted CuDNN path.
    losses = F.ctc_loss(log_probs, batch["targets"].to(device, non_blocking=True),
                        input_lengths, target_lengths, blank=0,
                        reduction="none", zero_infinity=False)
    losses = losses / target_lengths.to(device, dtype=torch.float32)
    if not torch.isfinite(losses).all():
        raise FloatingPointError(f"Non-finite CTC loss for samples {batch['sample_ids']}")
    return losses


def precision_context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate(model: SVTRv2CTC, loader: DataLoader, device: torch.device,
             vocabulary: Vocabulary, prediction_path: Path | None = None) -> dict:
    # Full FP32 evaluation makes the monitored objective consistent across epochs.
    model.eval()
    loss_sum = 0.0
    samples = char_errors = chars = word_errors = words = exact = 0
    handle = prediction_path.open("w", encoding="utf-8", newline="") if prediction_path else None
    writer = csv.DictWriter(handle, fieldnames=["sample_id", "reference", "prediction"]) if handle else None
    if writer:
        writer.writeheader()
    try:
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            logits = model(images)
            loss_sum += ctc_losses(logits, batch).sum().item()
            ids = logits.argmax(dim=-1).cpu().tolist()
            for sample_id, reference, sequence in zip(batch["sample_ids"], batch["texts"], ids):
                prediction = vocabulary.decode(sequence)
                char_errors += edit_distance(reference, prediction)
                chars += len(reference)
                word_errors += edit_distance(reference.split(), prediction.split())
                words += len(reference.split())
                exact += int(reference == prediction)
                samples += 1
                if writer:
                    writer.writerow({"sample_id": sample_id, "reference": reference, "prediction": prediction})
    finally:
        if handle:
            handle.close()
    if not samples:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": loss_sum / samples, "cer": char_errors / max(chars, 1),
            "wer": word_errors / max(words, 1), "exact_match": exact / samples,
            "samples": samples}


def train_epoch(model, loader, optimizer, device, precision, epoch, warmup_epochs,
                base_lr, gradient_clip, log_every) -> float:
    model.train()
    loss_sum, sample_count = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        if epoch <= warmup_epochs:
            progress = ((epoch - 1) * len(loader) + step) / (warmup_epochs * len(loader))
            for group in optimizer.param_groups:
                group["lr"] = base_lr * (0.1 + 0.9 * progress)
        images = batch["images"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, precision):
            logits = model(images)
        # CTC and log-softmax are intentionally outside the BF16 autocast region.
        losses = ctc_losses(logits, batch)
        loss = losses.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += losses.detach().sum().item()
        sample_count += len(batch["texts"])
        if log_every and step % log_every == 0:
            print(f"  epoch={epoch} batch={step}/{len(loader)} train_loss={loss_sum / sample_count:.5f}", flush=True)
    return loss_sum / sample_count


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def optimizer_for(model: nn.Module, learning_rate: float, weight_decay: float, device: torch.device):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 or name.endswith("char_query") else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=learning_rate, fused=device.type == "cuda")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default="./dataset", help="Directory containing manifest.csv")
    parser.add_argument("--run-dir", type=Path, default="./output", help="A new or empty output directory")
    parser.add_argument("--grid-search", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the parameter grid over SPLIT_ROUNDS (default); "
                             "--no-grid-search explicitly selects one single run")
    parser.add_argument("--model-size", choices=["compact", "reference"], default="compact")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs per model (1 through 50)")
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Single-run batch size; param_grid overrides this during grid search")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="FP32 evaluation can use more memory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Single-run learning rate; param_grid overrides this during grid search")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Single-run weight decay; param_grid overrides this during grid search")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.15,
                        help="Automatic validation fraction; test reserves 0.15, train takes the remainder")
    parser.add_argument("--group-column", help="Optional document/group ID column for related samples")
    parser.add_argument("--vocab-file", type=Path, help="JSON vocabulary; default is printable ASCII including space")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--train-eval-samples", type=int, default=256,
                        help="Fixed training subset evaluated in FP32/eval mode each epoch")
    parser.add_argument("--allow-small-dataset", action="store_true", help="Allow smoke tests with few groups/validation rows")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def train_run(args: argparse.Namespace, rows: list[dict] | None = None,
              splits: dict[str, list[int]] | None = None) -> dict:
    if not 1 <= args.min_epochs <= args.epochs <= 50 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("Require 1 <= min-epochs <= epochs <= 50 and 0 <= warmup-epochs < epochs")
    if min(args.batch_size, args.eval_batch_size, args.patience, args.train_eval_samples, args.cpu_threads) < 1:
        raise ValueError("Batch sizes, patience, train-eval-samples, and cpu-threads must be positive")
    if args.num_workers < 0 or args.lr_patience < 0 or args.log_every < 0 or args.min_delta < 0:
        raise ValueError("Worker count, lr-patience, log-every, and min-delta must be nonnegative")
    if not 0 < args.min_lr <= args.learning_rate or args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError("Invalid learning rate, weight decay, or gradient clipping value")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("run-dir is not empty; select a new directory to preserve previous checkpoints")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "mps" if args.device == "auto" and torch.backends.mps.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch environment; use --device cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training requires a supported CUDA device; use fp32 on CPU or MPS")

    rows = load_manifest(args.data_dir) if rows is None else rows
    vocabulary = Vocabulary.load(args.vocab_file) if args.vocab_file else Vocabulary()
    split_round = getattr(args, "split_round", None)
    split_seed = getattr(args, "split_seed", args.seed)
    splits = make_splits(rows, args.validation_fraction, split_seed, args.group_column) if splits is None else splits
    config = ModelConfig.preset(args.model_size, image_height=int(rows[0]["height"]),
                                image_width=int(rows[0]["width"]), num_classes=len(vocabulary),
                                dropout=args.dropout, drop_path=args.drop_path)
    for row in rows:
        vocabulary.encode(row["text"])
        needed = minimum_ctc_steps(row["text"])
        if needed > config.image_width // 4:
            raise ValueError(f"Sample {row['sample_id']} needs {needed} CTC steps, but the image provides "
                             f"{config.image_width // 4}; regenerate a wider image or shorten the sample")
    distinct = len({canonical_text(row["text"]) for row in rows})
    small_dataset = distinct < 20 or len(splits["val"]) < 10 or len(splits["test"]) < 10
    if small_dataset and not args.allow_small_dataset:
        raise ValueError(f"Only {distinct} distinct text groups, {len(splits['val'])} validation rows, "
                         f"and {len(splits['test'])} test rows. "
                         "Use --allow-small-dataset for a pipeline smoke test, not a quality estimate.")
    if small_dataset:
        warnings.warn("SMOKE TEST: this split is too small for a useful generalization estimate")
    train_chars = set("".join(rows[i]["text"] for i in splits["train"]))
    heldout_chars = set("".join(rows[i]["text"] for i in splits["val"] + splits["test"]))
    if heldout_chars - train_chars:
        warnings.warn(f"Held-out characters absent from training: {sorted(heldout_chars - train_chars)!r}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_path(args.run_dir, "config.json", split_round), config.to_dict())
    write_json(artifact_path(args.run_dir, "vocab.json", split_round), vocabulary.to_dict())
    eval_indices = write_split_files(args.run_dir, rows, splits, args.seed, args.train_eval_samples, split_round)
    split_json = artifact_path(args.run_dir, "splits.json", split_round)
    split_text = artifact_path(args.run_dir, "splits.txt", split_round)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "device": str(device), "precision": precision, "cuda": torch.version.cuda,
                   "ctc_device": "cpu" if device.type == "mps" else str(device),
                   "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    run_config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": environment,
        "manifest_sha256": hashlib.sha256((args.data_dir / "manifest.csv").read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).resolve().parent.glob("*.py")},
        "preprocessing": "native-size grayscale; float32; pixel/127.5 - 1",
        "image_column": "after_path", "label_column": "text", "smoke_test": small_dataset,
        "split_round": split_round, "split_seed": split_seed,
        "split_membership_sha256": hashlib.sha256(split_text.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_json.read_bytes()).hexdigest()}
    write_json(artifact_path(args.run_dir, "run_config.json", split_round), run_config)
    hyperparameters = {name: getattr(args, name) for name in
                       ("model_size", "learning_rate", "batch_size", "weight_decay", "dropout", "drop_path",
                        "gradient_clip", "epochs", "min_epochs", "patience", "min_delta", "warmup_epochs",
                        "lr_patience", "min_lr", "seed")}
    grid_params = {name: getattr(args, name) for name in
                   (param_grid if args.grid_search else ("learning_rate", "batch_size", "weight_decay"))}
    # Retain each numeric value's full string representation rather than rounding it.
    model_prefix = f"model_split_{split_round:02d}_" if split_round is not None else "model_"
    model_stem = model_prefix + "_".join(f"{name}_{value}" for name, value in grid_params.items())
    checkpoint_metadata = {"metadata_version": 3, "model_config": config.to_dict(),
                           "vocabulary": vocabulary.to_dict(), "hyperparameters": hyperparameters,
                           "grid_params": grid_params,
                           "run_config": run_config, "checkpoint_selection": "val_loss",
                           "search_selection": SELECTION_POLICY if args.grid_search else None}

    image_size = (config.image_width, config.image_height)
    generator = torch.Generator().manual_seed(args.seed)
    def loader(indices, training=False):
        return DataLoader(TextImageDataset(args.data_dir, rows, indices, vocabulary, image_size),
                          batch_size=args.batch_size if training else args.eval_batch_size,
                          shuffle=training, num_workers=args.num_workers,
                          collate_fn=collate_samples, worker_init_fn=seed_worker,
                          generator=generator if training else torch.Generator().manual_seed(args.seed + 2),
                          pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
                          drop_last=False)
    train_loader = loader(splits["train"], training=True)
    val_loader = loader(splits["val"])
    train_eval_loader = loader(eval_indices)
    model = SVTRv2CTC(config).to(device)
    optimizer = optimizer_for(model, args.learning_rate, args.weight_decay, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=args.min_lr)
    stopper = EarlyStopping(args.patience, args.min_epochs, args.warmup_epochs, args.min_delta)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device} precision={precision} parameters={parameters:,} image={image_size} "
          f"CTC_steps={model.time_steps} split_sizes="
          f"{ {key: len(value) for key, value in splits.items()} }", flush=True)
    if device.type == "mps":
        print("MPS training uses FP32; log-softmax and CTC loss run on CPU with gradients back to MPS.", flush=True)
    checkpoint_path = args.run_dir / f"{model_stem}.pth"
    best_path = args.run_dir / f"{model_stem}_BEST.pth"
    fields = ["epoch", "train_loss", "train_eval_loss", "val_loss", "val_cer", "val_wer",
              "val_exact_match", "lr", "bad_epochs", "best_epoch", "overfitting_suspected", "seconds"]
    stop_reason, completed_epochs = "max_epochs", 0
    try:
        with artifact_path(args.run_dir, "history.csv", split_round).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                started = time.monotonic()
                train_loss = train_epoch(model, train_loader, optimizer, device, precision, epoch,
                                         args.warmup_epochs, args.learning_rate, args.gradient_clip, args.log_every)
                train_metrics = evaluate(model, train_eval_loader, device, vocabulary)
                val_metrics = evaluate(model, val_loader, device, vocabulary)
                new_best, should_stop = stopper.update(val_metrics["loss"], epoch, train_metrics["loss"])
                epoch_metadata = {**checkpoint_metadata, "epoch": epoch, "validation": val_metrics,
                                  "training": train_metrics, "train_loss": train_loss,
                                  "early_stopping": {"bad_epochs": stopper.bad_epochs,
                                                     "patience": args.patience,
                                                     "training_loss_at_best": stopper.training_loss_at_best,
                                                     "overfitting_suspected": stopper.overfitting_suspected},
                                  "completed_epochs": epoch, "stop_reason": "running"}
                # Archive every validated epoch before updating the convenience
                # latest/BEST files. Earlier versions are never overwritten.
                save_weights(model, args.run_dir / f"{model_stem}_epoch_{epoch:04d}.pth",
                             {**epoch_metadata, "checkpoint_selection": "epoch"})
                save_weights(model, checkpoint_path,
                             {**epoch_metadata, "checkpoint_selection": "latest_epoch"})
                if new_best:
                    save_weights(model, best_path, epoch_metadata)
                    write_json(artifact_path(args.run_dir, "best.json", split_round),
                               {"epoch": epoch, "monitor": "val_loss", **val_metrics})
                lr_used = optimizer.param_groups[0]["lr"]
                if epoch > args.warmup_epochs:
                    scheduler.step(val_metrics["loss"])
                record = {"epoch": epoch, "train_loss": train_loss, "train_eval_loss": train_metrics["loss"],
                          "val_loss": val_metrics["loss"], "val_cer": val_metrics["cer"],
                          "val_wer": val_metrics["wer"], "val_exact_match": val_metrics["exact_match"],
                          "lr": lr_used, "bad_epochs": stopper.bad_epochs, "best_epoch": stopper.best_epoch,
                          "overfitting_suspected": stopper.overfitting_suspected,
                          "seconds": time.monotonic() - started}
                writer.writerow(record)
                handle.flush()
                completed_epochs = epoch
                print(f"epoch={epoch:03d} train={train_loss:.5f} train_eval={train_metrics['loss']:.5f} "
                      f"val={val_metrics['loss']:.5f} CER={val_metrics['cer']:.3f} "
                      f"exact={val_metrics['exact_match']:.3f} lr={lr_used:.2e} "
                      f"stale={stopper.bad_epochs}/{args.patience}"
                      f"{' [saved best]' if new_best else ''}", flush=True)
                if should_stop:
                    stop_reason = "early_stopping"
                    detail = ("Validation worsened while training loss improved; overfitting suspected."
                              if stopper.overfitting_suspected else "Validation loss has not improved sufficiently.")
                    print(f"{detail} Patience exhausted; restoring the lowest-validation-loss epoch.", flush=True)
                    break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("Training interrupted; preserving the best completed validation checkpoint.", flush=True)
    # Epoch archives are authoritative even when Ctrl+C interrupts a sidecar or
    # latest/BEST update. Repair metadata from their headers, never partial weights.
    archived = []
    for path in sorted(args.run_dir.glob(f"{model_stem}_epoch_*.pth")):
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            metadata = json.loads(checkpoint.metadata()["training"])
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file() or json.loads(sidecar.read_text(encoding="utf-8")) != metadata:
            write_json(sidecar, metadata)
        archived.append((path, metadata))
    if not archived:
        raise RuntimeError("No epoch completed validation; no best model is available")
    latest_archive, latest_metadata = max(archived, key=lambda item: item[1]["epoch"])
    best_archive, selected_metadata = min(archived, key=lambda item: (item[1]["validation"]["loss"],
                                                                    item[1]["epoch"]))
    completed_epochs = latest_metadata["epoch"]
    early_stopping_reason = ("overfitting_suspected" if latest_metadata["early_stopping"]["overfitting_suspected"]
                             else "validation_not_improving") if stop_reason == "early_stopping" else None
    model.load_state_dict(load_file(str(latest_archive), device="cpu"), strict=True)
    save_weights(model, checkpoint_path,
                 {**latest_metadata, "checkpoint_selection": "latest_epoch", "stop_reason": stop_reason,
                  "early_stopping_reason": early_stopping_reason, "completed_epochs": completed_epochs})
    # Reload the selected epoch, not the potentially overfit final epoch.
    model.load_state_dict(load_file(str(best_archive), device="cpu"), strict=True)
    selected_epoch = selected_metadata["epoch"]
    best_validation = evaluate(model, val_loader, device, vocabulary,
                               artifact_path(args.run_dir, "validation_predictions.csv", split_round))
    summary = {"stop_reason": stop_reason, "early_stopping_reason": early_stopping_reason,
               "completed_epochs": completed_epochs, "epoch_checkpoints": len(archived),
               "best_epoch": selected_epoch, "best_validation": best_validation,
               "parameters": parameters, "smoke_test": small_dataset,
               "hyperparameters": hyperparameters, "grid_params": grid_params,
               "split_round": run_config["split_round"], "split_seed": split_seed,
               "split_sha256": run_config["split_sha256"]}
    # Evaluate the reloaded validation-best epoch, never the last training epoch.
    # Across a grid, this test set is a selection set, not an unbiased final holdout.
    summary["test"] = evaluate(model, loader(splits["test"]), device, vocabulary,
                               artifact_path(args.run_dir, "test_predictions.csv", split_round))
    save_weights(model, best_path,
                 {**selected_metadata, "checkpoint_selection": "val_loss", "validation": best_validation,
                  "test": summary["test"], "stop_reason": stop_reason,
                  "early_stopping_reason": early_stopping_reason, "completed_epochs": completed_epochs})
    write_json(artifact_path(args.run_dir, "best.json", split_round),
               {"epoch": selected_epoch, "monitor": "val_loss", **best_validation})
    write_json(artifact_path(args.run_dir, "summary.json", split_round), summary)
    print(f"Selected epoch {selected_epoch}. Test CER={summary['test']['cer']:.5f}. "
          f"Portable checkpoint: {best_path}", flush=True)
    return summary


def copy_best_artifacts(source_dir: Path, destination_dir: Path) -> list[str]:
    """Copy one trial's selected bundle, excluding latest and epoch archives."""
    copied = []
    for source in sorted(source_dir.iterdir()):
        if source.is_file() and source.suffix in (".pth", ".json", ".csv", ".txt"):
            if source.name.startswith("model_") and not source.stem.endswith("_BEST"):
                continue
            destination = destination_dir / f"{source.stem}_OVERALL{source.suffix}"
            shutil.copy2(source, destination)
            copied.append(destination.name)
    return copied


def run_grid_search(args: argparse.Namespace) -> dict:
    """Run each frozen split's grid, publish its winner, then start the next round."""
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("run-dir is not empty; select a new directory to preserve previous checkpoints")
    if isinstance(SPLIT_ROUNDS, bool) or not isinstance(SPLIT_ROUNDS, int) or not 1 <= SPLIT_ROUNDS <= 5:
        raise ValueError("SPLIT_ROUNDS must be an integer from 1 through 5")
    allowed = {"learning_rate", "batch_size", "weight_decay", "dropout", "drop_path"}
    if not param_grid or set(param_grid) - allowed:
        raise ValueError(f"param_grid must use supported training parameters: {sorted(allowed)}")
    if any(not isinstance(values, list) or not values for values in param_grid.values()):
        raise ValueError("Each param_grid entry must be a non-empty list")
    candidates = [dict(zip(param_grid, values)) for values in product(*param_grid.values())]
    if len({json.dumps(candidate, sort_keys=True, allow_nan=False) for candidate in candidates}) != len(candidates):
        raise ValueError("param_grid contains duplicate parameter combinations")
    # Validate numeric grid entries before allocating models or creating trial directories.
    for candidate in candidates:
        for name, value in candidate.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid param_grid value for {name}: {value!r}")
            if name == "batch_size" and (not isinstance(value, int) or value < 1):
                raise ValueError("Grid batch_size values must be positive integers")
            if name == "learning_rate" and not 0 < args.min_lr <= value:
                raise ValueError("Grid learning_rate values must be positive and at least min-lr")
            if name == "weight_decay" and value < 0:
                raise ValueError("Grid weight_decay values must be nonnegative")
            if name in ("dropout", "drop_path") and not 0 <= value < 1:
                raise ValueError("Grid dropout and drop_path values must be in [0, 1)")

    rows = load_manifest(args.data_dir)
    # Vary membership, not fractions or training seeds. Explicit assignments remain
    # authoritative. Try at most five seeds; do not train duplicate partitions.
    explicit_splits = any(row.get("split", "").strip() for row in rows)
    split_plans, skipped_rounds, seen_splits = [], [], {}
    for split_round in range(1, (1 if explicit_splits else SPLIT_ROUNDS) + 1):
        split_seed = args.seed + split_round - 1
        splits = make_splits(rows, args.validation_fraction, split_seed, args.group_column)
        membership = tuple(tuple(splits[name]) for name in ("train", "val", "test"))
        if membership in seen_splits:
            skipped_rounds.append({"split_round": split_round, "split_seed": split_seed,
                                   "reason": "duplicate_membership",
                                   "matches_split_round": seen_splits[membership]})
            continue
        seen_splits[membership] = split_round
        split_plans.append({"split_round": split_round, "split_seed": split_seed, "splits": splits})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rounds = []
    # The root text file lists all scheduled rounds. Each round and trial also
    # retains its own membership files, so copied models can be traced to inputs.
    with (args.run_dir / "splits.txt").open("w", encoding="utf-8", newline="") as handle:
        membership_writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        membership_writer.writerow(["split_round", "split_seed", "bucket", "sample_id", "after_path", "before_path"])
        for plan in split_plans:
            directory = args.run_dir / f"split_{plan['split_round']:02d}"
            directory.mkdir()
            write_split_files(directory, rows, plan["splits"], args.seed, args.train_eval_samples,
                              plan["split_round"])
            split_json = artifact_path(directory, "splits.json", plan["split_round"])
            split_text = artifact_path(directory, "splits.txt", plan["split_round"])
            rounds.append({"split_round": plan["split_round"], "split_seed": plan["split_seed"],
                           "directory": directory.name,
                           "split_files": {"json": str(split_json.relative_to(args.run_dir)),
                                           "text": str(split_text.relative_to(args.run_dir))},
                           "split_sha256": hashlib.sha256(split_json.read_bytes()).hexdigest(),
                           "split_membership_sha256": hashlib.sha256(split_text.read_bytes()).hexdigest(),
                           "sizes": {name: len(indices) for name, indices in plan["splits"].items()}})
            with split_text.open(encoding="utf-8", newline="") as source:
                reader = csv.reader(source, delimiter="\t")
                next(reader)  # Round-local header; the root adds round and seed columns.
                for row in reader:
                    membership_writer.writerow([plan["split_round"], plan["split_seed"], *row])
    write_json(args.run_dir / "splits.json", {"rounds": rounds, "skipped_rounds": skipped_rounds})
    report = {"status": "running", "param_grid": param_grid,
              "split_rounds_requested": SPLIT_ROUNDS, "split_rounds": len(rounds),
              "completed_split_rounds": 0, "rounds": rounds, "skipped_rounds": skipped_rounds,
              "split_method": "explicit_manifest" if explicit_splits else "repeated_holdout",
              "trials_per_split": len(candidates), "total_trials": len(rounds) * len(candidates),
              "selection_policy": SELECTION_POLICY, "test_used_for_selection": True,
              "generalization_note": "Test data selects hyperparameters and split rounds; use a separate untouched holdout for final reporting.",
              "results": [], "round_results": [], "best_trial": None, "best_params": None,
              "best_split_round": None, "best_split_seed": None,
              "best_test": None, "best_files": []}
    report_path = args.run_dir / "grid_search.json"
    write_json(report_path, report)
    warnings.warn(report["generalization_note"])
    if explicit_splits:
        print("Explicit manifest partitions: running one grid without reshuffling assignments.", flush=True)
    for skipped in skipped_rounds:
        print(f"Skipping split round {skipped['split_round']} (seed={skipped['split_seed']}): "
              f"same membership as round {skipped['matches_split_round']}.", flush=True)
    print(f"Grid search: {len(rounds)} split rounds x {len(candidates)} combinations = "
          f"{report['total_trials']} sequential trials; values={param_grid}. "
          "Matching single-run parameter flags are overridden by the grid.", flush=True)
    fields = ["trial", "split_round", "split_seed", "split_sha256", *param_grid,
              "best_epoch", "completed_epochs", "stop_reason", "smoke_test", "val_loss",
              "test_loss", "test_cer", "test_wer", "test_exact_match", "test_samples"]
    current_trial = None
    winner, winner_score = None, None
    try:
        with (args.run_dir / "grid_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            for plan, round_info in zip(split_plans, rounds):
                round_winner, round_winner_score = None, None
                for index, parameters in enumerate(candidates, start=1):
                    current_trial = f"{round_info['directory']}/trial_{index:04d}"
                    trial_args = argparse.Namespace(**{**vars(args), **parameters, "grid_search": True,
                                                       "split_round": plan["split_round"],
                                                       "split_seed": plan["split_seed"],
                                                       "run_dir": args.run_dir / current_trial})
                    print(f"\n[{len(report['results']) + 1}/{report['total_trials']}] "
                          f"{current_trial}: split_seed={plan['split_seed']} {parameters}", flush=True)
                    # Each trial resets training state; all combinations in this
                    # round receive the exact same rows and frozen split object.
                    try:
                        summary = train_run(trial_args, rows=rows, splits=plan["splits"])
                    finally:
                        gc.collect()
                        if args.device in ("auto", "cuda") and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if args.device in ("auto", "mps") and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    result = {"trial": current_trial, "params": parameters, **summary,
                              "split_round": plan["split_round"], "split_seed": plan["split_seed"],
                              "split_sha256": round_info["split_sha256"]}
                    report["results"].append(result)
                    writer.writerow({"trial": current_trial, **parameters,
                                     **{name: result[name] for name in
                                        ("split_round", "split_seed", "split_sha256", "best_epoch",
                                         "completed_epochs", "stop_reason", "smoke_test")},
                                     "val_loss": summary["best_validation"]["loss"],
                                     **{f"test_{name}": value for name, value in summary["test"].items()}})
                    handle.flush()
                    if summary["stop_reason"] == "interrupted":
                        report["status"] = "interrupted"
                        write_json(report_path, report)
                        print("Grid interrupted; completed round winners are preserved. "
                              "No search-root _OVERALL bundle was created.", flush=True)
                        return report
                    # Compare against every completed trial, including earlier
                    # split rounds; keep the existing test-CER tie-breakers.
                    score = (result["test"]["cer"], result["test"]["wer"], -result["test"]["exact_match"],
                             result["test"]["loss"], result["trial"])
                    if round_winner is None or score < round_winner_score:
                        round_winner, round_winner_score = result, score
                    if winner is None or score < winner_score:
                        winner, winner_score = result, score
                        report.update({"best_trial": winner["trial"], "best_params": winner["params"],
                                       "best_test": winner["test"], "best_split_round": winner["split_round"],
                                       "best_split_seed": winner["split_seed"]})
                        print(f"Running grid best: {winner['trial']} test_CER={winner['test']['cer']:.5f}", flush=True)
                    write_json(report_path, report)
                # Complete this split's selection and copies before starting a
                # fresh model on the next split. Earlier round winners stay intact.
                round_dir = args.run_dir / round_info["directory"]
                round_result = {"split_round": plan["split_round"], "split_seed": plan["split_seed"],
                                "status": "completed", "completed_trials": len(candidates),
                                "selection_policy": SELECTION_POLICY,
                                "best_trial": round_winner["trial"], "best_params": round_winner["params"],
                                "best_test": round_winner["test"],
                                "best_files": copy_best_artifacts(args.run_dir / round_winner["trial"], round_dir)}
                write_json(artifact_path(round_dir, "round_summary.json", plan["split_round"]), round_result)
                report["round_results"].append(round_result)
                report["completed_split_rounds"] += 1
                write_json(report_path, report)
                print(f"Split {plan['split_round']} winner: {round_winner['trial']} "
                      f"test_CER={round_winner['test']['cer']:.5f}; _OVERALL copies saved in {round_dir}", flush=True)
        # Preserve the existing search-wide winner in addition to per-round winners.
        copied = copy_best_artifacts(args.run_dir / winner["trial"], args.run_dir)
        report.update({"status": "completed", "best_trial": winner["trial"],
                       "best_params": winner["params"], "best_test": winner["test"], "best_files": copied})
        write_json(report_path, report)
        print(f"\nGrid winner: {winner['trial']} test_CER={winner['test']['cer']:.5f}; "
              f"parameters={winner['params']}. Copied _OVERALL artifacts to {args.run_dir}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "failed_trial": current_trial, "error": f"{type(error).__name__}: {error}"})
        write_json(report_path, report)
        raise  # Do not hide OOM, invalid CTC losses, or other failed trials.
    return report


def main():
    args = parse_args()
    if args.grid_search:
        print(f"Execution mode: GRID SEARCH ({SPLIT_ROUNDS} requested split rounds).", flush=True)
        return run_grid_search(args)
    print("Execution mode: SINGLE RUN (--no-grid-search); parameter grid and outer loop are disabled.",
          flush=True)
    return train_run(args)


if __name__ == "__main__":
    main()
