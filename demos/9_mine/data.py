"""Manifest loading, leakage checks, character encoding, and image preparation."""
from __future__ import annotations

import csv
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def canonical_text(text: str) -> str:
    """Normalize ONLY the split key. The actual training label stays unchanged."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def minimum_ctc_steps(text: str) -> int:
    # Adjacent equal characters need an intervening CTC blank.
    return len(text) + sum(left == right for left, right in zip(text, text[1:]))


class Vocabulary:
    blank_id = 0

    def __init__(self, characters: list[str] | None = None):
        self.characters = characters if characters is not None else [chr(i) for i in range(32, 127)]
        if not self.characters or any(len(char) != 1 for char in self.characters):
            raise ValueError("Vocabulary must contain individual characters")
        if len(self.characters) != len(set(self.characters)):
            raise ValueError("Vocabulary contains duplicate characters")
        self.char_to_id = {char: index + 1 for index, char in enumerate(self.characters)}

    def __len__(self):
        return len(self.characters) + 1

    def encode(self, text: str) -> torch.Tensor:
        unknown = sorted(set(text) - self.char_to_id.keys())
        if unknown:
            raise ValueError(f"Characters missing from vocabulary: {unknown!r}")
        return torch.tensor([self.char_to_id[char] for char in text], dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        result, previous = [], self.blank_id
        for index in ids:
            # Collapse consecutive repeats BEFORE removing blank tokens.
            if index != self.blank_id and index != previous:
                result.append(self.characters[index - 1])
            previous = index
        return "".join(result)

    def to_dict(self) -> dict:
        return {"blank_id": self.blank_id, "characters": self.characters}

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["blank_id"] != 0:
            raise ValueError("This pipeline reserves class 0 for CTC blank")
        return cls(payload["characters"])


def dataset_path(root: Path, relative: str) -> Path:
    """Keep manifest paths within the supplied dataset directory."""
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a relative image path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes the dataset directory: {relative!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest(root: Path) -> list[dict]:
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "text", "after_path", "width", "height"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest.csv must contain columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest.csv contains no samples")
    seen_ids, seen_paths = set(), set()
    sizes = set()
    for row in rows:
        if not row["sample_id"] or row["sample_id"] in seen_ids:
            raise ValueError(f"Missing or duplicate sample_id: {row['sample_id']!r}")
        if not row["text"] or not row["text"].strip():
            raise ValueError(f"Empty label in sample {row['sample_id']}")
        if any(unicodedata.category(char).startswith("C") for char in row["text"]):
            raise ValueError(f"Control characters are not supported: {row['sample_id']}")
        path = dataset_path(root, row["after_path"])
        if path in seen_paths:
            raise ValueError(f"An after image is referenced more than once: {path}")
        seen_paths.add(path)
        seen_ids.add(row["sample_id"])
        size = (int(row["width"]), int(row["height"]))
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"Image size disagrees with manifest: {path}")
            image.verify()
        sizes.add(size)
    if len(sizes) != 1:
        raise ValueError("All images must use the same size; no implicit resizing is performed")
    return rows


def make_splits(rows: list[dict], validation_fraction: float, seed: int,
                group_column: str | None = None, test_fraction: float = 0.15) -> dict[str, list[int]]:
    """Preserve explicit partitions, otherwise target row fractions using whole groups."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    splits = {"train": [], "val": [], "test": []}
    explicit = [bool(row.get("split", "").strip()) for row in rows]
    if any(explicit):
        if not all(explicit):
            raise ValueError("The split column must be populated for every row")
        for index, row in enumerate(rows):
            split = row["split"].strip().lower()
            if split not in splits:
                raise ValueError("split values must be train, val, or test")
            splits[split].append(index)
    else:
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            key = row.get(group_column, "").strip() if group_column else canonical_text(row["text"])
            if not key:
                raise ValueError(f"Missing grouping value for sample {row['sample_id']}")
            groups[key].append(index)
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError("At least three distinct groups are required for train/val/test splitting")
        random.Random(seed).shuffle(keys)
        # Place larger groups first; the seeded shuffle breaks equal-size ties.
        # Target sample counts, not group counts, without splitting duplicate text/documents.
        keys.sort(key=lambda key: len(groups[key]), reverse=True)
        targets = {"train": len(rows) * (1 - validation_fraction - test_fraction),
                   "val": len(rows) * validation_fraction, "test": len(rows) * test_fraction}
        for position, key in enumerate(keys):
            empty = [name for name in splits if not splits[name]]
            # Reserve enough groups to keep every partition non-empty on small data.
            candidates = empty if len(keys) - position == len(empty) else splits
            split = max(candidates, key=lambda name: targets[name] - len(splits[name]))
            splits[split].extend(groups[key])
    if any(not indices for indices in splits.values()):
        raise ValueError("Training, validation, and test splits must all be non-empty")
    # Check text leakage even when the split or document grouping was user supplied.
    owners = {}
    for split, indices in splits.items():
        for index in indices:
            row = rows[index]
            keys = [("text", canonical_text(row["text"]))]
            if group_column:
                group = row.get(group_column, "").strip()
                if not group:
                    raise ValueError(f"Missing group column {group_column!r}")
                keys.append(("group", group))
            for key in keys:
                if key in owners and owners[key] != split:
                    raise ValueError(f"Cross-split leakage detected for {key!r}; revise grouping/splits")
                owners[key] = split
    return {split: sorted(indices) for split, indices in splits.items()}


def image_to_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Keep native geometry, convert to grayscale, normalize to [-1, 1]."""
    with Image.open(path) as image:
        if image.size != image_size:
            raise ValueError(f"Expected image size {image_size}, got {image.size}: {path}")
        pixels = np.array(image.convert("L"), dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).unsqueeze(0).div_(127.5).sub_(1.0)


class TextImageDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], indices: list[int],
                 vocabulary: Vocabulary, image_size: tuple[int, int]):
        self.root, self.rows, self.indices = root, rows, indices
        self.vocabulary, self.image_size = vocabulary, image_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        row = self.rows[self.indices[index]]
        image = image_to_tensor(dataset_path(self.root, row["after_path"]), self.image_size)
        return image, self.vocabulary.encode(row["text"]), row["text"], row["sample_id"]


def collate_samples(batch):
    images, targets, texts, sample_ids = zip(*batch)
    return {"images": torch.stack(images), "targets": torch.cat(targets),
            "target_lengths": torch.tensor([len(target) for target in targets], dtype=torch.long),
            "texts": list(texts), "sample_ids": list(sample_ids)}
