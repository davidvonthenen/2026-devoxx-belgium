#!/usr/bin/env python3
"""Create paired clear/blurred text images from a text file.

Each non-empty input line containing only printable ASCII becomes one sample:

    before/<sample_id>.png  # clear rendered text before redaction
    after/<sample_id>.png   # Gaussian-blurred image after redaction

The matching text label and relative image paths are written to manifest.csv.
The images are single-line and fixed-size, which matches the left-to-right
sequence assumption used by SVTRv2 + CTC text-recognition models. Lines with
non-ASCII or control characters are logged in skipped.csv without rendering.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_BLUR_RADIUS = 8.0


DEFAULT_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/Library/Fonts/Arial.ttf",                          # macOS
    "C:/Windows/Fonts/arial.ttf",                        # Windows
)


def read_sentences(input_path: Path) -> Iterator[tuple[int, str]]:
    """Yield non-empty lines, preserving unsupported characters for validation."""
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            # Remove line endings and ordinary padding, not Unicode whitespace
            # or tabs that must cause the whole sample to be skipped.
            text = raw_line.rstrip("\r\n").strip(" ")
            if text:
                yield line_number, text


def resolve_font_path(font_path: Path | None) -> str:
    """Return a usable TrueType/OpenType font path."""
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file does not exist: {font_path}")
        return str(font_path)

    for candidate in DEFAULT_FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    raise FileNotFoundError(
        "No default font was found. Pass a TrueType/OpenType font with --font."
    )


@lru_cache(maxsize=128)
def load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Load and cache fonts because many samples reuse the same sizes."""
    return ImageFont.truetype(font_path, font_size)


def text_to_uniform_image(
    text: str,
    *,
    width: int,
    height: int,
    font_path: str,
    max_font_size: int,
    min_font_size: int,
    margin: int,
) -> Image.Image:
    """Render one line of text into a fixed-size grayscale image.

    The font size is reduced until the full label fits. The function never
    truncates or wraps text because doing so would make the image disagree
    with its CTC label or introduce a multi-line reading-order problem.
    """
    if not text:
        raise ValueError("Text must not be empty")

    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError("Image dimensions must be larger than twice the margin")

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    selected_font: ImageFont.FreeTypeFont | None = None
    selected_bbox: tuple[int, int, int, int] | None = None

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width <= width - (2 * margin) and text_height <= height - (2 * margin):
            selected_font = font
            selected_bbox = bbox
            break

    if selected_font is None or selected_bbox is None:
        raise ValueError(
            f"Text does not fit at the minimum font size of {min_font_size}: {text!r}"
        )

    text_height = selected_bbox[3] - selected_bbox[1]

    # Keep every sample left-aligned and vertically centered. Subtracting the
    # bounding-box offsets accounts for ascenders, descenders, and punctuation.
    x = margin - selected_bbox[0]
    y = ((height - text_height) // 2) - selected_bbox[1]

    draw.text((x, y), text, fill=0, font=selected_font)
    return image


def apply_gaussian_blur(image: Image.Image, blur_radius: float) -> Image.Image:
    """Apply the dataset's Gaussian-blur redaction transform."""
    if blur_radius < 0:
        raise ValueError("Blur radius must be zero or greater")
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def save_before_after(
    before_image: Image.Image,
    after_image: Image.Image,
    *,
    output_dir: Path,
    sample_id: str,
) -> tuple[Path, Path]:
    """Save the clear image and its blurred counterpart as lossless PNG files."""
    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    before_path = before_dir / f"{sample_id}.png"
    after_path = after_dir / f"{sample_id}.png"

    before_image.save(before_path, format="PNG")
    after_image.save(after_path, format="PNG")
    return before_path, after_path


def relative_posix_path(path: Path, parent: Path) -> str:
    """Return a portable relative path for the CSV manifest."""
    return path.relative_to(parent).as_posix()


def create_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    font_path: str,
    max_font_size: int,
    min_font_size: int,
    margin: int,
    blur_radius: float,
) -> tuple[int, int]:
    """Create all image pairs and return ``(created_count, skipped_count)``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    skipped_path = output_dir / "skipped.csv"

    created_count = 0
    skipped_count = 0

    with (
        manifest_path.open("w", encoding="utf-8", newline="") as manifest_file,
        skipped_path.open("w", encoding="utf-8", newline="") as skipped_file,
    ):
        manifest_writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "sample_id",
                "source_line",
                "text",
                "before_path",
                "after_path",
                "width",
                "height",
                "blur_radius",
            ),
        )
        skipped_writer = csv.DictWriter(
            skipped_file,
            fieldnames=("source_line", "text", "reason"),
        )
        manifest_writer.writeheader()
        skipped_writer.writeheader()

        for source_line, text in read_sentences(input_path):
            # IDs are tied to the original line number, so skipped records do not
            # change the identity of later samples.
            sample_id = f"{source_line:08d}"

            try:
                # Match the training vocabulary before rendering or saving images.
                for character in text:
                    if not 32 <= ord(character) <= 126:
                        raise ValueError(
                            f"Unsupported character {character!r} (U+{ord(character):04X}); "
                            "only printable ASCII (U+0020-U+007E) is supported"
                        )
                before_image = text_to_uniform_image(
                    text,
                    width=width,
                    height=height,
                    font_path=font_path,
                    max_font_size=max_font_size,
                    min_font_size=min_font_size,
                    margin=margin,
                )
                after_image = apply_gaussian_blur(before_image, blur_radius)
                before_path, after_path = save_before_after(
                    before_image,
                    after_image,
                    output_dir=output_dir,
                    sample_id=sample_id,
                )
            except (OSError, ValueError) as error:
                skipped_writer.writerow(
                    {
                        "source_line": source_line,
                        "text": text,
                        "reason": str(error),
                    }
                )
                skipped_count += 1
                continue

            manifest_writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_line": source_line,
                    "text": text,
                    "before_path": relative_posix_path(before_path, output_dir),
                    "after_path": relative_posix_path(after_path, output_dir),
                    "width": width,
                    "height": height,
                    "blur_radius": blur_radius,
                }
            )
            created_count += 1

    return created_count, skipped_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clear/blurred text-image pairs for SVTRv2 + CTC."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="flatten.txt",
        help="UTF-8 text file containing one sentence per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./dataset",
        help="Directory in which to create before/, after/, and CSV files.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default="arial.ttf",
        help="Optional path to a .ttf or .otf font file.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max-font-size", type=int, default=40)
    parser.add_argument("--min-font-size", type=int, default=14)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=DEFAULT_BLUR_RADIUS,
        help=(
            "Gaussian blur radius used for every sample. Increase this value "
            f"to make the generated text harder to read (default: {DEFAULT_BLUR_RADIUS})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input text file does not exist: {args.input}")
    if args.max_font_size < args.min_font_size:
        raise ValueError("--max-font-size must be greater than or equal to --min-font-size")

    font_path = resolve_font_path(args.font)
    created_count, skipped_count = create_dataset(
        input_path=args.input,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        font_path=font_path,
        max_font_size=args.max_font_size,
        min_font_size=args.min_font_size,
        margin=args.margin,
        blur_radius=args.blur_radius,
    )

    print(f"Created {created_count} samples in: {args.output}")
    print(f"Skipped {skipped_count} samples; details: {args.output / 'skipped.csv'}")
    print(f"Manifest: {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
