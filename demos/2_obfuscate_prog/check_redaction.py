#!/usr/bin/env python3
# pip3 install PyMuPDF

import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF


def check_improper_redaction(pdf_path):
    """Detect if text exists under black rectangles."""
    try:
        doc = fitz.open(pdf_path)
        improper_redactions = []
        full_text = []

        for page_num, page in enumerate(doc):
            # Get all text with coordinates
            text_instances = page.get_text("dict")

            # Get all drawing operations (rectangles)
            drawings = page.get_drawings()

            # Find black-filled rectangles
            black_rects = []
            for drawing in drawings:
                if drawing.get("fill"):  # Has fill color
                    fill_color = drawing.get("fill")
                    # Check if black or very dark (RGB close to 0)
                    if all(c < 0.1 for c in fill_color[:3]):
                        black_rects.append(drawing["rect"])

            # Extract full page text for context
            full_text.append(page.get_text())

            # Check if any text falls within black rectangles
            for block in text_instances.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                bbox = fitz.Rect(span["bbox"])

                                # Check overlap with black rectangles
                                for rect in black_rects:
                                    if bbox.intersects(rect):
                                        improper_redactions.append(text)

        doc.close()

        if improper_redactions:
            # Save extracted improperly redacted text
            output_path = pdf_path.parent / f"{pdf_path.stem}-unredacted.txt"
            with open(output_path, 'w') as f:
                f.write(f"REDACTED WORDS: {improper_redactions}\n\n")
                f.write("FULL TEXT:\n")
                f.write("\n".join(full_text))

            print(f"⚠️  {pdf_path.name}: Found {len(improper_redactions)} improper redactions → {output_path.name}")
            return True
        else:
            print(f"✓ {pdf_path.name}: No improper redactions detected")
            return False

    except Exception as e:
        print(f"✗ Error processing {pdf_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Scan PDF files for improper redactions (text under black rectangles)'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default=None,
        help='Directory to scan (default: script location)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively scan subdirectories'
    )

    args = parser.parse_args()

    # Determine directory to scan
    if args.directory:
        scan_dir = Path(args.directory).expanduser()
    else:
        scan_dir = Path(__file__).parent

    if not scan_dir.exists():
        print(f"✗ Directory not found: {scan_dir}")
        sys.exit(1)

    # Find all PDF files
    if args.recursive:
        pdf_files = list(scan_dir.rglob("*.pdf"))
    else:
        pdf_files = list(scan_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {scan_dir}")
        sys.exit(0)

    print(f"Scanning {len(pdf_files)} PDF file(s) in {scan_dir}")
    print(f"Recursive: {args.recursive}\n")

    # Process each PDF
    total_improper = 0
    for pdf_file in pdf_files:
        if check_improper_redaction(pdf_file):
            total_improper += 1

    print(f"\n{'=' * 80}")
    print(f"Summary: {total_improper} file(s) with improper redactions out of {len(pdf_files)} total")


if __name__ == "__main__":
    main()