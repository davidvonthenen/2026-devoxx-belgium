#!/usr/bin/env python3
"""Inspect and export embedded PDF revisions without rewriting PDF objects.

Dependency: Python >= 3.10 and pyHanko==0.37.0
    python -m pip install 'pyHanko==0.37.0'

Examples:
    python pdf_revisions.py input.pdf --list
    python pdf_revisions.py input.pdf --revision previous -o previous.pdf
    python pdf_revisions.py input.pdf --revision 1 -o oldest.pdf
    python pdf_revisions.py input.pdf --all -d recovered
    python pdf_revisions.py input.pdf --list --objects

Revision numbers are 1-based, oldest first. "Previous" means the saved
revision immediately before the latest, not the second xref section.

pyHanko parses xref tables, xref streams, object streams, and /Prev history.
This script uses parser-reported container boundaries, not a global search
for %%EOF. Every candidate prefix is independently reparsed and checked
against the original xref chain before it is offered for export.

The xref metadata APIs are internal to pyHanko, hence the dependency pin.
A strict-parser failure is an error, never a reason to silently repair or
rewrite the source. Auxiliary xref sections (including linearization)
are grouped only when a complete, independently parseable prefix proves
the grouping. Unrecognized layouts fail rather than guessing.

Outputs preserve the original bytes through the selected %%EOF and its
existing line ending. Encryption is preserved; no decryption is performed.
This is not signature validation, a semantic object diff, or data carving.
Only history reachable through the current PDF's parsed xref chain is used.
Run untrusted PDFs in an isolated environment with resource limits.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Callable

PDF_WS = b" \t\r\n\f\x00"
FOOTER = re.compile(
    rb"startxref[ \t\r\n\f\x00]+(?P<xref>[0-9]+)"
    rb"[ \t\r\n\f\x00]+%%EOF(?=$|[ \t\r\n\f\x00])"
)


class RevisionError(Exception):
    """A requested snapshot cannot be established or safely written."""


@dataclass(frozen=True)
class Section:
    index: int
    xref: int
    kind: str
    footer_offset: int
    container_end: int


@dataclass(frozen=True)
class Snapshot:
    number: int
    through_section: int
    xref: int
    eof_end: int
    byte_end: int
    kind: str


def skip_gap(data: bytes, position: int) -> int:
    """Skip PDF whitespace and comments at a known token boundary."""
    while position < len(data):
        if data[position] in PDF_WS:
            position += 1
        elif data[position:position + 1] == b"%":
            while position < len(data) and data[position] not in b"\r\n":
                position += 1
        else:
            break
    return position


def parse_footer(data: bytes, position: int) -> tuple[int, int, int]:
    """Match only the footer immediately following a parsed xref container.

    Return (numeric xref pointer, EOF-token exclusive end, byte-prefix end).
    Keep existing horizontal whitespace and one existing line ending after
    %%EOF; do not manufacture a newline or consume the next revision.
    """
    position = skip_gap(data, position)
    match = FOOTER.match(data, position)
    if match is None:
        raise RevisionError(
            f"No supported startxref/%%EOF footer at parsed boundary {position}."
        )
    eof_end = match.end()
    byte_end = eof_end
    while byte_end < len(data) and data[byte_end] in b" \t":
        byte_end += 1
    if data[byte_end:byte_end + 2] == b"\r\n":
        byte_end += 2
    elif data[byte_end:byte_end + 1] in (b"\n", b"\r"):
        byte_end += 1
    return int(match.group("xref")), eof_end, byte_end


class PyHankoIndex:
    """Small adapter isolating pyHanko's version-sensitive xref APIs."""

    def __init__(self, data: bytes):
        try:
            from pyhanko.pdf_utils import generic
            from pyhanko.pdf_utils.reader import PdfFileReader
        except ImportError as exc:
            raise RevisionError(
                "Missing dependency. Run: python -m pip install 'pyHanko==0.37.0'"
            ) from exc
        self.data = data
        self.stream = io.BytesIO(data)
        self.reader = PdfFileReader(self.stream, strict=True)
        self.generic = generic
        self.count = self.reader.total_revisions
        self.pointers = tuple(
            int(self.reader.xrefs.get_startxref_for_revision(i))
            for i in range(self.count)
        )
        if not self.pointers or len(set(self.pointers)) != len(self.pointers):
            raise RevisionError("Missing or repeated xref offsets in parsed history.")
        if any(p < 0 or p >= len(data) for p in self.pointers):
            raise RevisionError("A parsed xref offset is outside the input file.")
        if int(self.reader.last_startxref) != self.pointers[-1]:
            raise RevisionError("The final footer and parsed xref chain disagree.")

    def sections(self) -> list[Section]:
        result = []
        for i, xref in enumerate(self.pointers):
            meta = self.reader.xrefs.get_xref_container_info(i)
            kind = meta.xref_section_type.name
            self.stream.seek(meta.end_location)
            if kind in ("STANDARD", "HYBRID_MAIN"):
                if (self.data[xref:xref + 4] != b"xref"
                        or xref + 4 >= len(self.data)
                        or self.data[xref + 4] not in PDF_WS):
                    raise RevisionError(f"Declared xref offset {xref} does not point to a table.")
                # A table's end_location is the START of its trailer dictionary.
                trailer = self.generic.DictionaryObject.read_from_stream(
                    self.stream, self.generic.TrailerReference(self.reader)
                )
                if not isinstance(trailer, self.generic.DictionaryObject):
                    raise RevisionError(f"Xref {xref}: trailer is not a dictionary.")
                end = self.stream.tell()
            elif kind == "STREAM":
                if int(meta.start_location) != xref:
                    raise RevisionError(f"Declared xref offset {xref} required parser correction.")
                # A stream's end_location follows endstream, BEFORE endobj.
                end = skip_gap(self.data, int(meta.end_location))
                if self.data[end:end + 6] != b"endobj":
                    raise RevisionError(f"Xref {xref}: expected endobj at byte {end}.")
                end += 6
                if end < len(self.data) and self.data[end] not in PDF_WS:
                    raise RevisionError(f"Xref {xref}: invalid endobj token boundary.")
            else:
                raise RevisionError(f"Unsupported top-level xref kind: {kind}.")
            result.append(Section(i, xref, kind, end, end))
        return result

    def check_catalog(self) -> None:
        # Xref recovery does not require decrypting document objects.
        if self.reader.encrypted:
            return
        root = self.reader.root
        if str(root.get("/Type")) != "/Catalog" or "/Pages" not in root:
            raise RevisionError("The recovered prefix has no valid document catalog.")
        pages = root["/Pages"]
        if str(pages.get("/Type")) != "/Pages" or int(pages["/Count"]) < 0:
            raise RevisionError("The recovered prefix has no valid page-tree root.")

    def object_summary(self, first: int, last: int) -> list[str]:
        """List xref entries, not claims of semantic changes to object content."""
        lines = []
        for i in range(first, last + 1):
            refs = self.reader.xrefs.explicit_refs_in_revision(i)
            for ref in sorted(refs, key=lambda r: (r.idnum, r.generation)):
                location = self.reader.xrefs.get_historical_ref(ref, i)
                if location is None:
                    description = "free entry"
                elif isinstance(location, int):
                    description = f"byte offset {location}"
                else:
                    description = (
                        f"object stream {location.obj_stream_id}, "
                        f"index {location.ix_in_stream}"
                    )
                lines.append(f"    {ref.idnum} {ref.generation} R: {description}")
        return lines


def discover_snapshots(
    data: bytes, index_factory: Callable[[bytes], Any] = PyHankoIndex
) -> tuple[Any, list[Snapshot]]:
    """Establish standalone byte prefixes using the parsed xref history.

    The footer after a linearized PDF's final xref can point to its initial
    xref. Mapping the pointer back to the parsed chain, then reparsing the
    prefix, avoids counting these two sections as two saved versions.
    """
    original = index_factory(data)
    sections = original.sections()
    pointer_to_index = {pointer: i for i, pointer in enumerate(original.pointers)}
    candidates: dict[int, Snapshot] = {}
    for section in sections:
        pointer, eof_end, byte_end = parse_footer(data, section.footer_offset)
        if pointer == 0 and pointer not in pointer_to_index:
            # A linearized first-page section can contain a placeholder footer.
            # It is not accepted as a standalone version. Coverage checks below
            # still require that a complete snapshot accounts for this section.
            continue
        if pointer not in pointer_to_index:
            raise RevisionError(
                f"Footer at byte {section.footer_offset} points to xref {pointer}, "
                "which is absent from the parsed history."
            )
        through = pointer_to_index[pointer]
        if through < section.index:
            raise RevisionError("A footer points backwards in the parsed section sequence.")
        expected = original.pointers[:through + 1]
        prefix = data[:byte_end]
        recovered = index_factory(prefix)
        if recovered.pointers != expected:
            raise RevisionError(
                f"Prefix ending at byte {byte_end} does not reproduce its expected xref chain."
            )
        recovered.check_catalog()
        if any(s.container_end > eof_end for s in sections[:through + 1]):
            raise RevisionError("A candidate prefix omits an xref container it needs.")
        if through in candidates:
            raise RevisionError("Multiple different footers map to the same saved revision.")
        candidates[through] = Snapshot(
            0, through, pointer, eof_end, byte_end, sections[through].kind
        )

    ordered = sorted(candidates.values(), key=lambda s: s.byte_end)
    if not ordered or ordered[-1].through_section != len(sections) - 1:
        raise RevisionError("Cannot establish a complete latest snapshot from parsed history.")
    snapshots = []
    previous_section = -1
    for number, candidate in enumerate(ordered, 1):
        if candidate.through_section <= previous_section:
            raise RevisionError("Snapshot byte order disagrees with xref history order.")
        snapshots.append(Snapshot(
            number, candidate.through_section, candidate.xref,
            candidate.eof_end, candidate.byte_end, candidate.kind
        ))
        previous_section = candidate.through_section
    return original, snapshots


def choose_snapshot(snapshots: list[Snapshot], selection: str) -> Snapshot:
    if selection == "previous":
        if len(snapshots) < 2:
            raise RevisionError("No earlier saved revision is present in the parsed history.")
        return snapshots[-2]
    if selection == "oldest":
        return snapshots[0]
    if selection == "latest":
        return snapshots[-1]
    try:
        number = int(selection)
    except ValueError as exc:
        raise RevisionError("Use previous, oldest, latest, or a 1-based revision number.") from exc
    if not 1 <= number <= len(snapshots):
        raise RevisionError(f"Revision must be between 1 and {len(snapshots)}.")
    return snapshots[number - 1]


def write_snapshot(source: Path, output: Path, data: bytes, snapshot: Snapshot) -> str:
    if source.resolve() == output.resolve():
        raise RevisionError("Refusing to overwrite the source PDF.")
    if output.exists() and os.path.samefile(source, output):
        raise RevisionError("Output is a hard link to the source PDF.")
    payload = memoryview(data)[:snapshot.byte_end]
    # Exclusive creation also rejects existing files, symlinks, and races.
    with output.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Original PDF (read only)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="List verified saved revisions (default)")
    action.add_argument("--revision", metavar="N|previous|oldest|latest")
    action.add_argument("--all", action="store_true", help="Export all verified saved revisions")
    parser.add_argument("-o", "--output", type=Path, help="New output path for --revision")
    parser.add_argument("-d", "--output-dir", type=Path, help="New directory for --all")
    parser.add_argument("--objects", action="store_true", help="Include explicit xref entries for each saved revision")
    args = parser.parse_args(argv)
    if args.revision and args.output is None:
        parser.error("--revision requires --output")
    if args.output and not args.revision:
        parser.error("--output requires --revision")
    if args.all and args.output_dir is None:
        parser.error("--all requires --output-dir")
    if args.output_dir and not args.all:
        parser.error("--output-dir requires --all")
    try:
        data = args.input.read_bytes()
        index, snapshots = discover_snapshots(data)
        print(f"Source: {args.input}")
        print(f"Source bytes: {len(data)}; SHA-256: {hashlib.sha256(data).hexdigest()}")
        print(f"Verified saved revisions: {len(snapshots)}; parsed xref sections: {index.count}")
        print("Revision  Xref offset  EOF-token end  Prefix bytes  Xref kind")
        first_section = 0
        for item in snapshots:
            tag = " (latest)" if item.number == len(snapshots) else ""
            print(f"{item.number:8}  {item.xref:11}  {item.eof_end:13}  {item.byte_end:12}  {item.kind}{tag}")
            if args.objects:
                for line in index.object_summary(first_section, item.through_section):
                    print(line)
            first_section = item.through_section + 1
        if args.revision:
            selected = choose_snapshot(snapshots, args.revision)
            digest = write_snapshot(args.input, args.output, data, selected)
            print(f"Wrote revision {selected.number}: {args.output}")
            print(f"Output SHA-256: {digest}")
        elif args.all:
            # A new directory prevents accidental mixing with earlier exports.
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for selected in snapshots:
                output = args.output_dir / f"{args.input.stem}.rev-{selected.number:03d}.pdf"
                digest = write_snapshot(args.input, output, data, selected)
                print(f"Wrote {output}; SHA-256: {digest}")
        return 0
    except (RevisionError, OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Includes pyHanko parse/decompression/security-handler exceptions.
        print(f"ERROR: PDF parser rejected the operation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
