#!/usr/bin/env python3
"""Compose one picture per design option from the capture kit's two widths.

For each ``option-<n>.html`` and the folder the capture kit wrote for it, place
the 1280 px capture left and the 375 px capture right on one canvas, labelled,
at most 4 MiB, and write ``option-<n>-1280-375.png`` (or ``.jpg``) plus one
``design-offer-manifest.json`` that ties every picture to the page it shows.

    compose_offer.py --out DIR --option option-1.html CAPTURES1 --option option-2.html CAPTURES2

Standard library plus Pillow. Nothing from a bundle or a page is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MAX_PICTURE_BYTES = 4 * 1024 * 1024
MIN_OPTIONS, MAX_OPTIONS = 2, 3
WIDTHS = (1280, 375)
MANIFEST = "design-offer-manifest.json"
BUNDLE_SCHEMA = "design_compare_bundle_v1"
PAGE_NAME = re.compile(r"option-([1-3])\.html")
SHA256 = re.compile(r"[0-9a-f]{64}")

MARGIN = 40
HEADER = 72
BACKGROUND = (240, 240, 240)
LABEL_COLOR = (20, 20, 20)
LABEL_SIZE = 32
JPEG_QUALITIES = (85, 70, 55, 40)
SCALE_STEP = 0.8
MIN_SCALE = 0.2


class OfferError(Exception):
    """The options cannot be composed; nothing was written."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bundle(folder: Path, width: int) -> tuple[dict, Path, bytes]:
    bundle_files = sorted(folder.glob(f"*-{width}.json"))
    if len(bundle_files) != 1:
        raise OfferError(f"{folder.name}: expected exactly one {width} px bundle, found {len(bundle_files)}")
    try:
        bundle = json.loads(bundle_files[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise OfferError(f"{folder.name}: the {width} px bundle cannot be read") from None
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise OfferError(f"{folder.name}: the {width} px bundle is not a capture bundle")
    if bundle.get("width") != width:
        raise OfferError(f"{folder.name}: the {width} px bundle names another width")
    if bundle.get("blocked_requests") != []:
        raise OfferError(f"{folder.name}: the {width} px capture blocked a request; refused")
    for key in ("kit_sha256", "chromium_version"):
        if not isinstance(bundle.get(key), str) or not bundle[key]:
            raise OfferError(f"{folder.name}: the {width} px bundle has no {key}")
    if not SHA256.fullmatch(bundle["kit_sha256"]):
        raise OfferError(f"{folder.name}: the {width} px bundle has a malformed kit_sha256")
    image = bundle.get("image")
    if not isinstance(image, str) or image != Path(image).name or not image.endswith(f"-{width}.png"):
        raise OfferError(f"{folder.name}: the {width} px bundle names a picture outside its folder")
    picture = folder / image
    if picture.is_symlink() or not picture.is_file():
        raise OfferError(f"{folder.name}: the {width} px picture {image} is missing")
    return bundle, picture, picture.read_bytes()


def _load_capture(folder: Path, width: int) -> dict:
    bundle, path, data = _read_bundle(folder, width)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise OfferError(f"{folder.name}: {path.name} is not a readable picture") from None
    if image.format != "PNG":
        raise OfferError(f"{folder.name}: {path.name} is not a PNG")
    height = bundle.get("page_height")
    if image.width != width or type(height) is not int or image.height != height:
        raise OfferError(f"{folder.name}: {path.name} does not match its bundle (width and page height)")
    return {
        "width": width, "file": path.name, "sha256": _sha256(data), "image": image.convert("RGB"),
        "kit_sha256": bundle["kit_sha256"], "chromium_version": bundle["chromium_version"],
    }


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _canvas(left: Image.Image, right: Image.Image, scale: float) -> Image.Image:
    def scaled(image: Image.Image) -> Image.Image:
        if scale == 1.0:
            return image
        return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)

    margin, header = round(MARGIN * scale), round(HEADER * scale)
    left, right = scaled(left), scaled(right)
    canvas = Image.new("RGB", (margin * 3 + left.width + right.width, header + margin + max(left.height, right.height)),
                       BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = _font(max(8, round(LABEL_SIZE * scale)))
    for x, image, label in ((margin, left, "1280 px"), (margin * 2 + left.width, right, "375 px")):
        canvas.paste(image, (x, header))
        draw.text((x, header // 2), label, fill=LABEL_COLOR, font=font, anchor="lm")
    return canvas


def _encode(canvas: Image.Image) -> tuple[bytes, str] | None:
    png = io.BytesIO()
    canvas.save(png, format="PNG")
    if png.tell() <= MAX_PICTURE_BYTES:
        return png.getvalue(), ".png"
    for quality in JPEG_QUALITIES:
        jpeg = io.BytesIO()
        canvas.save(jpeg, format="JPEG", quality=quality)
        if jpeg.tell() <= MAX_PICTURE_BYTES:
            return jpeg.getvalue(), ".jpg"
    return None


def _combined(left: Image.Image, right: Image.Image) -> tuple[bytes, str]:
    # At each scale: lossless PNG first, then JPEG from quality 85 down to 40; shrink evenly only when none fits.
    scale = 1.0
    while scale >= MIN_SCALE:
        encoded = _encode(_canvas(left, right, scale))
        if encoded:
            return encoded
        scale *= SCALE_STEP
    raise OfferError("a picture cannot be made small enough for the offer tool (4 MiB)")


def compose(options, out_dir) -> dict:
    """Write one picture per option and the manifest to *out_dir*; return the manifest.

    *options* is a list of (page file, capture folder). Everything is checked
    before anything is written, and no existing file is overwritten.
    """
    options = [(Path(page), Path(folder)) for page, folder in options]
    out_dir = Path(out_dir)
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise OfferError(f"an offer needs two or three options, got {len(options)}")
    numbers = []
    for page, _ in options:
        match = PAGE_NAME.fullmatch(page.name)
        if not match:
            raise OfferError(f"{page.name}: a page must be named option-<n>.html with n from 1 to 3")
        numbers.append(int(match.group(1)))
    if sorted(numbers) != list(range(1, len(options) + 1)):
        raise OfferError("the options must be numbered 1, 2 and 3 in order, with no gaps or repeats")

    kits, versions = set(), set()
    entries, outputs = [], []
    for (page, folder), n in sorted(zip(options, numbers), key=lambda item: item[1]):
        if page.is_symlink() or not page.is_file():
            raise OfferError(f"{page.name}: the page file is missing")
        if not folder.is_dir():
            raise OfferError(f"{page.name}: the capture folder is missing")
        captures = {w: _load_capture(folder, w) for w in WIDTHS}
        for capture in captures.values():
            kits.add(capture["kit_sha256"])
            versions.add(capture["chromium_version"])
        data, suffix = _combined(captures[1280]["image"], captures[375]["image"])
        name = f"option-{n}-1280-375{suffix}"
        outputs.append((out_dir / name, data))
        entries.append({
            "n": n,
            "page": {"file": page.name, "sha256": _sha256(page.read_bytes())},
            "captures": [{"width": w, "file": captures[w]["file"], "sha256": captures[w]["sha256"]} for w in WIDTHS],
            "picture": {"file": name, "sha256": _sha256(data), "bytes": len(data)},
        })
    if len(kits) != 1:
        raise OfferError("the bundles name different capture kit hashes")
    if len(versions) != 1:
        raise OfferError("the bundles name different Chromium versions")

    manifest = {"kit_sha256": kits.pop(), "chromium_version": versions.pop(), "options": entries}
    outputs.append((out_dir / MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode("utf-8")))
    taken = [path.name for path, _ in outputs if path.exists()]
    if taken:
        raise OfferError(f"refusing to overwrite {', '.join(taken)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, data in outputs:
        with open(path, "xb") as handle:
            handle.write(data)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="folder for the pictures and the manifest")
    parser.add_argument("--option", nargs=2, action="append", default=[], metavar=("PAGE", "CAPTURES"),
                        help="an option page file and the folder the capture kit wrote for it")
    args = parser.parse_args(argv)
    try:
        manifest = compose(args.option, args.out)
    except OfferError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    for entry in manifest["options"]:
        print(f"option {entry['n']} {entry['picture']['file']} {entry['picture']['bytes']} bytes")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
