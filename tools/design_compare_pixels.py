"""Design comparison pictures: difference regions between two stitched page images, and matching crops.

Pure: images come in as PIL images or PNG bytes, crops go out as PNG bytes.
"""

import io
import math
import re
from typing import NamedTuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from tools.design_compare_judge import PIXEL_CHANNEL_DELTA, PIXEL_REGION_MIN_PIXELS, PIXEL_REGION_MIN_SIDE, WIDTHS

MAX_PAGE_HEIGHT = 8000
CROP_PADDING = 24
MAX_CROP_SIDE = 1200
MAX_CROP_BYTES = 1048576
SHRINK = 0.75

_STEP = [0] * (PIXEL_CHANNEL_DELTA + 1) + [255] * (255 - PIXEL_CHANNEL_DELTA)
_ABSENT_FILL = (128, 128, 128)  # rows below the end of the shorter page
_ID = re.compile(r"[0-9a-f]{16}")
_ATTACHING_KINDS = ("layout", "type", "colour")


class PixelError(ValueError):
    pass


class Crops(NamedTuple):
    candidate: tuple  # (file name, PNG bytes)
    prototype: tuple
    scaled: bool


def _rgb(image):
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    return image.convert("RGB")


def _pair(candidate, prototype):
    cand, proto = _rgb(candidate), _rgb(prototype)
    if max(cand.height, proto.height) > MAX_PAGE_HEIGHT:
        raise PixelError(f"page image is taller than {MAX_PAGE_HEIGHT} px")
    if cand.width != proto.width:
        raise PixelError(f"page images have different widths ({cand.width} and {proto.width} px)")
    return cand, proto


def _rect(boxes, size, pad=0):
    left = min(math.floor(b["x"]) for b in boxes) - pad
    top = min(math.floor(b["y"]) for b in boxes) - pad
    right = max(math.ceil(b["x"] + b["width"]) for b in boxes) + pad
    bottom = max(math.ceil(b["y"] + b["height"]) for b in boxes) + pad
    return max(left, 0), max(top, 0), min(right, size[0]), min(bottom, size[1])


def _different(cand, proto, masks):
    red, green, blue = (band.point(_STEP) for band in ImageChops.difference(cand, proto).split())
    mask = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    draw = ImageDraw.Draw(mask)
    for m in masks:
        draw.rectangle((math.floor(m["x"]), math.floor(m["y"]), math.ceil(m["x"] + m["width"]) - 1,
                        math.ceil(m["y"] + m["height"]) - 1), fill=0)
    return mask


def _regions(mask, dx=0, dy=0):
    if not mask.getbbox():
        return []
    width, height = mask.size
    original = mask.tobytes()
    todo = {i for i, v in enumerate(mask.filter(ImageFilter.MaxFilter(3)).tobytes()) if v}
    found = []
    while todo:
        stack, count = [todo.pop()], 0
        x0, y0, x1, y1 = width, height, -1, -1
        while stack:
            y, x = divmod(stack.pop(), width)
            if original[y * width + x]:
                count += 1
                x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
            for ny in range(max(y - 1, 0), min(y + 2, height)):
                for nx in range(max(x - 1, 0), min(x + 2, width)):
                    if (ny * width + nx) in todo:
                        todo.remove(ny * width + nx)
                        stack.append(ny * width + nx)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w >= PIXEL_REGION_MIN_SIDE and h >= PIXEL_REGION_MIN_SIDE and count >= PIXEL_REGION_MIN_PIXELS:
            found.append({"x": x0 + dx, "y": y0 + dy, "width": w, "height": h})
    return sorted(found, key=lambda r: (r["y"], r["x"]))


def compare_pages(candidate, prototype, masks=(), row_limit=None):
    """Page pass: regions (page coordinates) over the common rows, above row_limit when given."""
    cand, proto = _pair(candidate, prototype)
    rows = min(cand.height, proto.height, row_limit if row_limit is not None else MAX_PAGE_HEIGHT)
    cut = (0, 0, cand.width, rows)
    return _regions(_different(cand.crop(cut), proto.crop(cut), masks))


def compare_element(candidate, prototype, candidate_box, prototype_box, masks=()):
    """Element pass: each side cut at its own box, compared over the common size; regions in candidate page coordinates.
    masks are page-coordinate boxes, moved into each side's crop; a pixel covered on either side is excluded."""
    cand, proto = _pair(candidate, prototype)
    cl, ct, cr, cb = _rect([candidate_box], cand.size)
    pl, pt, pr, pb = _rect([prototype_box], proto.size)
    w, h = min(cr - cl, pr - pl), min(cb - ct, pb - pt)
    if w <= 0 or h <= 0:
        return []
    local = [{**m, "x": m["x"] - ox, "y": m["y"] - oy} for m in masks for ox, oy in ((cl, ct), (pl, pt))]
    mask = _different(cand.crop((cl, ct, cl + w, ct + h)), proto.crop((pl, pt, pl + w, pt + h)), local)
    return _regions(mask, cl, ct)


def attach_regions(regions, findings, element_box):
    """Split regions into those inside the box of an element with a layout, type or colour finding
    ({finding id: regions}) and those left undecided; element_box maps a finding's element key to a box."""
    attached, undecided = {}, []
    for region in regions:
        for finding in findings:
            box = element_box(finding.element) if finding.kind in _ATTACHING_KINDS else None
            if box and box["x"] <= region["x"] and box["y"] <= region["y"] \
                    and region["x"] + region["width"] <= box["x"] + box["width"] \
                    and region["y"] + region["height"] <= box["y"] + box["height"]:
                attached.setdefault(finding.id, []).append(region)
                break
        else:
            undecided.append(region)
    return attached, undecided


def crop_rect(size, candidate_box, prototype_box):
    """Union of the boxes present (None for the side where the element is missing) plus padding, clipped to size."""
    return _rect([b for b in (candidate_box, prototype_box) if b], size, CROP_PADDING)


def crop_names(round_no, fid, width):
    if not isinstance(round_no, int) or not _ID.fullmatch(fid) or width not in WIDTHS:
        raise PixelError("crop names need an integer round, a 16 hex finding id and a width of 1280 or 375")
    return tuple(f"dc-r{round_no}-{fid}-{width}-{side}.png" for side in ("candidate", "prototype"))


def _png(image):
    data = io.BytesIO()
    image.save(data, "PNG")
    if data.tell() > MAX_CROP_BYTES:
        data = io.BytesIO()
        image.quantize(256, Image.Quantize.FASTOCTREE).save(data, "PNG", optimize=True)
    return data.getvalue()


def _cut(image, rect):
    left, top, right, bottom = rect
    out = Image.new("RGB", (right - left, bottom - top), _ABSENT_FILL)
    if top < image.height:
        out.paste(image.crop((left, top, right, min(bottom, image.height))), (0, 0))
    return out


def cut_crops(candidate, prototype, candidate_box, prototype_box, round_no, fid, width):
    """Cut both images at the same rectangle; scale both by one factor to fit 1200 x 1200 and 1 MiB each."""
    names = crop_names(round_no, fid, width)
    cand, proto = _pair(candidate, prototype)
    rect = crop_rect((cand.width, max(cand.height, proto.height)), candidate_box, prototype_box)
    crops = (_cut(cand, rect), _cut(proto, rect))
    w, h = crops[0].size
    scale = min(1, MAX_CROP_SIDE / w, MAX_CROP_SIDE / h)
    scaled = scale < 1
    while True:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        pngs = [_png(c if c.size == size else c.resize(size, Image.LANCZOS)) for c in crops]
        if all(len(p) <= MAX_CROP_BYTES for p in pngs):
            return Crops((names[0], pngs[0]), (names[1], pngs[1]), scaled)
        scale *= SHRINK
        scaled = True
