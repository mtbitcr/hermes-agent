"""Pillow contract for the vision image-prep helpers.

``tools/vision_tools_image_prep.py`` leans on Pillow to re-encode rasters the
vision APIs reject, to crop regions, and to bound the decode check on animated
images. These tests build small images in place and pin that behaviour across
Pillow upgrades: high-bit-depth grayscale TIFF/BMP and I-mode crops must come
out as real PNGs, and the decode check must reject animations over its frame or
aggregate-pixel budget while passing a small one.
"""

from __future__ import annotations

import struct

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

pytestmark = pytest.mark.skipif(Image is None, reason="Pillow not installed")

from tools.vision_tools_image_prep import (  # noqa: E402
    _crop_image_region,
    _normalize_to_supported_image,
    _validate_raster_image_decodable,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MODE_MAX = {"I;16": 0xFFFF, "I": 0x7FFFFFFF}


def _dark_left_bright_right(mode: str, size=(8, 4)):
    """Grayscale image at ``mode`` depth: left half 0, right half the mode's max value."""
    width, height = size
    image = Image.new(mode, size, 0)
    image.paste(_MODE_MAX[mode], (width // 2, 0, width, height))
    return image


def _grayscale_bmp(bits: int, width: int = 6, height: int = 4) -> bytes:
    """Uncompressed 16-bit (X1R5G5B5) or 32-bit (BGRX) BMP with equal channels: left half
    black, right half white. Pillow cannot write 16-bit BMPs, so the bytes are built by hand."""
    rows = []
    for _ in range(height):
        grays = [0] * (width // 2) + [255] * (width - width // 2)
        if bits == 16:
            row = b"".join(struct.pack("<H", ((g >> 3) << 10) | ((g >> 3) << 5) | (g >> 3)) for g in grays)
        else:
            row = b"".join(bytes((g, g, g, 0)) for g in grays)
        rows.append(row + b"\0" * (-len(row) % 4))
    pixels = b"".join(rows)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, bits, 0, len(pixels), 2835, 2835, 0, 0)
    offset = 14 + len(info)
    return struct.pack("<2sIHHI", b"BM", offset + len(pixels), 0, 0, offset) + info + pixels


def _assert_dark_left_bright_right_png(path, size):
    assert path is not None and path.read_bytes().startswith(_PNG_SIGNATURE)
    with Image.open(path) as png:
        assert png.format == "PNG"
        assert png.size == size
        gray = png.convert("L")
        assert gray.getpixel((0, 0)) == 0
        assert gray.getpixel((size[0] - 1, size[1] - 1)) == 255


def _write_animated_gif(path, frames: int, size=(4, 4)):
    images = [Image.new("L", size, 40 * index) for index in range(frames)]  # distinct, so none merge
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:], duration=50, loop=0)
    with Image.open(path) as written:
        assert written.n_frames == frames


@pytest.mark.parametrize("mode", ["I;16", "I"], ids=["16-bit", "32-bit"])
def test_high_bit_depth_grayscale_tiff_becomes_png(tmp_path, mode):
    source = tmp_path / "scan.tiff"
    _dark_left_bright_right(mode).save(source, format="TIFF")
    with Image.open(source) as opened:
        assert opened.mode == mode

    out_path, mime, error = _normalize_to_supported_image(source, "image/tiff")

    assert error is None
    assert mime == "image/png"
    _assert_dark_left_bright_right_png(out_path, (8, 4))


@pytest.mark.parametrize("bits", [16, 32], ids=["16-bit", "32-bit"])
def test_high_bit_depth_grayscale_bmp_becomes_png(tmp_path, bits):
    source = tmp_path / "screenshot.bmp"
    source.write_bytes(_grayscale_bmp(bits))

    out_path, mime, error = _normalize_to_supported_image(source, "image/bmp")

    assert error is None
    assert mime == "image/png"
    _assert_dark_left_bright_right_png(out_path, (6, 4))


def test_i_mode_crop_becomes_png(tmp_path):
    source = tmp_path / "depth.tiff"
    _dark_left_bright_right("I").save(source, format="TIFF")
    offset = {}

    out_path, mime, error = _crop_image_region(source, [2, 0, 8, 3], offset_out=offset)

    assert error is None
    assert mime == "image/png"
    assert offset == {"x": 2, "y": 0, "width": 6, "height": 3}
    _assert_dark_left_bright_right_png(out_path, (6, 3))


def test_decode_check_rejects_animation_over_frame_limit(tmp_path):
    gif = tmp_path / "long.gif"
    _write_animated_gif(gif, frames=4)

    error = _validate_raster_image_decodable(gif, max_frames=3)

    assert error is not None
    assert "rejected animation" in error
    assert "frame 4" in error and "validated frames" in error


def test_decode_check_rejects_animation_over_pixel_limit(tmp_path):
    gif = tmp_path / "large.gif"
    _write_animated_gif(gif, frames=3)  # 3 frames x 16 px = 48 px

    error = _validate_raster_image_decodable(gif, max_frames=10, max_pixels=40)

    assert error is not None
    assert "rejected animation" in error
    assert "pixel count would reach 48 at frame 3" in error


def test_decode_check_passes_small_animation(tmp_path):
    gif = tmp_path / "small.gif"
    _write_animated_gif(gif, frames=3)

    assert _validate_raster_image_decodable(gif) is None
