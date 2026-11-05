"""``_convert_to_png`` must leave a real PNG on disk for images Pillow can read.

Clipboards often hand over BMP (or other non-PNG) data, which is saved under the
destination ``.png`` name and converted in place. ``_convert_to_png`` reports
success whenever the file is non-empty, so these tests check the bytes left on
disk. The ImageMagick fallback is made unavailable so only Pillow can produce it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

pytestmark = pytest.mark.skipif(Image is None, reason="Pillow not installed")

from hermes_cli.clipboard import _convert_to_png  # noqa: E402

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _convert_with_pillow_only(path) -> bool:
    with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError("convert")):
        return _convert_to_png(path)


def test_bmp_becomes_valid_png(tmp_path):
    path = tmp_path / "clipboard.png"
    Image.new("RGB", (5, 3), (200, 30, 60)).save(path, format="BMP")

    assert _convert_with_pillow_only(path) is True

    assert path.read_bytes().startswith(_PNG_SIGNATURE)
    with Image.open(path) as png:
        assert png.format == "PNG"
        assert png.size == (5, 3)
        assert png.convert("RGB").getpixel((4, 2)) == (200, 30, 60)


def test_i_mode_image_becomes_valid_png(tmp_path):
    path = tmp_path / "clipboard.png"
    image = Image.new("I", (5, 3), 0)
    image.paste(0xFFFF, (3, 0, 5, 3))
    image.save(path, format="TIFF")
    with Image.open(path) as source:
        assert source.mode == "I"

    assert _convert_with_pillow_only(path) is True

    assert path.read_bytes().startswith(_PNG_SIGNATURE)
    with Image.open(path) as png:
        assert png.format == "PNG"
        assert png.size == (5, 3)
        gray = png.convert("L")
        assert gray.getpixel((0, 0)) == 0
        assert gray.getpixel((4, 2)) == 255
