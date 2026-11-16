#!/usr/bin/env python3
"""Embed the website's two font files in a full-page design file.

    embed_fonts.py --website WEBSITE_TREE --page full-page.html

Reads the two woff2 files from the website's installed packages, refuses unless the
website's package.json pins both packages and both installed packages are exactly
5.3.0, and writes two ``@font-face`` rules (families 'Geist' and 'Geist Mono') whose
``src`` is a ``data:font/woff2;base64,...`` address into the page's head. The font
bytes are never changed. A page that already names an outside address is refused.

Standard library only. Prints the SHA-256 of each font file, never page text.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import sys
from pathlib import Path

PINNED_VERSION = "5.3.0"
# (npm package, font file inside it, CSS family, CSS font-weight)
FONTS = (
    ("@fontsource-variable/geist", "files/geist-latin-wght-normal.woff2", "Geist", "400 500"),
    ("@fontsource-variable/geist-mono", "files/geist-mono-latin-wght-normal.woff2", "Geist Mono", "400"),
)
# A browser reads https:host, https:\\host and //host as outside addresses too, so the check runs on the text a browser sees.
OUTSIDE_ADDRESS = re.compile(r"""(?<![\w+.\-/])https?:(?=\S)|(?:^|[\s"'(=,])//[A-Za-z0-9]""", re.IGNORECASE)
CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\n\r\f]?|(\r\n|[\n\r\f])|(.))", re.DOTALL)
URL_NOISE = re.compile(r"[\x00-\x1f\x7f]")
# An XML namespace name of inline SVG is an identifier, not an address the page loads from.
NAMESPACE = re.compile(r"""\bxmlns(?::\w+)?\s*=\s*(["'])http://www\.w3\.org/[^"']*\1""", re.IGNORECASE)


class EmbedError(Exception):
    """The fonts cannot be embedded; the page was not changed."""


def _css_unescape(text: str) -> str:
    def one(m):
        if m.group(2):
            return ""
        if m.group(3):
            return m.group(3)
        code = int(m.group(1), 16)
        return chr(code) if 0 < code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF else "\ufffd"
    return CSS_ESCAPE.sub(one, text)


def names_outside_address(text: str) -> bool:
    """True when the page, read as a browser reads it, names an address outside itself."""
    plain = html.unescape(text)
    for base in (plain, URL_NOISE.sub("", plain)):
        css = _css_unescape(base)
        for seen in (base, css, base.replace("\\", "/"), css.replace("\\", "/")):
            if OUTSIDE_ADDRESS.search(NAMESPACE.sub("", seen)):
                return True
    return False


def _package_json(path: Path, what: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise EmbedError(f"{what} cannot be read") from None
    if not isinstance(data, dict):
        raise EmbedError(f"{what} is not a package file")
    return data


def _font_files(website: Path) -> list[tuple[str, bytes]]:
    pins = _package_json(website / "package.json", "the website's package.json")
    declared = {**(pins.get("devDependencies") or {}), **(pins.get("dependencies") or {})}
    fonts = []
    for package, relative, family, _weight in FONTS:
        if declared.get(package) != PINNED_VERSION:
            raise EmbedError(f"the website does not pin {package} at exactly {PINNED_VERSION}")
        folder = website / "node_modules" / package
        installed = _package_json(folder / "package.json", f"the installed {package}")
        if installed.get("version") != PINNED_VERSION:
            raise EmbedError(f"the installed {package} is not exactly {PINNED_VERSION}")
        try:
            fonts.append((family, (folder / relative).read_bytes()))
        except OSError:
            raise EmbedError(f"the font file {relative} of {package} is missing") from None
    return fonts


def embed(website, page) -> list[dict]:
    """Write the two font faces into *page*; return the family and SHA-256 of each font file."""
    website, page = Path(website), Path(page)
    try:
        text = page.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise EmbedError("the page file cannot be read") from None
    if names_outside_address(text):
        raise EmbedError("the page names an outside address; put everything inline")
    head_end = text.lower().find("</head>")
    if head_end < 0:
        raise EmbedError("the page has no head to hold the font faces")
    fonts = _font_files(website)
    rules = []
    for (_package, _relative, family, weight), (_name, data) in zip(FONTS, fonts):
        encoded = base64.b64encode(data).decode("ascii")
        rules.append(f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
                     f"src:url(data:font/woff2;base64,{encoded}) format('woff2')}}")
    page.write_text(text[:head_end] + "<style>" + "".join(rules) + "</style>" + text[head_end:], encoding="utf-8")
    return [{"family": family, "sha256": hashlib.sha256(data).hexdigest()} for family, data in fonts]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--website", required=True, help="the website tree, after npm ci")
    parser.add_argument("--page", required=True, help="the full-page file to change")
    args = parser.parse_args(argv)
    try:
        written = embed(args.website, args.page)
    except EmbedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    for font in written:
        print(f"{font['sha256']}  {font['family']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
