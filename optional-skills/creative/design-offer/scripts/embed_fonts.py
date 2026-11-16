#!/usr/bin/env python3
"""Embed the website's two font files in a full-page design file.

    embed_fonts.py --website WEBSITE_TREE --page full-page.html

Reads the two woff2 files from the website's installed packages, refuses unless the
website's package.json pins both packages and both installed packages are exactly
5.3.0, and writes two ``@font-face`` rules (families 'Geist' and 'Geist Mono') whose
``src`` is a ``data:font/woff2;base64,...`` address into the page's head. The font
bytes are never changed. The page is parsed with the standard library's HTML parser and
every address in it (an attribute that can hold one, a ``url()`` or an ``@import`` in a
style) must be a ``data:`` address or a ``#`` reference inside the page; any other value
is refused, before anything is written.

Standard library only. Prints the SHA-256 of each font file, never page text.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PINNED_VERSION = "5.3.0"
# (npm package, font file inside it, CSS family, CSS font-weight)
FONTS = (
    ("@fontsource-variable/geist", "files/geist-latin-wght-normal.woff2", "Geist", "400 500"),
    ("@fontsource-variable/geist-mono", "files/geist-mono-latin-wght-normal.woff2", "Geist Mono", "400"),
)
CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\n\r\f]?|(\r\n|[\n\r\f])|(.))", re.DOTALL)
# Attributes that can hold an address; a prefixed name (xlink:href) is judged by the part after the colon.
ADDRESS_ATTRIBUTES = frozenset({
    "src", "srcset", "imagesrcset", "href", "action", "formaction", "poster", "data", "codebase", "cite", "background",
    "longdesc", "usemap", "manifest", "ping", "itemtype", "profile", "classid", "lowsrc", "dynsrc", "icon", "xlink:href",
    "xml:base",
})
# Attributes whose value is a list of addresses.
SRCSET_ATTRIBUTES = frozenset({"srcset", "imagesrcset"})
LIST_ATTRIBUTES = frozenset({"ping", "itemtype"})
CSS_URL = re.compile(r"url\s*\(", re.IGNORECASE)
CSS_IMPORT = re.compile(r"@import", re.IGNORECASE)
CSS_IMAGE_FUNCTION = re.compile(r"(?<![\w-])(?:-webkit-)?(?:image-set|cross-fade|image|src)\s*\(", re.IGNORECASE)
ASCII_SPACE = "".join(chr(c) for c in range(0x21))
DROPPED = {9: None, 10: None, 13: None}


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


class _Outside(Exception):
    """An address that is neither a data: address nor a # reference; carries what to name, never page text."""


def _allowed(value: str) -> bool:
    # Decide as a browser's URL parser reads it: tab and newlines vanish, leading and trailing control characters and spaces are stripped.
    address = value.translate(DROPPED).strip(ASCII_SPACE)
    return address[:5].lower() == "data:" or address.startswith("#")


def _require(value: str, name: str) -> None:
    if not _allowed(value):
        raise _Outside(name)


def _srcset_urls(value: str) -> list[str]:
    urls, i, n = [], 0, len(value)
    while i < n:
        while i < n and (value[i] in ASCII_SPACE or value[i] == ","):
            i += 1
        start = i
        while i < n and value[i] not in ASCII_SPACE:
            i += 1
        url = value[start:i]
        if not url:
            break
        if url.endswith(","):
            url = url.rstrip(",")
        else:
            depth = 0
            while i < n:
                char = value[i]
                i += 1
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth = max(0, depth - 1)
                elif char == "," and depth == 0:
                    break
        urls.append(url)
    return urls


def _skip_css_space(text: str, i: int) -> int:
    while i < len(text):
        if text[i] in " \t\n\r\f":
            i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise _Outside("@import")
            i = end + 2
        else:
            break
    return i


def _css_string(text: str, i: int, name: str) -> tuple[str, int]:
    end = text.find(text[i], i + 1)
    if end < 0:
        raise _Outside(name)
    return text[i + 1:end], end + 1


def _check_css(css: str) -> None:
    text = _css_unescape(css)
    for match in CSS_URL.finditer(text):
        i = match.end()
        while i < len(text) and text[i] in " \t\n\r\f":
            i += 1
        if i < len(text) and text[i] in "\"'":
            value, _ = _css_string(text, i, "url()")
        else:
            end = text.find(")", i)
            if end < 0:
                raise _Outside("url()")
            value = text[i:end]
        _require(value, "url()")
    for match in CSS_IMPORT.finditer(text):
        i = _skip_css_space(text, match.end())
        if CSS_URL.match(text, i):
            continue
        if i < len(text) and text[i] in "\"'":
            value, _ = _css_string(text, i, "@import")
            _require(value, "@import")
        else:
            raise _Outside("@import")
    for match in CSS_IMAGE_FUNCTION.finditer(text):
        i, depth = match.end(), 1
        while i < len(text) and depth:
            char = text[i]
            if char in "\"'":
                value, i = _css_string(text, i, "url()")
                _require(value, "url()")
                continue
            depth += (char == "(") - (char == ")")
            i += 1
        if depth:
            raise _Outside("url()")


class _Allowlist(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._style: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attributes = {}
        for name, value in attrs:
            attributes.setdefault(name, value if value is not None else "")
        refresh = attributes.get("http-equiv", "").translate(DROPPED).strip(ASCII_SPACE).lower() == "refresh"
        for name, value in attrs:
            value = value if value is not None else ""
            shown = name if re.fullmatch(r"[a-z0-9:_.-]{1,40}", name) else "unnamed"
            local = name.rsplit(":", 1)[-1]
            if name == "xmlns" or name.startswith("xmlns:"):
                continue
            if name in ADDRESS_ATTRIBUTES or local in ADDRESS_ATTRIBUTES:
                if name in SRCSET_ATTRIBUTES or local in SRCSET_ATTRIBUTES:
                    urls = _srcset_urls(value) or [""]
                elif name in LIST_ATTRIBUTES or local in LIST_ATTRIBUTES:
                    urls = value.split() or [""]
                else:
                    urls = [value]
                for url in urls:
                    _require(url, f"{shown} attribute")
            elif name == "content" and refresh:
                rest = re.sub(r"^[\s\d.]*[;,]?\s*", "", value)
                rest = re.sub(r"^url\s*=\s*", "", rest, flags=re.IGNORECASE).strip(ASCII_SPACE + "\"'")
                if rest:
                    _require(rest, "content attribute")
            elif name == "srcdoc":
                inner = _Allowlist()
                inner.feed(value)
                inner.close()
                inner.finish()
            _check_css(value)
        if tag == "style":
            self._style = []

    handle_startendtag = handle_starttag

    def handle_data(self, data):
        if self._style is not None:
            self._style.append(data)

    def handle_endtag(self, tag):
        if tag == "style":
            self.finish()

    def finish(self):
        if self._style is not None:
            # A style element that is never closed runs to the end of the page; the parser keeps that text unparsed.
            css, self._style = "".join(self._style) + (self.rawdata if self.cdata_elem else ""), None
            _check_css(css)


def check_addresses(text: str) -> None:
    """Refuse unless every address of the page is a data: address or a # reference inside the page."""
    parser = _Allowlist()
    try:
        parser.feed(text)
        parser.close()
        parser.finish()
    except _Outside as outside:
        raise EmbedError(
            f"the page holds a {outside} that is not a data: address or a # reference; put everything inline") from None


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
    check_addresses(text)
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
