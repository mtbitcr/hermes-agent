"""``kanban_render_drawing`` — real registry dispatch, real boards, the real renderer.

The designer draws its layouts as SVG, which the owner app never shows, so the
tool turns an SVG drawing attached to the worker's own card into a PNG picture
at the drawing's own size, stored as a new attachment on the same card.

Every test that calls the tool runs in the exact environment the dispatcher
gives the worker it launches. It is taken from ``kanban_db._default_spawn``
itself, with only the process launch stubbed: ``HERMES_KANBAN_DB`` and
``HERMES_KANBAN_BOARD`` pinned to the worker's own board, ``HERMES_KANBAN_TASK``
set to its card, and ``HERMES_HOME`` at its profile home below a temporary
root that holds two more boards. From there the root registry and every other
board are checked to be still in reach. Drawings are attached with the real
``kanban_attach`` and rendered through the registry. Each proof is read back
from the three board stores with plain read-only SQL and from the attachment
files counted on disk.
"""

from __future__ import annotations

import base64
import codecs
import http.client
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
    non_dispatcher_owned_context,
)
from hermes_cli import kanban_db as kb

TOOL = "kanban_render_drawing"
OFFER = "kanban_offer_design_options"
OWN = "own-board"
OTHER = "other-board"
QUESTION = "Which home page should Raphael build?"
SVG_TYPE = "image/svg+xml"
NS = 'xmlns="http://www.w3.org/2000/svg"'
XLINK = 'xmlns:xlink="http://www.w3.org/1999/xlink"'
BLUE = (32, 80, 192, 255)
RED = (208, 32, 32, 255)
GREEN = (32, 160, 64, 255)
ONLY_THE_WORKER = "only for the worker the dispatcher spawned on its own task"


def _png_bytes() -> bytes:
    out = io.BytesIO()
    Image.new("RGBA", (8, 8), BLUE).save(out, format="PNG")
    return out.getvalue()


PICTURE = _png_bytes()  # a real 8x8 PNG
PICTURE_URI = "data:image/png;base64," + base64.b64encode(PICTURE).decode("ascii")
NESTED_URI = "data:image/svg+xml;base64," + base64.b64encode(
    f'<svg {NS} width="8" height="8"><rect width="8" height="8"/></svg>'.encode("utf-8")
).decode("ascii")
RECT = '<rect width="40" height="40" fill="#2050c0"/>'

# Drawn as the designer draws: a gradient painted through a <style> url(#...),
# and one shape reused through both href and xlink:href.
PLAIN = f"""<svg {NS} {XLINK} width="320" height="200">
  <defs>
    <linearGradient id="sky"><stop offset="0" stop-color="#2050c0"/><stop offset="1" stop-color="#2050c0"/></linearGradient>
    <rect id="card" width="100" height="60" fill="#d02020"/>
  </defs>
  <style>.panel {{ fill: url(#sky); }}</style>
  <rect class="panel" width="320" height="200"/>
  <use href="#card" x="20" y="20"/>
  <use xlink:href="#card" x="200" y="120"/>
</svg>"""
VIEWBOX_ONLY = f'<svg {NS} viewBox="0 0 640 480"><rect width="640" height="480" fill="#20a040"/></svg>'
# Sized in em and in ex of the svg element's own font size: 10em of 20 px is 200 px, and
# resvg takes 1ex as half of 1em, so 10ex is 100 px. The style sheet sets only the font
# size of text, never the svg element's.
EM = ('<svg xmlns="http://www.w3.org/2000/svg" width="10em" height="2em" font-size="20">'
      '<rect width="100%" height="100%" fill="#2050c0"/></svg>')
EX = (f'<svg {NS} width="10ex" height="4ex" font-size="20"><style>text {{ font-size: 40px; }}</style>'
      '<rect width="100%" height="100%" fill="#20a040"/></svg>')
# Fractal noise does not compress: about 5.6 MB of PNG at 1,280 px square.
NOISE = (
    f'<svg {NS} width="1280" height="1280"><filter id="n" x="0" y="0" width="1" height="1">'
    '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="7"/></filter>'
    '<rect width="1280" height="1280" filter="url(#n)"/></svg>'
)
# Sized in em of a font size the drawing's own CSS gives its svg element: a <style> rule
# beats the font-size attribute, so 10em by 2em of 40 px is 400x80 px, and the font
# shorthand's 15pt is 20 px, so 10em by 4em is 200x80 px.
SHEET = (f'<svg {NS} width="10em" height="2em" font-size="20"><style>svg {{ font-size: 40px; }}</style>'
         '<rect width="100%" height="100%" fill="#2050c0"/></svg>')
SHORTHAND = (f'<svg {NS} width="10em" height="4em" style="font: bold 15pt serif">'
             '<rect width="100%" height="100%" fill="#20a040"/></svg>')
NOISE_IN_EM = (  # the same noise at 64em of 20 px: 1,280 px square again
    f'<svg {NS} width="64em" height="64em" font-size="20"><filter id="n" x="0" y="0" width="1" height="1">'
    '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="7"/></filter>'
    '<rect width="1280" height="1280" filter="url(#n)"/></svg>'
)


def _svg(body: str) -> str:
    """A 40 px square drawing holding ``body``."""
    return f'<svg {NS} {XLINK} width="40" height="40">{body}</svg>'


def _fe_image(href: str) -> str:
    """A 40 px square drawing painted by the picture its filter's feImage names."""
    return _svg(
        f'<filter id="f"><feImage href="{href}"/></filter>'
        '<rect width="40" height="40" filter="url(#f)"/>'
    )


def _declared(encoding: str, drawing: str) -> str:
    """``drawing`` after an XML declaration that names ``encoding``."""
    return f'<?xml version="1.0" encoding="{encoding}"?>{drawing}'


# Width and height, or the viewBox where they are missing, as resvg reads them. A size in
# em or ex is measured by the svg element's own font size, its style attribute winning over
# its font-size attribute, and an !important declaration over a plain one.
OWN_SIZES = [
    pytest.param('width="320" height="200"', (320, 200), id="width-and-height"),
    pytest.param('viewBox="0 0 640 480"', (640, 480), id="viewbox-only"),
    pytest.param('width="200" viewBox="0 0 100 50"', (200, 100), id="width-and-viewbox"),
    pytest.param('height="30" viewBox="0 0 100 50"', (60, 30), id="height-and-viewbox"),
    pytest.param('width="50%" height="100%" viewBox="0 0 300 150"', (150, 150), id="percent-of-viewbox"),
    pytest.param('width="50mm" height="1in"', (189, 96), id="absolute-units"),
    pytest.param('width="100.6" height="50.4"', (101, 50), id="fractional"),
    pytest.param('width="4096" height="16"', (4096, 16), id="at-the-limit"),
    pytest.param('width="10ex" height="4ex" font-size="15pt"', (100, 40), id="ex-of-a-font-size-in-pt"),
    pytest.param('width="10em" height="2em" font-size="20" style="font-size: 30px"', (300, 60),
                 id="em-of-the-font-size-in-the-style-attribute"),
    pytest.param('width="5em" height="40" style="fill: none; font-size: 0.5in"', (240, 40),
                 id="em-and-px"),
    pytest.param('width="10em" viewBox="0 0 100 20" font-size="20"', (200, 40), id="em-and-viewbox"),
    # An important font size beats a plain one, wherever each stands; of two important
    # ones resvg keeps the first.
    pytest.param('width="10em" height="2em" style="font-size: 20px !important; font-size: 40px"', (200, 40),
                 id="em-of-an-important-font-size-before-a-plain-one"),
    pytest.param('width="10em" height="2em" style="font-size: 20px !important; font-size: 410px"', (200, 40),
                 id="em-of-an-important-font-size-before-one-above-the-limit"),
    pytest.param('width="10em" height="2em" style="font-size: 20px; font-size: 40px !important"', (400, 80),
                 id="em-of-an-important-font-size-after-a-plain-one"),
    pytest.param('width="10em" height="2em" style="font-size: 30px !important; font-size: 40px !important"',
                 (300, 60), id="em-of-the-first-of-two-important-font-sizes"),
    pytest.param('width="10em" height="2em" style="font: 20px serif"', (200, 40), id="em-of-the-font-shorthand"),
    pytest.param('width="10em" height="2em" font-size="30" style="font: 20px/1.5 serif"', (300, 60),
                 id="em-beside-a-font-shorthand-resvg-does-not-read"),
]

# Sized in em of the font size the drawing's own <style> rules give its svg element, read
# with the selectors that pick it and the cascade, as resvg reads them:
# (svg element attributes, style sheet, size).
SHEET_SIZES = [
    pytest.param("", "svg { font-size: 20px; }", (200, 40), id="svg-rule"),
    pytest.param('id="canvas"', "#canvas { font-size: 20px; }", (200, 40), id="id-rule"),
    pytest.param('class="wide canvas"', ".canvas { font-size: 20px; }", (200, 40), id="class-rule"),
    pytest.param("", "* { font-size: 20px; }", (200, 40), id="universal-rule"),
    pytest.param("", "rect, svg { font: bold 20px serif; }", (200, 40), id="font-shorthand-in-a-rule-list"),
    pytest.param('font-size="20"', "g > svg { font-size: 40px; }", (200, 40), id="rule-that-cannot-pick-the-root"),
    pytest.param('font-size="20"', "svg { font-size: 40px; }", (400, 80), id="rule-beats-the-attribute"),
    pytest.param('style="font-size: 30px"', "svg { font-size: 20px; }", (300, 60),
                 id="style-attribute-beats-a-rule"),
    pytest.param('style="font-size: 30px"', "svg { font-size: 20px !important; }", (200, 40),
                 id="important-rule-beats-the-style-attribute"),
    pytest.param('id="canvas"', "#canvas { font-size: 30px; } svg { font-size: 20px; }", (300, 60),
                 id="more-specific-rule-wins"),
    pytest.param("", "svg { font-size: 30px; } svg { font-size: 20px; }", (200, 40), id="later-rule-wins"),
    pytest.param("", "svg { font-size: 30px !important; } svg { font-size: 20px; }", (300, 60),
                 id="important-rule-beats-a-later-one"),
    pytest.param('font-size="20"', "@media print { svg { font-size: 40px; } }", (200, 40),
                 id="rule-inside-an-at-rule"),
    pytest.param('font-size="20"', "text ~ text { font-size: 40px; }", (200, 40),
                 id="rule-for-other-elements-only"),
    pytest.param('font-size="20"', "svg:hover { font-size: 40px; }", (200, 40),
                 id="rule-for-a-state-a-picture-is-never-in"),
    pytest.param('font-size="20"', "\\73 vg { font-size: 40px; } svg:first-child { font: 30px serif; }", (300, 60),
                 id="rule-whose-selector-resvg-skips"),
]

# A drawing in each encoding it may be written in: the one its byte order mark gives, or its
# first "<" written in two bytes, or else the one its XML declaration names. (byte order mark,
# Python's codec, the encoding the declaration names, a word in the encoding's own characters;
# windows-1252 writes € as a byte that ISO-8859-1 reads as another character)
ENCODINGS = [
    pytest.param(b"", "utf-8", "UTF-8", "café", id="utf-8"),
    pytest.param(codecs.BOM_UTF8, "utf-8", "UTF-8", "café", id="utf-8-with-a-byte-order-mark"),
    pytest.param(codecs.BOM_UTF16_BE, "utf-16-be", "UTF-16", "café", id="utf-16-big-endian-with-a-mark"),
    pytest.param(codecs.BOM_UTF16_LE, "utf-16-le", "UTF-16", "café", id="utf-16-little-endian-with-a-mark"),
    pytest.param(b"", "utf-16-be", "UTF-16BE", "café", id="utf-16be"),
    pytest.param(b"", "utf-16-le", "UTF-16LE", "café", id="utf-16le"),
    pytest.param(b"", "latin-1", "ISO-8859-1", "café", id="iso-8859-1"),
    pytest.param(b"", "cp1252", "windows-1252", "café€", id="windows-1252"),
]

# (filename, content, content type, a fragment of the plain reason)
REFUSED = [
    # A script element, in the SVG namespace or any other.
    pytest.param("script.svg", _svg(f"{RECT}<script>alert(1)</script>"), SVG_TYPE,
                 "holds a script element", id="script"),
    pytest.param("xhtml-script.svg",
                 _svg(f'{RECT}<html:script xmlns:html="http://www.w3.org/1999/xhtml">alert(1)</html:script>'),
                 SVG_TYPE, "holds a script element", id="script-in-another-namespace"),
    pytest.param("handler.svg", _svg(f'{RECT}<handler type="application/ecmascript">alert(1)</handler>'),
                 SVG_TYPE, "holds a script element", id="handler"),
    # Any event attribute.
    pytest.param("onload.svg", f'<svg {NS} width="40" height="40" onload="alert(1)">{RECT}</svg>', SVG_TYPE,
                 "has an event attribute", id="onload"),
    pytest.param("onclick.svg", _svg('<rect width="40" height="40" onclick="alert(1)"/>'), SVG_TYPE,
                 "has an event attribute", id="onclick"),
    # Any external reference.
    pytest.param("use-web.svg", _svg('<use href="https://example.com/parts.svg#logo"/>'), SVG_TYPE,
                 "has an external reference", id="use-href-web"),
    pytest.param("use-file.svg", _svg('<use xlink:href="parts.svg#logo"/>'), SVG_TYPE,
                 "has an external reference", id="use-xlink-href-file"),
    pytest.param("link.svg", _svg(f'<a href="https://example.com/">{RECT}</a>'), SVG_TYPE,
                 "has an external reference", id="link"),
    pytest.param("animated-link.svg",
                 _svg('<defs><rect id="card" width="40" height="40" fill="#2050c0"/></defs>'
                      '<use href="#card"><set attributeName="href" to="https://example.com/parts.svg#logo"/></use>'),
                 SVG_TYPE, "has an external reference", id="href-set-by-animation"),
    pytest.param("fill.svg", _svg('<rect width="40" height="40" fill="url(https://example.com/paint.svg#g)"/>'),
                 SVG_TYPE, "has an external reference", id="url-in-attribute"),
    pytest.param("style-attribute.svg",
                 _svg('<rect width="40" height="40" style="fill: url(\'file:///etc/paint.svg#g\')"/>'),
                 SVG_TYPE, "has an external reference", id="url-in-style-attribute"),
    pytest.param("style-element.svg", _svg(f"<style>rect {{ fill: url(paint.svg#g); }}</style>{RECT}"),
                 SVG_TYPE, "has an external reference", id="url-in-style-element"),
    pytest.param("escaped.svg", _svg(r'<rect width="40" height="40" style="fill: \75 rl(paint.svg#g)"/>'),
                 SVG_TYPE, "has an external reference", id="url-written-with-a-css-escape"),
    pytest.param("font.svg",
                 _svg("<style>@font-face { font-family: Brand; src: url(https://example.com/brand.woff2); }</style>"
                      f"{RECT}"),
                 SVG_TYPE, "has an external reference", id="url-of-a-font"),
    pytest.param("import.svg", _svg(f"<style>@import url(https://example.com/site.css);</style>{RECT}"),
                 SVG_TYPE, "has an external reference", id="css-import-url"),
    pytest.param("import-string.svg", _svg(f'<style>@import "site.css";</style>{RECT}'), SVG_TYPE,
                 "has an external reference", id="css-import-string"),
    pytest.param("entity.svg",
                 '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY ext SYSTEM "file:///etc/hostname">]>'
                 + _svg(f'{RECT}<text y="20">&ext;</text>'),
                 SVG_TYPE, "has an external reference", id="external-entity"),
    pytest.param("dtd.svg",
                 '<?xml version="1.0"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
                 '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">' + _svg(RECT),
                 SVG_TYPE, "has an external reference", id="external-dtd"),
    pytest.param("stylesheet.svg",
                 '<?xml-stylesheet href="https://example.com/site.css" type="text/css"?>' + _svg(RECT),
                 SVG_TYPE, "has an external reference", id="xml-stylesheet"),
    # Declarations that would make the drawing read differently by another parser.
    pytest.param("internal-entity.svg",
                 '<!DOCTYPE svg [<!ENTITY brand "Raphael">]>' + _svg(f'{RECT}<text y="20">&brand;</text>'),
                 SVG_TYPE, "declares the entity brand", id="internal-entity"),
    pytest.param("defaults.svg",
                 '<!DOCTYPE svg [<!ATTLIST svg width CDATA "40" height CDATA "40">]>' + f"<svg {NS}>{RECT}</svg>",
                 SVG_TYPE, "declares default attributes", id="attribute-defaults"),
    # A foreignObject element.
    pytest.param("foreign.svg",
                 _svg('<foreignObject width="40" height="40">'
                      '<div xmlns="http://www.w3.org/1999/xhtml">Home</div></foreignObject>'),
                 SVG_TYPE, "holds a foreignObject element", id="foreign-object"),
    # An embedded image: an image element, or a data: reference to a raster or a nested SVG.
    pytest.param("image.svg", _svg(f'<image href="{PICTURE_URI}" width="40" height="40"/>'), SVG_TYPE,
                 "holds an embedded image", id="image-element"),
    pytest.param("fe-image.svg", _fe_image(NESTED_URI), SVG_TYPE,
                 "holds an embedded image", id="data-reference-to-a-nested-svg"),
    pytest.param("data-fill.svg", _svg(f'<rect width="40" height="40" fill="url({PICTURE_URI})"/>'), SVG_TYPE,
                 "holds an embedded image", id="data-url-of-a-raster"),
    pytest.param("data-use.svg", _svg(f'<use href="{NESTED_URI}#x"/>'), SVG_TYPE,
                 "holds an embedded image", id="data-href-of-use"),
    # A width or height above 4,096 px.
    pytest.param("wide.svg", f'<svg {NS} width="4097" height="100">{RECT}</svg>', SVG_TYPE,
                 "is 4,097x100 px (at most 4,096 px per side", id="width-above-the-limit"),
    pytest.param("tall.svg", f'<svg {NS} viewBox="0 0 100 5000">{RECT}</svg>', SVG_TYPE,
                 "is 100x5,000 px (at most 4,096 px per side", id="viewbox-height-above-the-limit"),
    pytest.param("inches.svg", f'<svg {NS} width="43in" height="1in">{RECT}</svg>', SVG_TYPE,
                 "is 4,128x96 px (at most 4,096 px per side", id="inches-above-the-limit"),
    pytest.param("em-wide.svg", f'<svg {NS} width="300em" height="1em" font-size="20">{RECT}</svg>', SVG_TYPE,
                 "is 6,000x20 px (at most 4,096 px per side", id="em-above-the-limit"),
    # The important font size wins over the plain one after it: 10em of 410 px.
    pytest.param("important-font.svg",
                 f'<svg {NS} width="10em" height="2em" style="font-size: 410px !important; font-size: 20px">'
                 f"{RECT}</svg>",
                 SVG_TYPE, "is 4,100x820 px (at most 4,096 px per side",
                 id="em-of-an-important-font-size-above-the-limit"),
    pytest.param("sheet-wide.svg",
                 f'<svg {NS} width="10em" height="2em" font-size="20"><style>svg {{ font-size: 410px; }}</style>'
                 f"{RECT}</svg>",
                 SVG_TYPE, "is 4,100x820 px (at most 4,096 px per side", id="em-of-a-rules-font-size-above-the-limit"),
    # No size that can be determined.
    pytest.param("unsized.svg", f"<svg {NS}>{RECT}</svg>", SVG_TYPE, "has no size", id="no-size"),
    pytest.param("width-only.svg", f'<svg {NS} width="40">{RECT}</svg>', SVG_TYPE, "has no size",
                 id="width-without-height-or-viewbox"),
    pytest.param("percent.svg", f'<svg {NS} width="100%" height="100%">{RECT}</svg>', SVG_TYPE, "has no size",
                 id="percent-without-viewbox"),
    # A size in em or ex with no plain font size to measure it by: none, or one that is
    # relative, a keyword or inherited, wherever it is declared.
    pytest.param("font-units.svg", f'<svg {NS} width="10em" height="2em">{RECT}</svg>', SVG_TYPE,
                 "in em or ex units but no plain font size", id="em-without-a-font-size"),
    pytest.param("percent-font.svg", f'<svg {NS} width="10ex" height="10" font-size="200%">{RECT}</svg>',
                 SVG_TYPE, "in em or ex units but no plain font size", id="font-size-in-percent"),
    pytest.param("em-font.svg", f'<svg {NS} width="10em" height="2em" font-size="2em">{RECT}</svg>', SVG_TYPE,
                 "in em or ex units but no plain font size", id="font-size-in-em"),
    pytest.param("medium-font.svg", f'<svg {NS} width="10em" height="2em" font-size="medium">{RECT}</svg>',
                 SVG_TYPE, "in em or ex units but no plain font size", id="font-size-keyword"),
    pytest.param("larger-font.svg",
                 f'<svg {NS} width="10em" height="2em" font-size="20" style="font-size: larger">{RECT}</svg>',
                 SVG_TYPE, "in em or ex units but no plain font size", id="relative-font-size-in-the-style"),
    pytest.param("shorthand-font.svg", f'<svg {NS} width="10em" height="2em" style="font: 2em serif">{RECT}</svg>',
                 SVG_TYPE, "in em or ex units but no plain font size", id="relative-font-size-in-the-font-shorthand"),
    pytest.param("inherited-font.svg",
                 f'<svg {NS} width="10em" height="2em"><style>svg {{ font-size: inherit; }}</style>{RECT}</svg>',
                 SVG_TYPE, "in em or ex units but no plain font size", id="inherited-font-size-from-a-rule"),
    # resvg reads lengths as 32-bit floats: to it 2.4999999 px is 2.5, a 3 px side, so
    # the PNG would not be the 2 px side the drawing gives, and is not stored.
    pytest.param("rounding.svg", f'<svg {NS} width="2.4999999" height="10">{RECT}</svg>', SVG_TYPE,
                 "could not be rendered at its own size of 2x10 px", id="png-not-at-its-own-size"),
    # A resulting PNG larger than 4 MiB.
    pytest.param("noise.svg", NOISE, SVG_TYPE, "(at most 4 MiB is stored)", id="png-above-4-mib"),
    pytest.param("noise-em.svg", NOISE_IN_EM, SVG_TYPE, "(at most 4 MiB is stored)", id="png-in-em-above-4-mib"),
    # Not an SVG drawing: judged from the stored bytes, never from the name or type.
    pytest.param("drawing.png", PICTURE, "image/png", "is not an SVG drawing", id="png-picture"),
    pytest.param("notes.txt", "Draw the home page with a wide photo at the top.", "text/plain",
                 "is not an SVG drawing", id="text"),
    pytest.param("page.svg", '<html xmlns="http://www.w3.org/1999/xhtml"><body>Home</body></html>', SVG_TYPE,
                 "is not an SVG drawing", id="xhtml-page-named-svg"),
    pytest.param("bare.svg", f'<svg width="40" height="40">{RECT}</svg>', SVG_TYPE,
                 "is not an SVG drawing", id="svg-without-its-namespace"),
    pytest.param("broken.svg", f'<svg {NS} width="40" height="40"><rect></svg>', SVG_TYPE,
                 "is not an SVG drawing", id="not-well-formed"),
]


# ---------------------------------------------------------------------------
# Copied verbatim from tests/tools/test_kanban_run_ending_receipt.py (copied,
# not imported, so this file stands on its own).
# ---------------------------------------------------------------------------

def _spawned_worker_env(task, workspace: Path, monkeypatch) -> dict:
    """The environment ``_default_spawn`` gives the worker process it launches.

    Only the launch is stubbed: the stub records the argv and environment the
    dispatcher built and returns a pid, so no process is started.
    """
    launches = []

    class _Launched:
        pid = 4242

    def _launch(argv, **kwargs):
        log = kwargs.get("stdout")
        if hasattr(log, "close"):
            log.close()
        launches.append((list(argv), dict(kwargs.get("env") or {})))
        return _Launched()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", _launch)
        pid = kb._default_spawn(task, str(workspace), board=OWN)
    assert pid == _Launched.pid
    assert len(launches) == 1, launches
    argv, env = launches[0]
    assert f"work kanban task {task.id}" in argv
    return env


def _become(env: dict, monkeypatch) -> None:
    """Carry on as that process: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


def _assert_root_in_reach(root: Path) -> None:
    """From the worker's environment the root registry and every board are in reach.

    The kernel's answers are read against the registry file's own rows,
    counted with plain SQL.
    """
    registry = root / "kanban" / "board_register.db"
    assert kb.register_db_path().resolve() == registry.resolve()
    with closing(sqlite3.connect(registry)) as db:
        registered = {
            row[0] for row in db.execute("SELECT board_name FROM board_register")
        }
    assert {OWN, OTHER} <= registered
    listed = {board["slug"] for board in kb.list_boards()}
    assert listed - {kb.DEFAULT_BOARD} == registered - {kb.DEFAULT_BOARD}
    for slug in registered:
        entry = kb.get_register_entry(slug)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE, slug
        assert kb.board_dir(slug).resolve() == (root / "kanban" / "boards" / slug).resolve()


# ---------------------------------------------------------------------------
# Copied from tests/tools/test_kanban_design_options.py.
# ---------------------------------------------------------------------------

def _store_path(slug):
    """The board's own store file, resolved from its slug alone."""
    if slug == kb.DEFAULT_BOARD:
        return kb.kanban_home() / "kanban.db"
    return kb.board_dir(slug) / "kanban.db"


def _claimed_card(slug, title):
    """One card on board ``slug``, claimed by the dispatcher and running."""
    conn = kb.connect(board=slug)
    try:
        task_id = kb.create_task(conn, title=title, assignee="designer", board=slug)
        claimed = kb.claim_task(conn, task_id)
    finally:
        conn.close()
    assert claimed is not None and claimed.status == "running"
    return claimed


def _read(store, query, *params):
    """Rows of ``query`` as a separate read-only connection to ``store`` sees them."""
    conn = sqlite3.connect(f"{Path(store).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _dispatch(name, args):
    from tools.registry import registry

    return json.loads(registry.dispatch(name, args))


def _options(*attachment_ids):
    sentences = (
        "A calm layout with a large photo at the top.",
        "A bold layout with a big title and two buttons.",
    )
    alts = (
        "The home page with a wide photo above a short welcome.",
        "The home page with a large title and two buttons below it.",
    )
    return [
        {
            "n": n,
            "sentence": sentences[n - 1],
            "pictures": [{"attachment_id": str(attachment_id), "alt": alts[n - 1]}],
        }
        for n, attachment_id in enumerate(attachment_ids, start=1)
    ]


# ---------------------------------------------------------------------------
# Helpers for this file
# ---------------------------------------------------------------------------

def _bytes(data) -> bytes:
    return data.encode("utf-8") if isinstance(data, str) else data


def _attach(filename, data, content_type=SVG_TYPE):
    """The designer attaches one file to its own card with ``kanban_attach``."""
    out = _dispatch("kanban_attach", {
        "filename": filename,
        "content_base64": base64.b64encode(_bytes(data)).decode("ascii"),
        "content_type": content_type,
    })
    assert out.get("ok") is True, out
    return out["attachment_id"]


def _store(slug, card, filename, data, content_type=SVG_TYPE):
    """The board's owner attaches one file to ``card`` through the kernel."""
    conn = kb.connect(board=slug)
    try:
        return kb.store_attachment_bytes(
            conn, card.id, filename, _bytes(data),
            content_type=content_type, uploaded_by="owner", board=slug,
        )
    finally:
        conn.close()


def _state(worker):
    """Every attachment row of the three boards, read with plain SQL, and every
    attachment file under the root with its size, counted on disk."""
    rows = {
        slug: _read(
            store,
            "SELECT id, task_id, filename, content_type, size, uploaded_by "
            "FROM task_attachments ORDER BY id",
        )
        for slug, store in worker.stores.items()
    }
    base = worker.root.resolve()
    files = sorted(
        (path.relative_to(base).as_posix(), path.stat().st_size)
        for path in base.rglob("*")
        if path.is_file() and "attachments" in path.relative_to(base).parts
    )
    return rows, files


def _stored_picture(worker, attachment_id):
    """The PNG stored for ``attachment_id`` on the worker's own board, decoded from its file."""
    ((stored_path,),) = _read(
        worker.stores[OWN], "SELECT stored_path FROM task_attachments WHERE id = ?", attachment_id,
    )
    with Image.open(stored_path) as picture:
        assert picture.format == "PNG"
        return picture.convert("RGBA")


def _resvg_size(svg):
    """The size resvg itself gives ``svg``, rendered with no output size asked for."""
    import resvg_py

    with Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=svg, dpi=96.0))) as picture:
        return picture.size


def _design_vision_schema_names():
    """Tool names of the design_vision toolset this caller actually sees."""
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("design_vision")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    invalidate_check_fn_cache()
    return names


_PROCESS_VARS = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_PROFILE",
    "HERMES_SESSION_ID",
)


def _fresh_process(home: Path, monkeypatch) -> None:
    """This process names no card, run, profile or session of its own, and its Hermes root
    is ``home``; the kanban modules remember no board."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in _PROCESS_VARS:
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()
    kb._ARCHIVE_INITIALIZED_PATHS.clear()


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    """The Hermes root every test starts from, built and launched once for the module.

    A Hermes root with three live boards (the default one, the worker's own and another one):
    the worker's own card, a sibling card with a drawing, and on each other board a card with a
    picture and a drawing, so each of their drawings has the id the worker's next attachment
    gets on its own board. The worker's card is prepared as the dispatcher does and launched
    through ``_default_spawn`` (only the process launch stubbed), and the positive controls
    run. Building a root takes seconds of disk syncs, so it is kept as a pristine copy that
    each test gets back byte for byte at the same paths (the drawings' stored paths and the
    card's workspace are in the boards).
    """
    from tools.registry import discover_builtin_tools

    base = tmp_path_factory.mktemp("world")
    home = base / "hermes_home"
    home.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        hermes_state = sys.modules.get("hermes_state")
        if hermes_state is not None and hasattr(hermes_state, "DEFAULT_DB_PATH"):
            patch.setattr(hermes_state, "DEFAULT_DB_PATH", base / "state.db")
        _fresh_process(home, patch)

        kb.init_db()
        for slug in (OWN, OTHER):
            kb.create_board(slug)
            entry = kb.get_register_entry(slug)
            assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE

        mine = _claimed_card(OWN, "home page design")
        sibling = _claimed_card(OWN, "menu page design")
        foreign = _claimed_card(OTHER, "shop page design")
        default_card = _claimed_card(kb.DEFAULT_BOARD, "about page design")
        stores = {slug: _store_path(slug) for slug in (kb.DEFAULT_BOARD, OWN, OTHER)}
        drawings = {"sibling": _store(OWN, sibling, "menu.svg", PLAIN)}
        for name, slug, card in (("foreign", OTHER, foreign), ("default_card", kb.DEFAULT_BOARD, default_card)):
            _store(slug, card, "photo.png", PICTURE, "image/png")
            drawings[name] = _store(slug, card, "shop.svg", PLAIN)

        (home / "profiles" / mine.assignee).mkdir(parents=True)
        conn = kb.connect(board=OWN)
        try:
            workspace = kb.resolve_workspace(mine, board=OWN)
            kb.set_workspace_path(conn, mine.id, str(workspace))
        finally:
            conn.close()
        inherited = dict(os.environ)
        env = _spawned_worker_env(mine, Path(workspace), patch)
        # What the launch sets, and what it takes out, of the environment it inherits.
        launch_sets = {name: value for name, value in env.items() if inherited.get(name) != value}
        launch_drops = [name for name in inherited if name not in env]

        # The positive controls, which also leave the tool discovery cache warm in the copy.
        _become(env, patch)
        assert kb.kanban_db_path(board=OTHER).resolve() == stores[OWN].resolve()
        assert "tools.kanban_render_drawing" in discover_builtin_tools()
        assert TOOL in _design_vision_schema_names()

    pristine = base / "pristine"
    shutil.copytree(home, pristine, symlinks=True)
    return SimpleNamespace(
        base=base, home=home, pristine=pristine, stores=stores, mine=mine, sibling=sibling,
        foreign=foreign, default_card=default_card, drawings=drawings,
        launch_sets=launch_sets, launch_drops=launch_drops,
    )


@pytest.fixture
def root(_world, monkeypatch):
    """A Hermes root with three live boards: the default one, the worker's own
    and another one.

    The test process plays the dispatcher: it names no card, run, profile or
    session of its own.
    """
    shutil.rmtree(_world.home)
    shutil.copytree(_world.pristine, _world.home, symlinks=True)
    _fresh_process(_world.home, monkeypatch)
    for slug in (OWN, OTHER):
        entry = kb.get_register_entry(slug)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE
    return _world.home


@pytest.fixture
def worker(_world, root, monkeypatch):
    """This process becomes the designer's worker for its card on ``own-board``.

    The worker's own board also holds a sibling card with a drawing. The other
    board and the default board each hold a card with a picture and a drawing,
    so each of their drawings has the id the worker's next attachment gets on
    its own board. The card is prepared as the dispatcher does, and this
    process takes the environment ``_default_spawn`` gives its worker.
    """
    from tools.registry import discover_builtin_tools

    mine, stores = _world.mine, _world.stores
    env = dict(os.environ)
    env.update(_world.launch_sets)
    for name in _world.launch_drops:
        env.pop(name, None)

    # Pinned to its own card, run and board, under its profile home below the
    # root; the session id is the worker's own to publish once it starts.
    assert env["HERMES_KANBAN_TASK"] == mine.id
    assert env["HERMES_KANBAN_RUN_ID"] == str(mine.current_run_id)
    assert env["HERMES_KANBAN_BOARD"] == OWN
    assert env["HERMES_KANBAN_DB"] == str(root / "kanban" / "boards" / OWN / "kanban.db")
    assert env["HERMES_HOME"] == str(root / "profiles" / mine.assignee)
    assert env["HERMES_PROFILE"] == mine.assignee
    assert "HERMES_SESSION_ID" not in env
    for name in (
        "HERMES_HOME",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_WORKSPACE",
    ):
        assert Path(env[name]).resolve().is_relative_to(_world.base.resolve()), name

    _become(env, monkeypatch)
    _assert_root_in_reach(root)

    # Positive controls: the pin is live (the resolver hands back the own
    # store for any slug), the built-in discovery imports the tool's module
    # by name, and the worker's design_vision schema lists the tool.
    assert kb.kanban_db_path(board=OTHER).resolve() == stores[OWN].resolve()
    assert "tools.kanban_render_drawing" in discover_builtin_tools()
    assert TOOL in _design_vision_schema_names()
    return SimpleNamespace(
        root=root, profile_home=Path(env["HERMES_HOME"]), stores=stores, mine=mine,
        sibling=_world.sibling, foreign=_world.foreign, default_card=_world.default_card,
        drawings=_world.drawings,
    )


def test_a_plain_drawing_becomes_a_png_of_its_own_size_that_the_offer_accepts(worker):
    layout = _attach("layout.svg", PLAIN)
    wide = _attach("wide.svg", VIEWBOX_ONLY)
    before_rows, before_files = _state(worker)

    rendered = [_dispatch(TOOL, {"attachment_id": drawing}) for drawing in (layout, wide)]

    for out in rendered:
        assert out.get("ok") is True, out
        assert set(out) == {"ok", "task_id", "attachment_id", "width", "height", "size"}
        assert out["task_id"] == worker.mine.id
    assert [(out["width"], out["height"]) for out in rendered] == [(320, 200), (640, 480)]
    after_rows, after_files = _state(worker)
    # Only the worker's own board changed: it gained one PNG per drawing, on
    # the worker's card, and the file of each under that card's directory.
    assert {slug: after_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)} == {
        slug: before_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)
    }
    assert after_rows[OWN][:len(before_rows[OWN])] == before_rows[OWN]
    assert after_rows[OWN][len(before_rows[OWN]):] == [
        (out["attachment_id"], worker.mine.id, name, "image/png", out["size"], TOOL)
        for out, name in zip(rendered, ("layout.png", "wide.png"))
    ]
    card_dir = f"kanban/boards/{OWN}/attachments/{worker.mine.id}"
    assert set(before_files) <= set(after_files)
    assert sorted(set(after_files) - set(before_files)) == [
        (f"{card_dir}/layout.png", rendered[0]["size"]),
        (f"{card_dir}/wide.png", rendered[1]["size"]),
    ]
    layout_png = _stored_picture(worker, rendered[0]["attachment_id"])
    assert layout_png.size == (320, 200)
    assert layout_png.getpixel((300, 10)) == BLUE  # the panel, painted by the style's gradient
    assert layout_png.getpixel((70, 50)) == RED  # the shape used through href
    assert layout_png.getpixel((250, 150)) == RED  # the shape used through xlink:href
    wide_png = _stored_picture(worker, rendered[1]["attachment_id"])
    assert wide_png.size == (640, 480) and wide_png.getpixel((320, 240)) == GREEN
    assert not (worker.profile_home / "kanban").exists()

    offered = _dispatch(OFFER, {
        "question": QUESTION, "options": _options(*(out["attachment_id"] for out in rendered)),
    })

    assert offered == {
        "ok": True,
        "task_id": worker.mine.id,
        "status": "blocked",
        "block_kind": "needs_input",
        "declared": True,
    }


def test_a_drawing_sized_in_em_or_ex_of_its_own_font_size_becomes_a_png_that_the_offer_accepts(worker):
    # resvg itself gives the two drawings these sizes.
    assert [_resvg_size(svg) for svg in (EM, EX)] == [(200, 40), (100, 40)]
    em = _attach("em.svg", EM)
    ex = _attach("ex.svg", EX)
    before_rows, before_files = _state(worker)

    rendered = [_dispatch(TOOL, {"attachment_id": drawing}) for drawing in (em, ex)]

    for out in rendered:
        assert out.get("ok") is True, out
        assert set(out) == {"ok", "task_id", "attachment_id", "width", "height", "size"}
        assert out["task_id"] == worker.mine.id
    assert [(out["width"], out["height"]) for out in rendered] == [(200, 40), (100, 40)]
    after_rows, after_files = _state(worker)
    # Only the worker's own board changed: it gained one PNG per drawing, on
    # the worker's card, and the file of each under that card's directory.
    assert {slug: after_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)} == {
        slug: before_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)
    }
    assert after_rows[OWN][:len(before_rows[OWN])] == before_rows[OWN]
    assert after_rows[OWN][len(before_rows[OWN]):] == [
        (out["attachment_id"], worker.mine.id, name, "image/png", out["size"], TOOL)
        for out, name in zip(rendered, ("em.png", "ex.png"))
    ]
    card_dir = f"kanban/boards/{OWN}/attachments/{worker.mine.id}"
    assert set(before_files) <= set(after_files)
    assert sorted(set(after_files) - set(before_files)) == [
        (f"{card_dir}/em.png", rendered[0]["size"]),
        (f"{card_dir}/ex.png", rendered[1]["size"]),
    ]
    em_png = _stored_picture(worker, rendered[0]["attachment_id"])
    assert em_png.size == (200, 40) and em_png.getpixel((100, 20)) == BLUE
    ex_png = _stored_picture(worker, rendered[1]["attachment_id"])
    assert ex_png.size == (100, 40) and ex_png.getpixel((50, 20)) == GREEN
    assert not (worker.profile_home / "kanban").exists()

    offered = _dispatch(OFFER, {
        "question": QUESTION, "options": _options(*(out["attachment_id"] for out in rendered)),
    })

    assert offered == {
        "ok": True,
        "task_id": worker.mine.id,
        "status": "blocked",
        "block_kind": "needs_input",
        "declared": True,
    }


def _assert_rendered_at_its_own_size(worker, svg, pixels):
    """The tool renders ``svg`` as a PNG of the size resvg itself gives it, ``pixels``."""
    assert _resvg_size(svg) == pixels
    drawing = _attach("size.svg", svg)

    out = _dispatch(TOOL, {"attachment_id": drawing})

    assert out.get("ok") is True, out
    assert (out["width"], out["height"]) == pixels
    picture = _stored_picture(worker, out["attachment_id"])
    assert picture.size == pixels
    assert picture.getpixel((pixels[0] // 2, pixels[1] // 2)) == BLUE


def test_a_drawing_sized_by_its_own_css_becomes_a_png_that_the_offer_accepts(worker):
    # resvg itself gives the two drawings these sizes.
    assert [_resvg_size(svg) for svg in (SHEET, SHORTHAND)] == [(400, 80), (200, 80)]
    sheet = _attach("sheet.svg", SHEET)
    shorthand = _attach("shorthand.svg", SHORTHAND)
    before_rows, before_files = _state(worker)

    rendered = [_dispatch(TOOL, {"attachment_id": drawing}) for drawing in (sheet, shorthand)]

    for out in rendered:
        assert out.get("ok") is True, out
        assert set(out) == {"ok", "task_id", "attachment_id", "width", "height", "size"}
        assert out["task_id"] == worker.mine.id
    assert [(out["width"], out["height"]) for out in rendered] == [(400, 80), (200, 80)]
    after_rows, after_files = _state(worker)
    # Only the worker's own board changed: it gained one PNG per drawing, on
    # the worker's card, and the file of each under that card's directory.
    assert {slug: after_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)} == {
        slug: before_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)
    }
    assert after_rows[OWN][:len(before_rows[OWN])] == before_rows[OWN]
    assert after_rows[OWN][len(before_rows[OWN]):] == [
        (out["attachment_id"], worker.mine.id, name, "image/png", out["size"], TOOL)
        for out, name in zip(rendered, ("sheet.png", "shorthand.png"))
    ]
    card_dir = f"kanban/boards/{OWN}/attachments/{worker.mine.id}"
    assert set(before_files) <= set(after_files)
    assert sorted(set(after_files) - set(before_files)) == [
        (f"{card_dir}/sheet.png", rendered[0]["size"]),
        (f"{card_dir}/shorthand.png", rendered[1]["size"]),
    ]
    sheet_png = _stored_picture(worker, rendered[0]["attachment_id"])
    assert sheet_png.size == (400, 80) and sheet_png.getpixel((200, 40)) == BLUE
    shorthand_png = _stored_picture(worker, rendered[1]["attachment_id"])
    assert shorthand_png.size == (200, 80) and shorthand_png.getpixel((100, 40)) == GREEN
    assert not (worker.profile_home / "kanban").exists()

    offered = _dispatch(OFFER, {
        "question": QUESTION, "options": _options(*(out["attachment_id"] for out in rendered)),
    })

    assert offered == {
        "ok": True,
        "task_id": worker.mine.id,
        "status": "blocked",
        "block_kind": "needs_input",
        "declared": True,
    }


@pytest.mark.parametrize(("size", "pixels"), OWN_SIZES)
def test_the_png_has_the_drawings_own_size(worker, size, pixels):
    _assert_rendered_at_its_own_size(
        worker, f'<svg {NS} {size}><rect width="100%" height="100%" fill="#2050c0"/></svg>', pixels,
    )


@pytest.mark.parametrize(("attributes", "sheet", "pixels"), SHEET_SIZES)
def test_the_png_has_the_size_the_drawings_own_style_sheet_gives(worker, attributes, sheet, pixels):
    _assert_rendered_at_its_own_size(
        worker,
        f'<svg {NS} width="10em" height="2em" {attributes}><style>{sheet}</style>'
        '<rect width="100%" height="100%" fill="#2050c0"/></svg>',
        pixels,
    )


@pytest.mark.parametrize(("mark", "codec", "encoding", "word"), ENCODINGS)
def test_a_drawing_in_its_own_encoding_becomes_a_png_of_its_own_size(worker, mark, codec, encoding, word):
    # The blue rectangle is drawn only through an href that holds the word, and its id gives
    # the word in character references, so it is painted only when the drawing that was
    # checked and rendered was read in its own encoding.
    shape = "".join(char if char.isascii() else f"&#{ord(char)};" for char in word)
    drawing = _attach("encoded.svg", mark + _declared(encoding, (
        f'<svg {NS} width="320" height="200"><defs><rect id="{shape}" width="320" height="200" '
        f'fill="#2050c0"/></defs><use href="#{word}"/></svg>'
    )).encode(codec))
    before_rows, before_files = _state(worker)

    out = _dispatch(TOOL, {"attachment_id": drawing})

    assert out.get("ok") is True, out
    assert (out["task_id"], out["width"], out["height"]) == (worker.mine.id, 320, 200)
    after_rows, after_files = _state(worker)
    # Only the worker's own board changed: it gained the PNG, on the worker's card, and its
    # file under that card's directory.
    assert {slug: after_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)} == {
        slug: before_rows[slug] for slug in (kb.DEFAULT_BOARD, OTHER)
    }
    assert after_rows[OWN] == before_rows[OWN] + [
        (out["attachment_id"], worker.mine.id, "encoded.png", "image/png", out["size"], TOOL)
    ]
    assert set(before_files) <= set(after_files)
    assert sorted(set(after_files) - set(before_files)) == [
        (f"kanban/boards/{OWN}/attachments/{worker.mine.id}/encoded.png", out["size"]),
    ]
    picture = _stored_picture(worker, out["attachment_id"])
    assert picture.size == (320, 200) and picture.getpixel((160, 100)) == BLUE


@pytest.mark.parametrize(("filename", "content", "content_type", "reason"), REFUSED)
def test_a_refused_drawing_leaves_no_attachment_and_gives_a_plain_reason(
    worker, filename, content, content_type, reason,
):
    drawing = _attach(filename, content, content_type)
    before = _state(worker)

    out = _dispatch(TOOL, {"attachment_id": drawing})

    assert "ok" not in out and reason in out["error"], out
    assert out["error"].startswith(f"attachment {drawing} "), out
    assert out["error"].endswith("; nothing is stored"), out
    assert _state(worker) == before


def test_a_drawing_in_another_encoding_is_still_checked_and_one_that_cannot_be_read_is_refused(worker):
    entity = ('<!DOCTYPE svg [<!ENTITY ext SYSTEM "file:///etc/hostname">]>'
              + _svg(f'{RECT}<text y="20">&ext;</text>'))
    drawings = [
        # Read in its own encoding, a drawing is checked as any other.
        (codecs.BOM_UTF16_LE + _svg(f"{RECT}<script>alert(1)</script>").encode("utf-16-le"),
         "holds a script element"),
        (_declared("UTF-16BE", entity).encode("utf-16-be"), "has an external reference (the external entity ext)"),
        (_declared("ISO-8859-1", _svg(f'<a href="https://example.com/café">{RECT}</a>')).encode("latin-1"),
         "has an external reference"),
        # Its byte order mark and its XML declaration give different encodings.
        (codecs.BOM_UTF16_LE + _declared("UTF-8", _svg(RECT)).encode("utf-16-le"),
         "is not an SVG drawing (it is in UTF-16LE, but its XML declaration names UTF-8)"),
        (codecs.BOM_UTF8 + _declared("ISO-8859-1", _svg(RECT)).encode("utf-8"),
         "is not an SVG drawing (it is in UTF-8, but its XML declaration names ISO-8859-1)"),
        # A byte its encoding has no character for, or half of a UTF-16 character.
        (_declared("US-ASCII", _svg(f"<title>Café</title>{RECT}")).encode("latin-1"),
         "is not an SVG drawing (it is not US-ASCII text)"),
        (_declared("UTF-16LE", _svg(RECT)).encode("utf-16-le")[:-1],
         "is not an SVG drawing (it is not UTF-16LE text)"),
        # An encoding that is not read: an unknown one, one of more than one byte per character,
        # or UTF-16 named by a drawing written in single bytes.
        (_declared("x-unknown", _svg(RECT)).encode("ascii"),
         "is not an SVG drawing (it cannot be read in x-unknown, the encoding its XML declaration names"),
        (_declared("Shift_JIS", _svg(f"<title>カフェ</title>{RECT}")).encode("shift_jis"),
         "is not an SVG drawing (it cannot be read in Shift_JIS,"),
        (_declared("UTF-16", _svg(RECT)).encode("ascii"), "is not an SVG drawing (it cannot be read in UTF-16,"),
    ]
    attached = [_attach(f"encoding-{n}.svg", data) for n, (data, _) in enumerate(drawings)]
    before = _state(worker)

    for drawing, (_, reason) in zip(attached, drawings):
        out = _dispatch(TOOL, {"attachment_id": drawing})
        assert "ok" not in out and reason in out["error"], out
        assert out["error"].startswith(f"attachment {drawing} "), out
        assert out["error"].endswith("; nothing is stored"), out

    assert _state(worker) == before


@pytest.mark.parametrize(("args", "reason"), [
    pytest.param({}, "attachment_id must be a positive whole number", id="missing"),
    pytest.param({"attachment_id": "1"}, "attachment_id must be a positive whole number", id="text"),
    pytest.param({"attachment_id": 0}, "attachment_id must be a positive whole number", id="zero"),
    pytest.param({"attachment_id": True}, "attachment_id must be a positive whole number", id="true"),
    pytest.param({"attachment_id": 999}, "attachment 999 is not on your own task", id="unknown"),
])
def test_a_missing_or_unknown_attachment_id_is_refused(worker, args, reason):
    _attach("layout.svg", PLAIN)
    before = _state(worker)

    out = _dispatch(TOOL, args)

    assert "ok" not in out and reason in out["error"], out
    assert _state(worker) == before


def _without_the_card(patch):
    """The same process without the dispatcher's card: an orchestrator or a chat."""
    patch.delenv("HERMES_KANBAN_TASK")
    return nullcontext()


def _as_a_delegated_child(patch):
    """A delegate_task child, run inside the worker's own process."""
    return delegated_child_context()


def _as_a_delegated_childs_process(patch):
    """A process a delegate_task child started: it carries the child marker."""
    patch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    return nullcontext()


def _as_a_cron_run_in_the_worker(patch):
    """A cron job fired inside the worker's process, which inherits its whole env."""
    return non_dispatcher_owned_context()


@pytest.mark.parametrize(("enter", "reason"), [
    pytest.param(_without_the_card, ONLY_THE_WORKER, id="no-card"),
    pytest.param(_as_a_delegated_child, "delegate_task child agents are not Kanban run owners",
                 id="delegated-child"),
    pytest.param(_as_a_delegated_childs_process, ONLY_THE_WORKER, id="delegated-child-process"),
    pytest.param(_as_a_cron_run_in_the_worker, ONLY_THE_WORKER, id="cron-run-in-the-worker"),
])
def test_a_caller_that_is_not_the_dispatchers_worker_is_refused(worker, monkeypatch, enter, reason):
    drawing = _attach("layout.svg", PLAIN)
    call = {"task_id": worker.mine.id, "attachment_id": drawing}
    before = _state(worker)

    with monkeypatch.context() as patch, enter(patch):
        listed = _design_vision_schema_names()
        out = _dispatch(TOOL, call)

    assert TOOL not in listed
    assert "ok" not in out and reason in out["error"], out
    assert _state(worker) == before
    # Positive control: the worker itself sees the tool and the same call renders.
    assert TOOL in _design_vision_schema_names()
    assert _dispatch(TOOL, call).get("ok") is True


def test_a_drawing_of_another_card_on_the_same_board_is_refused(worker):
    mine = _attach("layout.svg", PLAIN)
    before = _state(worker)
    scoped = f"worker is scoped to task {worker.mine.id}"

    for call, reason in (
        ({"attachment_id": worker.drawings["sibling"]}, "is not on your own task"),
        ({"task_id": worker.sibling.id, "attachment_id": worker.drawings["sibling"]}, scoped),
        ({"task_id": worker.sibling.id, "attachment_id": mine}, scoped),
    ):
        out = _dispatch(TOOL, call)
        assert "ok" not in out and reason in out["error"], (call, out)

    assert _state(worker) == before
    # Positive control: the worker's own drawing renders.
    assert _dispatch(TOOL, {"attachment_id": mine}).get("ok") is True


@pytest.mark.parametrize(("card", "slug"), [
    pytest.param("foreign", OTHER, id="other-board"),
    pytest.param("default_card", kb.DEFAULT_BOARD, id="default-board"),
])
def test_a_drawing_on_another_board_is_refused(worker, card, slug):
    drawing = worker.drawings[card]
    # The drawing's id is still free on the worker's own board.
    assert _read(worker.stores[OWN], "SELECT id FROM task_attachments WHERE id = ?", drawing) == []
    before = _state(worker)
    scoped = f"worker is scoped to task {worker.mine.id}"

    for call, reason in (
        ({"attachment_id": drawing}, "is not on your own task"),
        ({"attachment_id": drawing, "board": slug}, "is not on your own task"),
        ({"task_id": getattr(worker, card).id, "attachment_id": drawing}, scoped),
    ):
        out = _dispatch(TOOL, call)
        assert "ok" not in out and reason in out["error"], (call, out)

    assert _state(worker) == before
    # Positive control: the worker's own drawing, given that same id on its
    # own board, renders on its own card.
    mine = _attach("layout.svg", PLAIN)
    assert mine == drawing
    out = _dispatch(TOOL, {"attachment_id": mine})
    assert out.get("ok") is True and out["task_id"] == worker.mine.id


_FIFO_WRITER = (
    "import os, pathlib, sys\n"
    "fd = os.open(sys.argv[1], os.O_WRONLY)\n"
    "pathlib.Path(sys.argv[2]).write_text('opened')\n"
    "os.close(fd)\n"
)


class _CountingHandler(BaseHTTPRequestHandler):
    """Answers every request with 404 and records its path on the server."""

    def _count(self):
        self.server.hits.append(self.path)
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_HEAD = do_POST = _count

    def log_message(self, format, *args):  # noqa: A002 (the base class's signature)
        pass


def test_external_references_are_refused_before_anything_is_rendered(worker, tmp_path):
    """Nothing a drawing names outside itself is ever opened or fetched.

    A FIFO stands in for a local picture: a writer process blocks opening it
    until someone opens it for reading, then leaves a marker before closing
    it, so whoever read it to the end finds the marker already there. A local
    web server counts every request. Both are proved live at the end: the
    renderer itself opens the FIFO when handed a drawing that names it, and a
    plain request reaches the server.
    """
    fifo = tmp_path / "picture.png"
    marker = tmp_path / "opened"
    os.mkfifo(fifo)
    writer = subprocess.Popen([sys.executable, "-c", _FIFO_WRITER, str(fifo), str(marker)])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    web = f"http://127.0.0.1:{port}"
    try:
        drawings = [
            (_svg(f'<image href="{fifo}" width="40" height="40"/>'), "holds an embedded image"),
            (_svg(f'<image xlink:href="{fifo}" width="40" height="40"/>'), "holds an embedded image"),
            (_fe_image(str(fifo)), "has an external reference"),
            (_fe_image(fifo.as_uri()), "has an external reference"),
            (_fe_image(f"{web}/photo.png"), "has an external reference"),
            (_svg(f'<use href="{web}/parts.svg#logo"/>'), "has an external reference"),
            (_svg(f"<style>@import url({web}/site.css);</style>{RECT}"), "has an external reference"),
            (_svg(f'<rect width="40" height="40" fill="url({web}/paint.svg#g)"/>'), "has an external reference"),
            (_svg(f'{RECT}<script>fetch("{web}/script")</script>'), "holds a script element"),
            (f"<svg {NS} width=\"40\" height=\"40\" onload=\"fetch('{web}/onload')\">{RECT}</svg>",
             "has an event attribute"),
        ]
        attached = [_attach(f"reference-{n}.svg", svg) for n, (svg, _) in enumerate(drawings)]
        before = _state(worker)

        for drawing, (_, reason) in zip(attached, drawings):
            out = _dispatch(TOOL, {"attachment_id": drawing})
            assert "ok" not in out and reason in out["error"], out

        assert not marker.exists()
        assert server.hits == []
        assert _state(worker) == before

        # Positive controls. The writer is still waiting, and the renderer
        # itself opens the FIFO when it is handed a drawing that names it.
        assert writer.poll() is None
        import resvg_py

        resvg_py.svg_to_bytes(svg_string=_fe_image(str(fifo)), width=40, height=40, dpi=96.0)
        assert marker.read_text() == "opened"
        assert writer.wait(timeout=10) == 0
        # The server answers a plain request and counts it.
        probe = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            probe.request("GET", "/probe")
            assert probe.getresponse().status == 404
        finally:
            probe.close()
        assert server.hits == ["/probe"]
    finally:
        server.shutdown()
        server.server_close()
        if writer.poll() is None:
            reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            try:
                writer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                writer.kill()
                writer.wait()
            finally:
                os.close(reader)


def test_the_tool_is_reached_only_through_the_design_vision_toolset():
    import toolsets
    from tools.registry import discover_builtin_tools, registry

    assert "tools.kanban_render_drawing" in discover_builtin_tools()
    assert registry.get_toolset_for_tool(TOOL) == "design_vision"
    assert TOOL in toolsets.resolve_toolset("design_vision", include_registry=False)
    assert TOOL not in toolsets._HERMES_CORE_TOOLS
    for name in toolsets.TOOLSETS:
        if name in ("design_vision", "all", "*"):
            continue
        assert TOOL not in toolsets.resolve_toolset(name), name
    # Its schema names no tool of another toolset.
    schema = registry.get_schema(TOOL)
    texts = [schema["description"]] + [
        prop.get("description", "") for prop in schema["parameters"]["properties"].values()
    ]
    others = {
        name for name in registry.get_all_tool_names()
        if registry.get_toolset_for_tool(name) != "design_vision"
    }
    assert OFFER in others
    named = {name for name in others for text in texts if re.search(rf"\b{re.escape(name)}\b", text)}
    assert named == set()
