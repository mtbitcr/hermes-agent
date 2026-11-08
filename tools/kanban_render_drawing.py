"""SVG drawings on the worker's own card rendered as PNG pictures: ``kanban_render_drawing``.

The designer draws its layouts as SVG, because its runs have no build machine, terminal or file
tools. The owner app never shows SVG, and ``kanban_offer_design_options`` takes only PNG, JPEG or
WebP, so this tool turns one drawing into a picture the offer accepts. One call names one
``attachment_id``. In order:

1. The gate is ``kanban_offer_design_options``'s own: its ``check_fn`` is imported, and the
   handler repeats the offer handler's checks, with the same helpers, in the same order. Only a
   dispatcher-owned worker sees or runs the tool, and only on the card the dispatcher gave it
   (``HERMES_KANBAN_TASK``), on that card's own board.
2. The attachment must be on that card. What it holds is judged from the stored bytes, never from
   the declared content type or the file name: XML whose root is an ``svg`` element in the SVG
   namespace. Its text is read once, in its own encoding, and that text is what is checked and
   rendered. A byte order mark gives the encoding, or else a first "<" written in two bytes,
   which is UTF-16 in that byte order, and an encoding its XML declaration names must be that
   one. Any other drawing is in the encoding its XML declaration names, or else in UTF-8; any
   other encoding must have one byte per character, such as ISO-8859-1, and is read byte by
   byte, as expat reads it.
3. The drawing is parsed with expat, which never loads an external entity or DTD. It is refused
   when it holds a script, an event attribute, a foreignObject, an image element or a ``data:``
   reference; when an href (an animated one too) or a CSS ``url()`` names anything but a
   ``#fragment`` of the drawing, or its CSS has an ``@import``; or when its DOCTYPE names an
   outside DTD or declares entities or attribute defaults, or an xml-stylesheet instruction
   names a style sheet. CSS is read both with and without its comments, its escapes undone, so
   neither can hide a reference.
4. The drawing's own size is read from its svg element as resvg reads it: its width and height,
   or its viewBox where they are missing. A size in em or ex units is measured by the svg
   element's own font size, found as resvg finds it and read with the rules of resvg's own CSS
   parsing: its font-size attribute, then the declarations of the drawing's <style> rules whose
   selectors pick the svg element, from the least to the most specific, then those of its style
   attribute, the font shorthand included; a later declaration replaces an earlier one unless
   that one is !important. That font size must be a number or an absolute length; resvg takes
   1ex as half of 1em. A drawing with no size, a size in em or ex units whose font size is
   missing, relative, a keyword or unreadable, or a side above 4,096 px is refused before
   anything is rendered.
5. resvg renders it at that size, through ``resvg_py``. resvg runs no script, and once the
   checks above pass it opens nothing the drawing names; the only files it reads are the
   system's fonts. The PNG must be exactly that size and at most 4 MiB.
6. The PNG is stored under the worker's run as a new attachment on the same card, and its id is
   returned.

Every refusal is a plain tool error that says why, and stores nothing.
"""

from __future__ import annotations

import codecs
import functools
import logging
import math
import os
import re
import struct
from pathlib import Path
from xml.parsers import expat

from tools.kanban_design_options import _check_design_options_mode
from tools.kanban_tools import (
    _board_of_connection,
    _connect,
    _enforce_worker_task_ownership,
    _is_dispatcher_owned_worker,
    _ok,
    _reject_delegated_child_mutation,
    _worker_run_id,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "kanban_render_drawing"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
XML_NS = "http://www.w3.org/XML/1998/namespace"
MAX_SIDE = 4096
MAX_PNG_BYTES = 4 * 1024 * 1024  # the largest picture a design option may show
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# resvg reads a size attribute of the svg element by its local name in any of these
# namespaces, and the first one in the start tag wins.
_SIZE_NAMESPACES = frozenset({"", SVG_NS, XLINK_NS, XML_NS})
_FRAGMENT = re.compile(r"#\S+")
_NUMBER = r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_LENGTH = re.compile(rf"[ \t\r\n]*({_NUMBER})(px|in|cm|mm|pt|pc|em|ex|%)?")
_LIST_NUMBER = re.compile(rf"[ \t\r\n]*({_NUMBER})[ \t\r\n]*,?")
_FONT_SIZE = re.compile(rf"[ \t\r\n]*({_NUMBER})(px|in|cm|mm|pt|pc)?")
_PX_PER_UNIT = {"px": 1.0, "in": 96.0, "cm": 96 / 2.54, "mm": 96 / 25.4, "pt": 4 / 3, "pc": 16.0}  # at 96 dpi
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_CSS_ESCAPE = re.compile(r"\\([0-9a-fA-F]{1,6})[ \t\r\n\f]?|\\(.)", re.S)
_CSS_URL = re.compile(r"url\(\s*['\"]?\s*([^'\")\s]*)", re.I)
# The svg element's font size is read from CSS as resvg reads it, with the rules of its CSS
# parser (simplecss) and of its parser for the font shorthand and lengths (svgtypes).
_CSS_SPACE = " \t\r\n\f"
_CSS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_PSEUDO_CLASSES = frozenset({"first-child", "link", "visited", "hover", "active", "focus"})
_SVG_SPACE = re.compile(r"[ \t\r\n]*")
_SVG_WORD = re.compile(r"[0-9A-Za-z_-]*")
_SVG_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]*)(?:[eE](?![mx])[+-]?[0-9]*)?")
_SVG_UNIT = re.compile(r"%|em|ex|px|in|cm|mm|pt|pc")
_FONT_KEYWORDS = frozenset(
    "normal small-caps italic oblique bold bolder lighter 100 200 300 400 500 600 700 800 900 "
    "ultra-condensed extra-condensed condensed semi-condensed semi-expanded expanded extra-expanded "
    "ultra-expanded".split()
)
_FONT_SIZE_KEYWORDS = frozenset("xx-small x-small small medium large x-large xx-large larger smaller".split())
_NO_FONT_SIZE = (
    "has its size in em or ex units but no plain font size to measure them by "
    "(give its svg element a font-size in px, or its width and height in px)"
)
# A drawing that starts with a byte order mark, or with its first "<" written in two bytes, is in
# the encoding that start gives: (start, Python's codec, the encoding's name, Python's names of the
# encodings its XML declaration may name, "utf-16" naming either byte order).
_UNICODE_STARTS = (
    (codecs.BOM_UTF8, "utf-8-sig", "UTF-8", frozenset({"utf-8"})),
    (codecs.BOM_UTF16_BE, "utf-16", "UTF-16BE", frozenset({"utf-16", "utf-16-be"})),
    (codecs.BOM_UTF16_LE, "utf-16", "UTF-16LE", frozenset({"utf-16", "utf-16-le"})),
    (b"\0<", "utf-16-be", "UTF-16BE", frozenset({"utf-16", "utf-16-be"})),
    (b"<\0", "utf-16-le", "UTF-16LE", frozenset({"utf-16", "utf-16-le"})),
)
_XML_ENCODING = re.compile(  # the encoding an XML declaration names, as XML 1.0 writes it
    r"<\?xml[ \t\r\n]+version[ \t\r\n]*=[ \t\r\n]*(?:\"[^\"]*\"|'[^']*')[ \t\r\n]+"
    r"encoding[ \t\r\n]*=[ \t\r\n]*([\"'])([A-Za-z][A-Za-z0-9._-]*)\1"
)


class _Refused(Exception):
    """The drawing is not rendered; the message is the plain reason."""


def _codec_name(encoding: str):
    """Python's own name for ``encoding``, or None when Python knows no such encoding. Python
    remembers every name it looks up and does not know, so a name longer than any encoding's is
    not looked up."""
    if len(encoding) > 64:
        return None
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None


def _one_byte_table(codec, name: str) -> str:
    """Each byte's character in Python's ``codec`` for the encoding ``name``, read from the 256
    bytes in turn as expat reads an encoding that is not its own, U+FFFE marking a byte that is no
    character; a refusal when the encoding does not have one byte per character."""
    try:
        table = bytes(range(256)).decode(codec, "replace") if codec else ""
    except (LookupError, UnicodeError):  # not an encoding of text, or one that cannot be read so
        table = ""
    if len(table) != 256:
        raise _Refused(
            f"is not an SVG drawing (it cannot be read in {name}, the encoding its XML declaration "
            "names; write it in UTF-8)"
        )
    return table.replace("\ufffd", "\ufffe")


def _decode(data: bytes) -> str:
    """The drawing's text, read once in its own encoding, or a refusal.

    A byte order mark gives the encoding, or else a first "<" written in two bytes, which is
    UTF-16 in that byte order; an encoding its XML declaration names must be that one. Any other
    drawing is in the encoding its XML declaration names, or else in UTF-8, and any other encoding
    is read byte by byte, as expat reads one.
    """
    for start, codec, name, declarable in _UNICODE_STARTS:
        if data.startswith(start):
            try:
                text = data.decode(codec)
            except UnicodeDecodeError:
                raise _Refused(f"is not an SVG drawing (it is not {name} text)") from None
            declared = _XML_ENCODING.match(text)
            if declared and _codec_name(declared[2]) not in declarable:
                raise _Refused(
                    f"is not an SVG drawing (it is in {name}, but its XML declaration names {declared[2]})"
                )
            return text
    declared = _XML_ENCODING.match(data.decode("latin-1"))  # a declaration is written in ASCII
    name = declared[2] if declared else "UTF-8"
    codec = _codec_name(name)
    try:
        if codec == "utf-8":
            return data.decode("utf-8")
        return codecs.charmap_decode(data, "strict", _one_byte_table(codec, name))[0]
    except UnicodeDecodeError:
        raise _Refused(f"is not an SVG drawing (it is not {name} text)") from None


def _check_href(value: str, where: str) -> None:
    """Refuse a reference that is a data: URL or names anything but a #fragment of the drawing."""
    if value.strip().lower().startswith("data:"):
        raise _Refused(f"holds an embedded image (a data: reference in {where})")
    if not _FRAGMENT.fullmatch(value):
        raise _Refused(f"has an external reference ({where} is not a #fragment of the drawing)")


def _unescape_css(css: str) -> str:
    def _char(found: re.Match) -> str:
        if found.group(1) is None:
            return found.group(2)
        code = int(found.group(1), 16)
        return chr(code) if 0 < code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF else "\ufffd"

    return _CSS_ESCAPE.sub(_char, css)


def _check_css(css: str, where: str) -> None:
    """Refuse CSS that imports a style sheet or whose url() names anything but a #fragment."""
    for view in (_CSS_COMMENT.sub("", css), css):
        view = _unescape_css(view)
        if "@import" in view.lower():
            raise _Refused(f"has an external reference (a CSS @import in {where})")
        for target in _CSS_URL.findall(view):
            if target.lower().startswith("data:"):
                raise _Refused(f"holds an embedded image (a data: url() in {where})")
            if not _FRAGMENT.fullmatch(target):
                raise _Refused(f"has an external reference (a url() in {where} is not a #fragment of the drawing)")


def _inspect(svg: str) -> tuple:
    """Parse the drawing, refusing anything it may not hold, and return its svg element's
    attributes and the style sheets resvg reads from its <style> elements.

    The attributes are (name, value) pairs in the order of the start tag, each name as expat
    gives it with namespaces on: "namespace local-name", or the bare local name. resvg reads a
    style sheet from each element named exactly style, in any namespace, whose first type
    attribute is missing or "text/css": the text of its first child, when that child is text
    (CDATA and character references included).
    """
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.ordered_attributes = True
    root = None
    styles: list = []  # all the text inside each open <style> element, which is checked
    sheets: list = []  # the text of each style sheet resvg reads, in document order
    reading = None  # the sheet whose first child is being read

    def doctype(name, system_id, public_id, has_internal_subset):
        if system_id or public_id:
            raise _Refused("has an external reference (the DTD its DOCTYPE names)")

    def entity(name, is_parameter_entity, value, base, system_id, public_id, notation_name):
        if value is None:
            raise _Refused(f"has an external reference (the external entity {name})")
        raise _Refused(f"declares the entity {name} in its DOCTYPE (write its text out in the drawing instead)")

    def attlist(element_name, attribute_name, attribute_type, default, required):
        raise _Refused(
            f"declares default attributes for <{element_name}> in its DOCTYPE (write them on the elements instead)"
        )

    def instruction(target, data):
        nonlocal reading
        reading = None
        if target.lower() == "xml-stylesheet":
            raise _Refused("has an external reference (an xml-stylesheet instruction)")

    def comment(data):
        nonlocal reading
        reading = None

    def start(name, attributes):
        nonlocal root, reading
        reading = None
        pairs = list(zip(attributes[::2], attributes[1::2]))
        if root is None:
            if name != f"{SVG_NS} svg":
                raise _Refused(
                    f'is not an SVG drawing (its root must be an svg element with xmlns="{SVG_NS}")'
                )
            root = pairs
        local = name.rsplit(" ", 1)[-1]
        element = local.lower()
        if element in ("script", "handler"):
            raise _Refused("holds a script element")
        if element == "foreignobject":
            raise _Refused("holds a foreignObject element")
        if element == "image":
            raise _Refused("holds an embedded image (an image element)")
        animates_href = any(
            key.rsplit(" ", 1)[-1].lower() == "attributename" and value.strip().lower().endswith("href")
            for key, value in pairs
        )
        for key, value in pairs:
            attr = key.rsplit(" ", 1)[-1]
            lowered = attr.lower()
            if lowered.startswith("on"):
                raise _Refused(f"has an event attribute ({attr} on <{local}>)")
            if lowered == "href":
                _check_href(value, f"the href of <{local}>")
            elif animates_href and lowered in ("from", "to", "by", "values"):
                for item in value.split(";"):
                    if item.strip():
                        _check_href(item.strip(), f"an href that <{local}> animates")
            _check_css(value, f"the {attr} attribute of <{local}>")
        if element == "style":
            styles.append([])
        style_type = next((value for key, value in pairs if key.rsplit(" ", 1)[-1] == "type"), "text/css")
        if local == "style" and style_type == "text/css":
            reading = []
            sheets.append(reading)

    def characters(data):
        if styles:
            styles[-1].append(data)
        if reading is not None:
            reading.append(data)

    def end(name):
        nonlocal reading
        reading = None
        if name.rsplit(" ", 1)[-1].lower() == "style":
            _check_css("".join(styles.pop()), "a <style> element")

    parser.StartDoctypeDeclHandler = doctype
    parser.EntityDeclHandler = entity
    parser.AttlistDeclHandler = attlist
    parser.ProcessingInstructionHandler = instruction
    parser.CommentHandler = comment
    parser.StartElementHandler = start
    parser.CharacterDataHandler = characters
    parser.EndElementHandler = end
    try:
        parser.Parse(svg, True)
    except expat.ExpatError:
        raise _Refused("is not an SVG drawing (it is not well-formed XML)") from None
    return root, ["".join(sheet) for sheet in sheets]


def _root_value(attributes: list, name: str):
    """The value resvg reads for the svg element's attribute ``name``, or None."""
    for key, value in attributes:
        namespace, _, local = key.rpartition(" ")
        if local == name and namespace in _SIZE_NAMESPACES:
            return value
    return None


def _view_box(value):
    """The viewBox's width and height as resvg reads them, or None when it reads no viewBox."""
    if value is None:
        return None
    numbers, position = [], 0
    for _ in range(4):  # resvg reads the first four numbers and ignores the rest
        found = _LIST_NUMBER.match(value, position)
        if not found:
            return None
        numbers.append(float(found.group(1)))
        position = found.end()
    width, height = numbers[2], numbers[3]
    if not all(math.isfinite(number) for number in numbers) or width <= 0 or height <= 0:
        return None
    return width, height


class _Css:
    """CSS text read from ``pos`` on, as simplecss reads it; a reading error raises ValueError."""

    def __init__(self, text: str):
        self.text, self.pos = text, 0

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def peek(self, ahead: int = 0) -> str:
        """The character ``ahead`` of the current one, or "" past the end."""
        return self.text[self.pos + ahead : self.pos + ahead + 1]

    def need(self) -> str:
        """The current character; the end is an error."""
        if self.at_end():
            raise ValueError("unexpected end of CSS")
        return self.text[self.pos]

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise ValueError(f"expected {char!r}")
        self.pos += 1

    def skip_over(self, chars: str) -> None:
        while not self.at_end() and self.text[self.pos] in chars:
            self.pos += 1

    def skip_to(self, chars: str) -> None:
        while not self.at_end() and self.text[self.pos] not in chars:
            self.pos += 1

    def skip_spaces(self) -> None:
        self.skip_over(_CSS_SPACE)

    def skip_comment(self) -> None:
        """Skip a comment; an unclosed one is skipped to the end, and is an error."""
        self.expect("/")
        self.expect("*")
        close = self.text.find("*/", self.pos)
        self.pos = len(self.text) if close < 0 else close + 2
        if close < 0:
            raise ValueError("unclosed CSS comment")

    def skip_spaces_and_comments(self) -> None:
        self.skip_spaces()
        while self.text.startswith("/*", self.pos):
            self.skip_comment()
            self.skip_spaces()

    def ident(self) -> str:
        """An identifier (ASCII only); a leading "-" stays read when no identifier follows it."""
        start = self.pos
        if self.peek() == "-":
            self.pos += 1
        name = _CSS_NAME.match(self.text, self.pos)
        if not name:
            raise ValueError("expected a CSS identifier")
        self.pos = name.end()
        return self.text[start : self.pos]

    def string(self) -> str:
        """A quoted string's text, where a quote after a backslash does not end it, or an identifier."""
        quote = self.need()
        if quote not in "'\"":
            return self.ident()
        self.pos += 1
        start = self.pos
        while not self.at_end() and not (self.text[self.pos] == quote and self.text[self.pos - 1] != "\\"):
            self.pos += 1
        value = self.text[start : self.pos]
        self.expect(quote)
        return value

    def block_end(self) -> None:
        """Skip past the closing brace of the block being read, over any nested blocks."""
        depth = 0
        while not self.at_end():
            char = self.text[self.pos]
            if char == "}":
                if not depth:
                    break
                depth -= 1
            elif char == "{":
                depth += 1
            self.pos += 1
        if self.peek() == "}":
            self.pos += 1


def _css_term(css: _Css) -> bool:
    """Read one term of a declaration's value; False when none can be read."""
    try:
        char = css.need()
        if char == "#":
            css.pos += 1
            try:
                css.ident()
            except ValueError:
                css.skip_over("0123456789abcdefABCDEF")
        elif char in "+-.0123456789":
            css.pos += 1
            css.skip_over("0123456789")
            if css.peek() == ".":
                css.pos += 1
                css.skip_over("0123456789")
            if css.peek() == "%":
                css.pos += 1
            else:
                try:
                    css.ident()
                except ValueError:
                    pass
        elif char in "'\"":
            css.string()
        elif char == ",":
            css.pos += 1
        else:
            css.ident()
            if css.peek() == "(":
                css.skip_to(")")
                css.expect(")")
    except ValueError:
        return False
    return True


def _css_declaration(css: _Css) -> tuple:
    """Read one declaration: (name, value, important)."""
    css.skip_spaces_and_comments()
    if css.peek() == "*":
        css.pos += 1
    name = css.ident()
    css.skip_spaces_and_comments()
    css.expect(":")
    css.skip_spaces_and_comments()
    start = end = css.pos
    while _css_term(css):
        end = css.pos
        css.skip_spaces_and_comments()
    css.skip_spaces_and_comments()
    important = False
    if css.peek() == "!":
        css.pos += 1
        css.skip_spaces_and_comments()
        if css.text.startswith("important", css.pos):
            css.pos += len("important")
            important = True
    css.skip_spaces_and_comments()
    while css.peek() == ";":
        css.pos += 1
        css.skip_spaces_and_comments()
    if start == end:
        raise ValueError("empty CSS value")
    return name, css.text[start:end], important


def _css_declarations(css: _Css) -> list:
    """Read the declarations of a block, up to its closing brace; an error ends the block."""
    declarations = []
    while not css.at_end() and css.peek() != "}":
        try:
            declarations.append(_css_declaration(css))
        except ValueError:
            css.block_end()
            break
    return declarations


def _css_selector(css: _Css):
    """Read one selector of a selector list, up to a comma or brace, as a list of compound
    selectors, each [type, sub-selectors] with the type None for *, or None when simplecss
    drops it.

    A sub-selector is (attribute, operator, value), the operator one of "]" (it exists), "=",
    "~" (one of its words) and "|" (it or a prefix ending in -), or (None, ":", pseudo-class).
    """
    compounds, after_combinator, combined = [], True, False

    def sub(selector):
        nonlocal combined
        if compounds and not combined:
            compounds[-1][1].append(selector)
        else:
            compounds.append([None, [selector]])
            combined = False

    try:
        while True:
            char = css.peek()
            if char in ("", ",", "{") or (char == "/" and css.peek(1) != "*"):
                return None if after_combinator else compounds
            if char == "/":
                css.skip_comment()
            elif char in _CSS_SPACE:
                css.skip_spaces()
                if not after_combinator:
                    while css.peek() == "/":
                        css.skip_comment()
                        css.skip_spaces()
                    if css.peek() not in ("", ">", "+", ",", "{"):
                        after_combinator = combined = True  # a descendant combinator
            elif char in ">+":
                if after_combinator:
                    return None
                css.pos += 1
                after_combinator = combined = True
            elif char == "*":
                if not after_combinator:
                    return None
                css.pos += 1
                compounds.append([None, []])
                after_combinator = combined = False
            elif char in "#.[:":
                css.pos += 1
                after_combinator = False
                name = css.ident()
                if char == "#":
                    sub(("id", "=", name))
                elif char == ".":
                    sub(("class", "~", name))
                elif char == "[":
                    operator, value = css.need(), None
                    if operator in "=~|":
                        css.pos += 1
                        if operator != "=":
                            css.expect("=")
                        value = css.string()
                    elif operator != "]":
                        return None
                    css.expect("]")
                    sub((name, operator, value))
                elif name == "lang":
                    css.expect("(")
                    start = css.pos
                    css.skip_to(")")
                    language = css.text[start : css.pos].strip()
                    css.expect(")")
                    if not language:
                        return None
                    sub((None, ":", name))
                elif name in _PSEUDO_CLASSES:
                    sub((None, ":", name))
                else:
                    return None
            else:
                name = css.ident()
                if not after_combinator:
                    return None
                compounds.append([name, []])
                after_combinator = combined = False
    except ValueError:
        return None


def _css_rules(sheet: str) -> list:
    """The rules simplecss reads from a style sheet, as (selector, declarations) pairs in
    source order; rules without declarations are dropped."""
    css, rules = _Css(sheet), []
    try:
        css.skip_spaces_and_comments()
        while not css.at_end():
            css.skip_spaces_and_comments()
            try:
                if css.peek() == "@":  # an at-rule, which is skipped
                    css.pos += 1
                    css.ident()
                    css.skip_to(";{")
                    opens_a_block = css.need() == "{"
                    css.pos += 1
                    if opens_a_block:
                        css.block_end()
                    continue
                first = len(rules)
                while css.need() == "," or len(rules) == first:
                    if css.peek() == ",":
                        css.pos += 1
                    selector = _css_selector(css)
                    css.skip_spaces()
                    if selector:
                        rules.append([selector, []])
                    if css.need() != ",":
                        css.skip_to("{")
                        break
                if css.peek() == "{":
                    css.pos += 1
                declarations = _css_declarations(css)
                for rule in rules[first:]:
                    rule[1] = declarations
                if css.peek() == "}":
                    css.pos += 1
            except ValueError:
                pass
    except ValueError:
        pass
    return [(selector, declarations) for selector, declarations in rules if declarations]


def _picks_the_svg_element(selector: list, attributes: list) -> bool:
    """Whether a selector picks the svg element, which has no parent and no element before it."""
    if len(selector) != 1 or selector[0][0] not in (None, "svg"):
        return False
    for name, operator, value in selector[0][1]:
        if operator == ":":
            if value != "first-child":
                return False
            continue
        actual = next((text for key, text in attributes if key.rpartition(" ")[2] == name), None)
        if actual is None or not (
            operator == "]"
            or (operator == "=" and actual == value)
            or (operator == "~" and value in actual.split(" "))
            or (operator == "|" and (actual == value or actual.startswith(f"{value}-")))
        ):
            return False
    return True


def _specificity(selector: list) -> tuple:
    """A selector's specificity as simplecss counts it: ids, other sub-selectors, types."""
    ids = sum(sub[0] == "id" for _, subs in selector for sub in subs)
    others = sum(len(subs) for _, subs in selector) - ids
    types = sum(name is not None for name, _ in selector)
    return min(ids, 255), min(others, 255), min(types, 255)


def _svg_length_end(text: str, position: int):
    """Where the length that svgtypes reads from ``position`` ends, or None when it reads none."""
    number = _SVG_NUMBER.match(text, _SVG_SPACE.match(text, position).end())
    try:
        if not number or not math.isfinite(float(number.group())):
            return None
    except ValueError:
        return None
    return (_SVG_UNIT.match(text, number.end()) or number).end()


def _font_shorthand_size(value: str):
    """The font size that a font shorthand sets, as written, or None when resvg ignores it.

    Up to four style, variant, weight or stretch keywords come first, then the size: a length
    that starts with a digit, or a size keyword. An optional /line-height and the family follow.
    """
    position = _SVG_SPACE.match(value).end()
    for _ in range(4):
        word = _SVG_WORD.match(value, position)
        if word.group() not in _FONT_KEYWORDS:
            break
        position = _SVG_SPACE.match(value, word.end()).end()
    start = position
    if start >= len(value):
        return None
    if value[start] in "0123456789":
        end = _svg_length_end(value, start)
    else:
        end = _SVG_WORD.match(value, start).end()
        if value[start:end] not in _FONT_SIZE_KEYWORDS:
            return None
    if end is None:
        return None
    rest = _SVG_SPACE.match(value, end).end()
    if value[rest : rest + 1] == "/":
        rest = _svg_length_end(value, rest + 1)
        if rest is None:
            return None
        rest = _SVG_SPACE.match(value, rest).end()
    return value[start:end] if rest < len(value) else None


def _root_font_size(attributes: list, sheets: list):
    """The svg element's font size as resvg finds it, as written, or None when it has none.

    The svg element's font-size attribute comes first, then the declarations of the rules that
    pick the svg element, from the least to the most specific and in source order, then those
    of its first style attribute, in any namespace. font-size and the font shorthand set it; a
    declaration replaces it unless the one that set it is !important, and inherit makes it medium.
    """
    value, important = _root_value(attributes, "font-size"), False

    def declare(name, declared, flagged):
        nonlocal value, important
        if name == "font":
            declared = _font_shorthand_size(declared)
        elif name != "font-size":
            return
        if declared is not None and not important:
            value, important = ("medium", False) if declared == "inherit" else (declared, flagged)

    rules = [rule for sheet in sheets for rule in _css_rules(sheet) if _picks_the_svg_element(rule[0], attributes)]
    for _, declarations in sorted(rules, key=lambda rule: _specificity(rule[0])):
        for declaration in declarations:
            declare(*declaration)
    style = next((text for key, text in attributes if key.rpartition(" ")[2] == "style"), None)
    if style is not None:
        css = _Css(style)
        while True:
            try:
                css.skip_spaces_and_comments()
            except ValueError:
                pass
            if css.at_end():
                break
            try:
                declare(*_css_declaration(css))
            except ValueError:
                break
    return value


def _font_size(attributes: list, sheets: list) -> float:
    """The svg element's own font size in px, which its em and ex units are measured by, or a refusal.

    Only a number or an absolute length is used: none, a relative size, a keyword or a value that
    resvg cannot read is refused.
    """
    value = _root_font_size(attributes, sheets)
    found = _FONT_SIZE.fullmatch(value) if value is not None else None
    pixels = float(found.group(1)) * _PX_PER_UNIT[found.group(2) or "px"] if found else math.nan
    if not math.isfinite(pixels):
        raise _Refused(_NO_FONT_SIZE)
    return pixels


def _length(value, side, font_size):
    """A width or height of the svg element in px as resvg reads it, or None when it reads none.

    ``side`` is the matching side of the viewBox, which a percentage is a share of, and
    ``font_size`` gives the svg element's own font size, which em and ex are measured by.
    """
    found = _LENGTH.fullmatch(value) if value is not None else None
    if not found:
        return None
    number, unit = float(found.group(1)), found.group(2)
    if unit in ("em", "ex"):
        pixels = number * font_size() / (2 if unit == "ex" else 1)  # resvg takes 1ex as half of 1em
    elif unit == "%":
        if side is None:
            return None
        pixels = side * number / 100
    else:
        pixels = number * _PX_PER_UNIT[unit or "px"]
    return pixels if math.isfinite(pixels) else None


def _px(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _own_size(attributes: list, sheets: list) -> tuple:
    """The drawing's own size in whole pixels, as resvg rounds it, or a refusal."""
    box = _view_box(_root_value(attributes, "viewBox"))
    box_width, box_height = box or (None, None)
    font_size = functools.partial(_font_size, attributes, sheets)
    width = _length(_root_value(attributes, "width"), box_width, font_size)
    height = _length(_root_value(attributes, "height"), box_height, font_size)
    if box:
        if width is None and height is None:
            width, height = box
        elif width is None:
            width = height * box_width / box_height
        elif height is None:
            height = width * box_height / box_width
    if width is None or height is None or width <= 0 or height <= 0:
        raise _Refused("has no size (give its svg element a width and height, or a viewBox)")
    if width > MAX_SIDE or height > MAX_SIDE:
        raise _Refused(f"is {_px(width)}x{_px(height)} px (at most 4,096 px per side is rendered)")
    return max(1, int(width + 0.5)), max(1, int(height + 0.5))


def _render(svg: str, width: int, height: int) -> bytes:
    """The drawing as a PNG of exactly ``width`` x ``height`` px and at most 4 MiB, or a refusal."""
    try:
        import resvg_py
    except ImportError:
        raise _Refused("cannot be rendered here (the SVG renderer resvg-py is not installed)") from None
    try:
        png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height, dpi=96.0)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # a panic inside resvg arrives as pyo3's PanicException
        raise _Refused(f"could not be rendered ({e})") from None
    if (
        len(png) < 24
        or not png.startswith(PNG_SIGNATURE)
        or png[12:16] != b"IHDR"
        or struct.unpack(">II", png[16:24]) != (width, height)
    ):
        raise _Refused(f"could not be rendered at its own size of {width}x{height} px")
    if len(png) > MAX_PNG_BYTES:
        raise _Refused(f"would be a PNG of {len(png):,} bytes (at most 4 MiB is stored)")
    return png


def _handle_render_drawing(args: dict, **kw) -> str:
    """Render one SVG drawing of the worker's own card and store the PNG on the same card."""
    delegated_err = _reject_delegated_child_mutation(TOOL_NAME)
    if delegated_err:
        return delegated_err
    own_task = os.environ.get("HERMES_KANBAN_TASK")
    if not own_task or not _is_dispatcher_owned_worker():
        return tool_error(
            f"{TOOL_NAME} is only for the worker the dispatcher spawned on "
            "its own task"
        )
    tid = args.get("task_id") or own_task
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    attachment_id = args.get("attachment_id")
    if isinstance(attachment_id, bool) or not isinstance(attachment_id, int) or attachment_id < 1:
        return tool_error("attachment_id must be a positive whole number; nothing is stored")
    try:
        kb, conn = _connect()
        try:
            board = _board_of_connection(kb, conn) or kb.get_current_board()
            drawing = kb.get_attachment(conn, attachment_id)
            if drawing is None or drawing.task_id != tid:
                raise _Refused("is not on your own task")
            svg = _decode(kb.read_attachment_bytes(drawing, board=board))
            width, height = _own_size(*_inspect(svg))
            png = _render(svg, width, height)
            stored = kb.store_attachment_bytes(
                conn,
                tid,
                f"{Path(drawing.filename).stem}.png",
                png,
                content_type="image/png",
                uploaded_by=TOOL_NAME,
                board=board,
                max_bytes=MAX_PNG_BYTES,
                expected_run_id=_worker_run_id(tid),
            )
            return _ok(task_id=tid, attachment_id=stored, width=width, height=height, size=len(png))
        finally:
            conn.close()
    except _Refused as e:
        return tool_error(f"attachment {attachment_id} {e}; nothing is stored")
    except ValueError as e:
        return tool_error(f"{TOOL_NAME}: {e}")
    except Exception as e:
        logger.exception("%s failed", TOOL_NAME)
        return tool_error(f"{TOOL_NAME}: {e}")


KANBAN_RENDER_DRAWING_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Turn an SVG drawing attached to your own task into a PNG picture, for example to offer "
        "it as a design option. The PNG has the drawing's own size: the width and height of its "
        "svg element, or its viewBox where those are missing. Sizes in em or ex units are "
        "measured by the svg element's own font size, as the drawing's own CSS sets it: its "
        "font-size attribute, <style> rules that pick the svg element and its style attribute, "
        "the font shorthand and !important included. That font size must be a number or an "
        "absolute length (1ex is half of 1em). It is stored as a new attachment "
        "on the same task, and its attachment_id is returned. Only plain, self-contained "
        'drawings whose svg element has xmlns="http://www.w3.org/2000/svg" are rendered. '
        "Nothing is stored, and the reason is returned, when the drawing holds a script, an "
        "event attribute such as onclick, a foreignObject, an image element or a data: "
        "reference; when an href or url() points anywhere but a #fragment of the drawing, or it "
        "uses a CSS @import; when it declares entities, default attributes or an outside DTD; "
        "when it has no size, a size in em or ex units without such a font size, or a side "
        "above 4,096 px; or when the PNG would be larger than 4 MiB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "attachment_id": {
                "type": "integer",
                "minimum": 1,
                "description": "The id of the SVG attachment on your own task.",
            },
        },
        "required": ["attachment_id"],
    },
}


registry.register(
    name=TOOL_NAME,
    toolset="design_vision",
    schema=KANBAN_RENDER_DRAWING_SCHEMA,
    handler=_handle_render_drawing,
    check_fn=_check_design_options_mode,
    emoji="🖌️",
)
