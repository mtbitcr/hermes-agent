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
   the declared content type or the file name: UTF-8 XML whose root is an ``svg`` element in the
   SVG namespace.
3. The drawing is parsed with expat, which never loads an external entity or DTD. It is refused
   when it holds a script, an event attribute, a foreignObject, an image element or a ``data:``
   reference; when an href (an animated one too) or a CSS ``url()`` names anything but a
   ``#fragment`` of the drawing, or its CSS has an ``@import``; or when its DOCTYPE names an
   outside DTD or declares entities or attribute defaults, or an xml-stylesheet instruction
   names a style sheet. CSS is read both with and without its comments, its escapes undone, so
   neither can hide a reference.
4. The drawing's own size is read from its svg element as resvg reads it: its width and height,
   or its viewBox where they are missing. A drawing with no size, a size in em or ex units (they
   depend on the font) or a side above 4,096 px is refused before anything is rendered.
5. resvg renders it at that size, through ``resvg_py``. resvg runs no script, and once the
   checks above pass it opens nothing the drawing names; the only files it reads are the
   system's fonts. The PNG must be exactly that size and at most 4 MiB.
6. The PNG is stored under the worker's run as a new attachment on the same card, and its id is
   returned.

Every refusal is a plain tool error that says why, and stores nothing.
"""

from __future__ import annotations

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
_PX_PER_UNIT = {"px": 1.0, "in": 96.0, "cm": 96 / 2.54, "mm": 96 / 25.4, "pt": 4 / 3, "pc": 16.0}  # at 96 dpi
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_CSS_ESCAPE = re.compile(r"\\([0-9a-fA-F]{1,6})[ \t\r\n\f]?|\\(.)", re.S)
_CSS_URL = re.compile(r"url\(\s*['\"]?\s*([^'\")\s]*)", re.I)


class _Refused(Exception):
    """The drawing is not rendered; the message is the plain reason."""


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


def _inspect(svg: str) -> list:
    """Parse the drawing, refusing anything it may not hold, and return its svg element's attributes.

    The attributes are (name, value) pairs in the order of the start tag, each name as expat
    gives it with namespaces on: "namespace local-name", or the bare local name.
    """
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.ordered_attributes = True
    root = None
    styles: list = []

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
        if target.lower() == "xml-stylesheet":
            raise _Refused("has an external reference (an xml-stylesheet instruction)")

    def start(name, attributes):
        nonlocal root
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

    def characters(data):
        if styles:
            styles[-1].append(data)

    def end(name):
        if name.rsplit(" ", 1)[-1].lower() == "style":
            _check_css("".join(styles.pop()), "a <style> element")

    parser.StartDoctypeDeclHandler = doctype
    parser.EntityDeclHandler = entity
    parser.AttlistDeclHandler = attlist
    parser.ProcessingInstructionHandler = instruction
    parser.StartElementHandler = start
    parser.CharacterDataHandler = characters
    parser.EndElementHandler = end
    try:
        parser.Parse(svg, True)
    except expat.ExpatError:
        raise _Refused("is not an SVG drawing (it is not well-formed XML)") from None
    return root


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


def _length(value, side):
    """A width or height of the svg element in px as resvg reads it, or None when it reads none.

    ``side`` is the matching side of the viewBox, which a percentage is a share of.
    """
    found = _LENGTH.fullmatch(value) if value is not None else None
    if not found:
        return None
    number, unit = float(found.group(1)), found.group(2)
    if unit in ("em", "ex"):
        raise _Refused("has its size in em or ex units, which depend on the font (give its width and height in px)")
    if unit == "%":
        if side is None:
            return None
        pixels = side * number / 100
    else:
        pixels = number * _PX_PER_UNIT[unit or "px"]
    return pixels if math.isfinite(pixels) else None


def _px(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _own_size(attributes: list) -> tuple:
    """The drawing's own size in whole pixels, as resvg rounds it, or a refusal."""
    box = _view_box(_root_value(attributes, "viewBox"))
    box_width, box_height = box or (None, None)
    width = _length(_root_value(attributes, "width"), box_width)
    height = _length(_root_value(attributes, "height"), box_height)
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
            try:
                svg = kb.read_attachment_bytes(drawing, board=board).decode("utf-8-sig")
            except UnicodeDecodeError:
                raise _Refused("is not an SVG drawing (it is not UTF-8 text)") from None
            width, height = _own_size(_inspect(svg))
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
        "svg element, or its viewBox where those are missing. It is stored as a new attachment "
        "on the same task, and its attachment_id is returned. Only plain, self-contained "
        'drawings whose svg element has xmlns="http://www.w3.org/2000/svg" are rendered. '
        "Nothing is stored, and the reason is returned, when the drawing holds a script, an "
        "event attribute such as onclick, a foreignObject, an image element or a data: "
        "reference; when an href or url() points anywhere but a #fragment of the drawing, or it "
        "uses a CSS @import; when it declares entities, default attributes or an outside DTD; "
        "when it has no size, a size in em or ex units, or a side above 4,096 px; or when the "
        "PNG would be larger than 4 MiB."
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
