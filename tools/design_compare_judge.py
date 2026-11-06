"""Design comparison rules: the measurements of a candidate and of a prototype at one width in, findings out.

Pure: no browser, file or network access.
"""

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import NamedTuple

SCHEMA = "design_compare_bundle_v1"
WIDTHS = (1280, 375)

BOX_TOLERANCE = 2.0
FONT_SIZE_TOLERANCE = 0.5
LINE_HEIGHT_TOLERANCE = 1.0
COLOUR_DELTA_E = 2.0
ALPHA_TOLERANCE = 0.02
PIXEL_CHANNEL_DELTA = 24
PIXEL_REGION_MIN_SIDE = 8
PIXEL_REGION_MIN_PIXELS = 64

COLOUR_PROPERTIES = (
    "color", "background-color", "border-top-color", "border-right-color", "border-bottom-color",
    "border-left-color", "text-decoration-color", "fill", "stroke",
)
BUNDLE_FIELDS = ("schema", "page", "width", "elements")
ELEMENT_FIELDS = ("landmarks", "tag", "role", "label", "alt", "text", "parent", "fixed", "box", "style")
BOX_FIELDS = ("x", "y", "width", "height")
STYLE_FIELDS = ("font-family", "font-face", "font-size", "font-weight", "line-height") + COLOUR_PROPERTIES

_EPSILON = 1e-9
_ALLOWED_LINE = re.compile(r'^\s*Allowed text change: "(.*?)" becomes "(.*)"\s*$', re.MULTILINE)


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    id: str
    kind: str
    page: str
    width: int
    element: str
    property: str
    measured: str
    expected: str
    tolerance: str
    crop_candidate: str = ""
    crop_prototype: str = ""


class Comparison(NamedTuple):
    findings: tuple
    notes: tuple


def finding_id(page, width, kind, element, prop):
    return hashlib.sha256(f"v1|{page}|{width}|{kind}|{element}|{prop}".encode()).hexdigest()[:16]


def _norm(text):
    return " ".join(unicodedata.normalize("NFKC", text or "").split()).lower()


def _name(text):
    return _norm(text)[:60]


def parse_allowed_changes(card_text):
    return tuple((_name(a), _name(b)) for a, b in _ALLOWED_LINE.findall(card_text))


def _require(obj, fields, where):
    if not isinstance(obj, dict):
        raise BundleError(f"{where} is not an object")
    for field in fields:
        if field not in obj:
            raise BundleError(f"{where} is missing field {field}")


def _validate(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        raise BundleError(f"unknown bundle schema, expected {SCHEMA}")
    _require(bundle, BUNDLE_FIELDS, "bundle")
    if bundle["width"] not in WIDTHS:
        raise BundleError(f"bundle width {bundle['width']} is not one of {WIDTHS}")
    for i, element in enumerate(bundle["elements"]):
        _require(element, ELEMENT_FIELDS, f"element {i}")
        _require(element["box"], BOX_FIELDS, f"element {i} box")
        _require(element["style"], STYLE_FIELDS, f"element {i} style")


def _entries(bundle):
    seen = Counter()
    out = []
    for idx, el in enumerate(bundle["elements"]):
        parts = []
        for tag, heading in el["landmarks"]:
            heading = _norm(heading)[:40]
            parts.append(f"{tag}:{heading}" if heading else tag)
        chain, role = "/".join(parts), el["role"] or el["tag"]
        name = _name(el["label"]) or _name(el["alt"]) or _name(el["text"])
        seen[chain, role, name] += 1
        out.append({"idx": idx, "chain": chain, "role": role, "name": name, "el": el,
                    "key": f'{chain}/{role}:"{name}"#{seen[chain, role, name]}'})
    return out


def srgb_to_lab(rgb):
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (v / 255 for v in rgb[:3])]
    r, g, b = linear
    xyz = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
           0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
           0.0193339 * r + 0.1191920 * g + 0.9503041 * b)
    f = [t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
         for t in (xyz[0] / 0.95047, xyz[1], xyz[2] / 1.08883)]
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def ciede2000(lab1, lab2):
    (l1, a1, b1), (l2, a2, b2) = lab1, lab2
    c_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1, c2 = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1 = math.degrees(math.atan2(b1, a1p)) % 360 if c1 else 0.0
    h2 = math.degrees(math.atan2(b2, a2p)) % 360 if c2 else 0.0
    dh = 0.0 if not c1 * c2 else (h2 - h1) - 360 * round((h2 - h1) / 360)
    d_hue = 2 * math.sqrt(c1 * c2) * math.sin(math.radians(dh / 2))
    l_bar, c_bar = (l1 + l2) / 2, (c1 + c2) / 2
    if not c1 * c2:
        h_bar = h1 + h2
    elif abs(h1 - h2) <= 180:
        h_bar = (h1 + h2) / 2
    else:
        h_bar = (h1 + h2 + (360 if h1 + h2 < 360 else -360)) / 2
    rad = math.radians
    t = (1 - 0.17 * math.cos(rad(h_bar - 30)) + 0.24 * math.cos(rad(2 * h_bar))
         + 0.32 * math.cos(rad(3 * h_bar + 6)) - 0.20 * math.cos(rad(4 * h_bar - 63)))
    r_c = 2 * math.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7))
    r_t = -math.sin(rad(60 * math.exp(-(((h_bar - 275) / 25) ** 2)))) * r_c
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    x, y = (c2 - c1) / (1 + 0.045 * c_bar), d_hue / (1 + 0.015 * c_bar * t)
    return math.sqrt(((l2 - l1) / s_l) ** 2 + x ** 2 + y ** 2 + r_t * x * y)


def _within(a, b, tolerance):
    return abs(a - b) <= tolerance + _EPSILON


def _px(value):
    return value if isinstance(value, str) else f"{value:g} px"


def _colour(value):
    return "none" if value is None else "rgba({:g}, {:g}, {:g}, {:g})".format(*value)


def _first_family(value):
    return value.split(",")[0].strip().strip("\"'").lower()


def _style_differences(cand, proto):
    pm = f"plus or minus {FONT_SIZE_TOLERANCE:g} px"
    for prop, same in (("font-family", _first_family(cand["font-family"]) == _first_family(proto["font-family"])),
                       ("font-face", _norm(cand["font-face"]) == _norm(proto["font-face"]))):
        if not same:
            yield "type", prop, str(cand[prop]), str(proto[prop]), "exact"
    if not _within(cand["font-size"], proto["font-size"], FONT_SIZE_TOLERANCE):
        yield "type", "font-size", _px(cand["font-size"]), _px(proto["font-size"]), pm
    if cand["font-weight"] != proto["font-weight"]:
        yield "type", "font-weight", f"{cand['font-weight']:g}", f"{proto['font-weight']:g}", "exact"
    lc, lp = cand["line-height"], proto["line-height"]
    if ("normal" in (lc, lp) and lc != lp) or (
            "normal" not in (lc, lp) and not _within(lc, lp, LINE_HEIGHT_TOLERANCE)):
        yield "type", "line-height", _px(lc), _px(lp), f"plus or minus {LINE_HEIGHT_TOLERANCE:g} px"
    for prop in COLOUR_PROPERTIES:
        c, p = cand[prop], proto[prop]
        if c is None and p is None:
            continue
        if c is None or p is None or ciede2000(srgb_to_lab(c), srgb_to_lab(p)) > COLOUR_DELTA_E + _EPSILON \
                or not _within(c[3], p[3], ALPHA_TOLERANCE):
            yield "colour", prop, _colour(c), _colour(p), f"delta E {COLOUR_DELTA_E}, alpha {ALPHA_TOLERANCE}"


def _shift(entry, allowed_pairs):
    el = entry["el"]
    if el["fixed"]:
        return 0, 0
    box, dx, dy = el["box"], 0, 0
    for a, c in allowed_pairs:
        if a is entry:
            continue
        ab, cb = a["el"]["box"], c["el"]["box"]
        bottom = ab["y"] + ab["height"]
        if (el["parent"] == a["el"]["parent"] and box["y"] < bottom and box["y"] + box["height"] > ab["y"]
                and box["x"] >= ab["x"] + ab["width"]):
            dx += cb["width"] - ab["width"]
        if box["y"] >= bottom:
            dy += cb["height"] - ab["height"]
    return dx, dy


def compare_bundles(candidate, prototype, allowed_changes=()):
    for bundle in (candidate, prototype):
        _validate(bundle)
    if candidate["width"] != prototype["width"]:
        raise BundleError("bundle pair has different widths")
    if candidate["page"] != prototype["page"]:
        raise BundleError("bundle pair has different pages")
    page, width = prototype["page"], prototype["width"]
    protos, cands = _entries(prototype), _entries(candidate)

    by_key = {c["key"]: c for c in cands}
    matched, left_protos = [], []
    for p in protos:
        c = by_key.pop(p["key"], None)
        if c:
            matched.append((p, c, False))
        else:
            left_protos.append(p)
    groups = defaultdict(lambda: ([], []))
    for p in left_protos:
        groups[p["chain"], p["role"]][0].append(p)
    for c in by_key.values():
        groups[c["chain"], c["role"]][1].append(c)
    pairs, missing, extra = [], [], []
    for ps, cs in groups.values():
        pairs += zip(ps, cs)
        missing += ps[len(cs):]
        extra += cs[len(ps):]

    allowed = list(dict.fromkeys((_name(a), _name(b)) for a, b in allowed_changes))
    allowed_pairs = [(p, c) for p, c in pairs if (p["name"], c["name"]) in allowed]
    allowed_ids = {id(p) for p, _ in allowed_pairs}
    found = {(p["name"], c["name"]) for p, c in allowed_pairs}
    notes = tuple(f'allowed text change not found on page {page} at {width} px: "{a}" becomes "{b}"'
                  for a, b in allowed if (a, b) not in found)

    findings = []

    def add(kind, element, prop, measured, expected, tolerance):
        findings.append(Finding(finding_id(page, width, kind, element, prop), kind, page, width, element,
                                prop, measured, expected, tolerance))

    compared = sorted(matched + [(p, c, True) for p, c in allowed_pairs], key=lambda t: t[0]["idx"])
    for p, c, is_allowed in compared:
        dx, dy = _shift(p, allowed_pairs)
        pb, cb = p["el"]["box"], c["el"]["box"]
        expected = {"x": pb["x"] + dx, "y": pb["y"] + dy, "width": pb["width"], "height": pb["height"]}
        for field in BOX_FIELDS[:2] if is_allowed else BOX_FIELDS:
            if not _within(cb[field], expected[field], BOX_TOLERANCE):
                add("layout", p["key"], f"box.{field}", _px(cb[field]), _px(expected[field]),
                    f"plus or minus {BOX_TOLERANCE:g} px")
        for diff in _style_differences(c["el"]["style"], p["el"]["style"]):
            add(diff[0], p["key"], *diff[1:])
    for p, c in sorted(pairs, key=lambda t: t[0]["idx"]):
        if id(p) not in allowed_ids:
            add("element", p["key"], "text", c["name"], p["name"], "exact")
    for p in sorted(missing, key=lambda e: e["idx"]):
        add("element", p["key"], "presence", "absent", "present", "exact")
    for c in sorted(extra, key=lambda e: e["idx"]):
        add("element", c["key"], "presence", "present", "absent", "exact")
    return Comparison(tuple(findings), notes)
