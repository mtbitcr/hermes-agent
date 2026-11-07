"""design_compare: the reviewer compares a website card with the chosen design file pinned in its body.

prepare verifies the pin and stores the hashed capture kit on the card; judge reads the exported
bundles and pictures, runs the measurement check (design_compare_judge) and the pixel check
(design_compare_pixels), stores the crops and returns findings for the typed handback, at most four
undecided regions to look at, or verdict stop with one plain question for the owner; classify records
the reviewer's judgment of undecided regions. Every call passes check_design_access first, and only
the reviewer on a review run gets through.
"""

import base64
import hashlib
import json
import math
import os
import re
from contextlib import closing
from pathlib import Path
from typing import NamedTuple, Optional

from tools.design_access import REVIEWER, check_design_access
from tools.design_compare_judge import (
    PIXEL_CHANNEL_DELTA, PIXEL_REGION_MIN_PIXELS, PIXEL_REGION_MIN_SIDE, WIDTHS, Finding, _entries, _name,
    compare_bundles, parse_allowed_changes,
)
from tools.design_compare_pixels import (
    MAX_CROP_BYTES, _rgb, attach_regions, compare_element, compare_pages, cut_crops,
)
from tools.registry import registry, tool_error

KIT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "design_compare"
KIT_FILES = ("capture.mjs", "probe.js")
CAPTURE_KEYS = ("candidate_bundle", "candidate_image", "prototype_bundle", "prototype_image")
CATEGORIES = ("layout", "type", "colour", "element", "none")
MAX_PROTOTYPE_BYTES = 8 * 1024 * 1024
MAX_FINDINGS = 24
MAX_REGIONS = 4
UPLOADER = "design_compare"  # the kit, the crops and the classification records
RECORD = "design_compare_regions.json"

REFUSED = "design compare refused: "
NOT_REVIEWER = "only the reviewer may use it, in a review run"
NO_PIN = "this card names no chosen design file"
NO_PAGES = "this card names no pages to compare"
NO_CANDIDATE = "this review run names no reviewed candidate"
MISMATCH = ("The chosen design file for '{title}' is not the one this card was created with. "
            "Which design should Raphael compare against?")
STOP = ("The website change '{title}' still differs from the chosen design after {rounds} review rounds: "
        "{count} differences, with pictures on the card. Should Raphael keep the design exactly and try "
        "once more, or accept the change as it is now?")

_PIN = re.compile(r"Prototype pin: attachment=(\d+) sha256=([0-9a-f]{64}) file=\S+\.html")
_ID = re.compile(r"[0-9a-f]{16}")
_TAG = re.compile(r"\[dc:([0-9a-f]{16}|pixel)\]")  # an element finding's id, or a pixel finding's mark
_RECORD = re.compile(r"design_compare_regions( \(\d+\))?\.json")


class _Run(NamedTuple):
    task: str
    run: int
    board: Optional[str]


def _config() -> dict:
    from hermes_cli.config import read_raw_config_readonly

    section = (read_raw_config_readonly() or {}).get("design_browser")
    return section if isinstance(section, dict) else {}


def _kit_sha256() -> str:
    return hashlib.sha256(b"".join((KIT_DIR / name).read_bytes() for name in KIT_FILES)).hexdigest()


def _lines(body: str, prefix: str) -> list:
    return [line.rstrip() for line in body.splitlines() if line.startswith(prefix)]


def _pinned(kb, conn, ctx, task, action):
    """The pinned prototype verified (attachment, sha256, pages, candidate digest), or the result to return."""
    pins, pages = _lines(task.body or "", "Prototype pin:"), _lines(task.body or "", "Compare pages:")
    pin = _PIN.fullmatch(pins[0]) if len(pins) == 1 else None
    if pin is None:
        return tool_error(NO_PIN)
    routes = list(dict.fromkeys(pages[0][len("Compare pages:"):].split())) if len(pages) == 1 else []
    if not routes or not all(route.startswith("/") for route in routes):
        return tool_error(NO_PAGES)
    stop = json.dumps({"action": action, "verdict": "stop", "question": MISMATCH.format(title=task.title)})
    att = kb.get_attachment(conn, int(pin.group(1)))
    if (att is None or not att.filename.lower().endswith(".html") or att.size > MAX_PROTOTYPE_BYTES
            or att.task_id not in kb.parent_ids(conn, task.id)):  # never the build card itself
        return stop
    try:
        data = kb.read_attachment_bytes(att, board=ctx.board)
    except (OSError, ValueError):
        return stop
    if len(data) > MAX_PROTOTYPE_BYTES or hashlib.sha256(data).hexdigest() != pin.group(2):
        return stop
    digest = _candidate_digest(kb, conn, ctx.task)
    if digest is None:
        return tool_error(f"{action} refused: {NO_CANDIDATE}")
    return att, pin.group(2), routes, digest


def _candidate_digest(kb, conn, task_id) -> Optional[str]:
    from hermes_cli.kanban_review_record import record_candidate

    head = kb._latest_review_head_provenance(conn, task_id)
    record = None if head else record_candidate(conn, task_id)
    return head or (record.digest if record else None)


def _prepare(kb, conn, ctx, task, args):
    pinned = _pinned(kb, conn, ctx, task, "prepare")
    if not isinstance(pinned, tuple):
        return pinned
    att, sha256, pages, digest = pinned
    kit = []
    for name in KIT_FILES:
        data = (KIT_DIR / name).read_bytes()
        kit_id = kb.store_attachment_bytes(conn, ctx.task, name, data, content_type="text/javascript",
                                           uploaded_by=UPLOADER, board=ctx.board, expected_run_id=ctx.run)
        kit.append({"attachment_id": kit_id, "filename": name, "sha256": hashlib.sha256(data).hexdigest()})
    return json.dumps({
        "action": "prepare", "prototype": {"attachment_id": att.id, "sha256": sha256}, "kit": kit,
        "kit_sha256": _kit_sha256(), "pages": pages, "widths": list(WIDTHS), "run_id": str(ctx.run),
        "candidate_digest": digest,
        "next": ("Import the prototype and both kit files with their SHA-256 values, the kit files into one "
                 "folder under these file names; capture every page at both widths for the candidate and for "
                 "the prototype with this run id and candidate digest; export the bundles and pictures to this "
                 "card; then call judge with their attachment ids."),
    })


def _round(kb, conn, ctx):
    """This round's number and the [dc:] tags of the previous round's findings document. Only the documents
    the kernel delivered in a review handback on this card count, in delivery order, each bound to the
    candidate its delivery recorded; any other attachment, whatever its name, is no review round."""
    tagged = []
    for delivery in kb._review_findings_history(conn, ctx.task):
        att = kb.get_attachment(conn, delivery["attachment_id"])
        if att is None or att.task_id != ctx.task:
            continue
        try:
            doc = kb.parse_review_findings_document(json.loads(kb.read_attachment_bytes(att, board=ctx.board)))
        except (OSError, ValueError):
            continue
        if doc["candidate_digest"] != delivery["candidate_digest"]:
            continue
        ids = {fid for finding in doc["findings"] for fid in _TAG.findall(finding["problem"])}
        if ids:
            tagged.append(ids)
    return len(tagged) + 1, (tagged[-1] if tagged else set())


def _classified(kb, conn, ctx) -> dict:
    regions = {}
    for att in sorted(kb.list_attachments(conn, ctx.task), key=lambda a: a.id):
        if att.uploaded_by == UPLOADER and _RECORD.fullmatch(att.filename):
            record = json.loads(kb.read_attachment_bytes(att, board=ctx.board))
            if record.get("run_id") == str(ctx.run):
                regions.update(record["regions"])
    return regions


def _inside(region, box) -> bool:
    return (box["x"] <= region["x"] and box["y"] <= region["y"]
            and region["x"] + region["width"] <= box["x"] + box["width"]
            and region["y"] + region["height"] <= box["y"] + box["height"])


def _union(*boxes):
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    left, top = min(b["x"] for b in boxes), min(b["y"] for b in boxes)
    return {"x": left, "y": top, "width": max(b["x"] + b["width"] for b in boxes) - left,
            "height": max(b["y"] + b["height"] for b in boxes) - top}


def _text_pairs(cents, pents) -> dict:
    """Q2's text pairs as compare_bundles forms them, {prototype key: candidate key}: the left-over elements
    of each chain and role, paired in document order."""
    groups = {}
    for side, entries, other in ((0, pents, cents), (1, cents, pents)):
        keys = {e["key"] for e in other}
        for e in entries:
            if e["key"] not in keys:
                groups.setdefault((e["chain"], e["role"]), ([], []))[side].append(e["key"])
    return {p: c for ps, cs in groups.values() for p, c in zip(ps, cs)}


def _compare(capture, allowed):
    """One page at one width: its measurement findings, its element boxes by finding key and its undecided
    regions."""
    cand, proto, cimg, pimg = capture[:4]
    comparison = compare_bundles(cand, proto, allowed)
    cboxes = {e["key"]: e["el"]["box"] for e in _entries(cand)}
    pboxes = {e["key"]: e["el"]["box"] for e in _entries(proto)}
    pairs = _text_pairs(_entries(cand), _entries(proto))
    named = {**cboxes, **{p: cboxes[c] for p, c in pairs.items()}}  # findings name a pair by its prototype key
    old, new = {_name(a) for a, _ in allowed}, {_name(b) for _, b in allowed}
    pallowed = [e["el"]["box"] for e in _entries(proto) if e["key"] not in cboxes and e["name"] in old]
    callowed = [e["el"]["box"] for e in _entries(cand) if e["key"] not in pboxes and e["name"] in new]
    cmasks, pmasks = list(cand.get("masks") or []) + callowed, list(proto.get("masks") or []) + pallowed
    matched, boxed = cboxes.keys() & pboxes.keys(), {f.element for f in comparison.findings if f.kind == "layout"}
    regions = [r for key in sorted(matched - boxed)
               for r in compare_element(cimg, pimg, cboxes[key], pboxes[key], cmasks, pmasks)]
    shifted = [math.floor(min(cboxes[key]["y"], pboxes[key]["y"])) for key in matched
               if (cboxes[key]["x"], cboxes[key]["y"]) != (pboxes[key]["x"], pboxes[key]["y"])]
    if min(shifted, default=None) != 0:  # the page pass covers the rows above the first shifted element
        regions += compare_pages(cimg, pimg, cmasks + pmasks, min(shifted, default=None))
    unique = []
    for region in sorted(regions, key=lambda r: -r["width"] * r["height"]):
        if not any(_inside(region, kept) for kept in unique):
            unique.append(region)
    boxes = {key: _union(named.get(key), pboxes.get(key)) for key in cboxes.keys() | pboxes.keys()}
    _, undecided = attach_regions(sorted(unique, key=lambda r: (r["y"], r["x"])), comparison.findings, boxes.get)
    explained = [boxes[f.element] for f in comparison.findings if f.kind == "element"]  # missing, extra, text
    return comparison, named, pboxes, [r for r in undecided if not any(_inside(r, box) for box in explained)]


def _png_part(data: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def _judge(kb, conn, ctx, task, args):
    pinned = _pinned(kb, conn, ctx, task, "judge")
    if not isinstance(pinned, tuple):
        return pinned
    pages, digest = pinned[2:]
    captures = args.get("captures")
    if not isinstance(captures, list) or not captures or not all(isinstance(c, dict) for c in captures):
        return tool_error("judge needs captures: one entry per page and width with the attachment ids of "
                          + ", ".join(CAPTURE_KEYS))
    loaded = []
    for capture in captures:
        files = []
        for key in CAPTURE_KEYS:
            att = kb.get_attachment(conn, capture.get(key)) if type(capture.get(key)) is int else None
            if att is None or att.task_id != ctx.task:
                return tool_error(f"judge refused: {key} must be the id of an attachment on this card")
            files.append(kb.read_attachment_bytes(att, board=ctx.board))
        loaded.append((json.loads(files[0]), json.loads(files[2]), _rgb(files[1]), _rgb(files[3]),
                       [hashlib.sha256(data).hexdigest() for data in files[1::2]]))
    bundles = [bundle for capture in loaded for bundle in capture[:2]]
    if not all(isinstance(bundle, dict) for bundle in bundles):
        return tool_error("judge refused: a bundle is not a JSON object")
    for capture in loaded:  # each picture is the kit's stitched page: viewport width by page_height
        for bundle, image in zip(capture[:2], capture[2:4]):
            size = (bundle.get("width"), bundle.get("page_height"))
            if not all(type(v) is int and v > 0 for v in size) or image.size != size:
                return tool_error(f"judge refused: a picture is {image.width} x {image.height} px but its bundle "
                                  f"gives width {size[0]} and page_height {size[1]}")
    for field, expected, what in (("kit_sha256", _kit_sha256(), "another capture kit"),
                                  ("run_id", str(ctx.run), "another review run"),
                                  ("candidate_digest", digest, "another candidate")):
        if any(bundle.get(field) != expected for bundle in bundles):
            return tool_error(f"judge refused: a bundle comes from {what} (its {field} is not this review's)")
    versions = {bundle.get("chromium_version") for bundle in bundles}
    if len(versions) != 1 or not all(isinstance(v, str) and v for v in versions):
        return tool_error("judge refused: the bundles do not name one same Chromium version")
    wanted = [(page, width) for page in pages for width in WIDTHS]
    have = [(capture[0].get("page"), capture[0].get("width")) for capture in loaded]
    if sorted(have, key=repr) != sorted(wanted, key=repr):
        gaps = ", ".join(f"page {p} at {w} px" for p, w in wanted if (p, w) not in have) or "none"
        return tool_error(f"judge refused: the captures must be each compared page at 1280 and 375 px, "
                          f"once each (missing: {gaps})")
    loaded.sort(key=lambda capture: wanted.index((capture[0]["page"], capture[0]["width"])))

    allowed = parse_allowed_changes(task.body or "")  # the card body only, never comments
    rounds, previous = _round(kb, conn, ctx)
    judged = _classified(kb, conn, ctx)
    findings, pictures, regions, notes = [], {}, [], []
    for capture in loaded:
        comparison, cboxes, pboxes, undecided = _compare(capture, allowed)
        notes += comparison.notes
        for finding in comparison.findings:
            findings.append(finding)
            pictures[finding.id] = (capture, cboxes.get(finding.element), pboxes.get(finding.element))
        page, width = capture[0]["page"], capture[0]["width"]
        for r in undecided:  # its id binds a classification to this run, capture, rectangle and the crops shown
            crops = cut_crops(capture[2], capture[3], r, r, rounds, "0" * 16, width)  # only the bytes are used
            evidence = [ctx.run, page, width, capture[4], r,
                        [hashlib.sha256(data).hexdigest() for _, data in crops[:2]]]
            regions.append((hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()[:16], page,
                            width, f"region {r['x']},{r['y']} {r['width']}x{r['height']}", r, capture, crops))

    open_regions = [region for region in regions if region[0] not in judged]
    if open_regions:
        shown, content = [], []
        for rid, page, width, _, box, _, crops in open_regions[:MAX_REGIONS]:
            shown.append({"id": rid, "page": page, "width": width, "box": box})
            content += [{"type": "text", "text": f"Region {rid}: the candidate, then the prototype."},
                        _png_part(crops.candidate[1]), _png_part(crops.prototype[1])]
        text = json.dumps({
            "action": "judge", "verdict": "undecided", "round": rounds, "regions": shown,
            "remaining": len(open_regions) - len(shown),
            "next": ("Classify each region with classify: layout, type, colour or element, or none with a "
                     "one-sentence reason; then call judge again with the same captures."),
        })
        return {"_multimodal": True, "content": [{"type": "text", "text": text}, *content], "text_summary": text,
                "meta": {"regions": len(shown)}}

    for rid, page, width, element, box, capture, _ in regions:
        category = judged[rid].get("category")
        if category in CATEGORIES[:4]:
            findings.append(Finding(
                rid, "pixel_judged", page, width, element, "pixels",
                f"{box['width']} x {box['height']} px region judged {category}", "no differing region",
                f"{PIXEL_CHANNEL_DELTA} of 255 per channel, regions from {PIXEL_REGION_MIN_SIDE} x "
                f"{PIXEL_REGION_MIN_SIDE} px and {PIXEL_REGION_MIN_PIXELS} pixels"))
            pictures[rid] = (capture, box, box)

    kernel = []
    for finding in findings[:MAX_FINDINGS]:
        capture, cbox, pbox = pictures[finding.id]
        crops = cut_crops(capture[2], capture[3], cbox, pbox, rounds, finding.id, finding.width)
        names = [kb.get_attachment(conn, kb.store_attachment_bytes(
            conn, ctx.task, name, data, content_type="image/png", uploaded_by=UPLOADER, board=ctx.board,
            max_bytes=MAX_CROP_BYTES, expected_run_id=ctx.run)).filename for name, data in crops[:2]]
        tag = "pixel" if finding.kind == "pixel_judged" else finding.id  # a pixel finding has no recurrence id
        kernel.append({
            "severity": "blocking" if finding.kind == "element" else "major",
            "file": f"page {finding.page} at {finding.width} px", "lines": finding.element,
            "problem": f"{finding.kind} difference in {finding.property} [dc:{tag}]",
            "impact": (f"measured {finding.measured}, expected {finding.expected}, tolerance {finding.tolerance}; "
                       f"crops {names[0]} and {names[1]}" + (", scaled" if crops.scaled else "")),
            "smallest_fix": f"make {finding.property} of this element match the prototype",
            "candidate_digest": digest,
        })
    if findings and (rounds >= 3 or previous & {f.id for f in findings if f.kind != "pixel_judged"}):
        return json.dumps({"action": "judge", "verdict": "stop", "round": rounds,
                           "question": STOP.format(title=task.title, rounds=rounds, count=len(findings))})
    result = {"action": "judge", "verdict": "findings", "round": rounds, "candidate_digest": digest,
              "findings": kernel}
    if len(findings) > MAX_FINDINGS:
        result["more"] = f"and {len(findings) - MAX_FINDINGS} more of the same kinds"
    if notes:
        result["notes"] = notes
    return json.dumps(result)


def _classify(kb, conn, ctx, task, args):
    items, record = args.get("regions"), {}
    if not isinstance(items, list) or not items:
        return tool_error("classify needs regions: each with id, category and, for none, a reason")
    for item in items:
        item = item if isinstance(item, dict) else {}
        rid, category, reason = item.get("id"), item.get("category"), item.get("reason")
        reason = reason.strip() if isinstance(reason, str) else ""
        if not isinstance(rid, str) or not _ID.fullmatch(rid):
            return tool_error("classify refused: each region needs the 16 hex id judge gave it")
        if category not in CATEGORIES:
            return tool_error("classify refused: the category must be layout, type, colour, element or none; "
                              "motion timing and font smoothing are not categories")
        if category == "none" and (not reason or "\n" in reason):
            return tool_error("classify refused: none needs a one-sentence reason")
        record[rid] = {"category": category, "reason": reason}
    kb.store_attachment_bytes(conn, ctx.task, RECORD, json.dumps({"run_id": str(ctx.run), "regions": record},
                                                                 sort_keys=True).encode(),
                              content_type="application/json", uploaded_by=UPLOADER, board=ctx.board,
                              expected_run_id=ctx.run)
    return json.dumps({"action": "classify", "recorded": len(record),
                       "next": "Call judge again with the same captures."})


_ACTIONS = {"prepare": _prepare, "judge": _judge, "classify": _classify}


def design_compare(args: dict, task_id: Optional[str] = None):
    access = check_design_access(_config())
    if not access.allowed:
        return tool_error(REFUSED + access.reason)
    if access.role != REVIEWER:
        return tool_error(REFUSED + NOT_REVIEWER)
    action = args.get("action")
    if action not in _ACTIONS:
        return tool_error("action must be prepare, judge or classify")
    from hermes_cli import kanban_db as kb

    ctx = _Run(os.environ["HERMES_KANBAN_TASK"], int(os.environ["HERMES_KANBAN_RUN_ID"]),
               os.environ.get("HERMES_KANBAN_BOARD") or None)
    try:
        with closing(kb.connect(board=ctx.board)) as conn:
            return _ACTIONS[action](kb, conn, ctx, kb.get_task(conn, ctx.task), args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return tool_error(f"{action} failed: {exc}")


DESIGN_COMPARE_SCHEMA = {
    "name": "design_compare",
    "description": (
        "Reviewer only, in a review run: compare this website card with the chosen design file pinned in its "
        "body. prepare verifies the pin, stores the capture kit on this card and says what to import. judge "
        "reads the exported capture bundles and pictures and returns findings to submit unchanged, at most 4 "
        "undecided regions to look at, or verdict stop with one question for the owner. classify records your "
        "judgment of undecided regions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "captures": {
                "type": "array",
                "description": "judge: one entry per compared page and width, with the attachment ids on this "
                               "card of the exported bundles and pictures.",
                "items": {"type": "object", "properties": {key: {"type": "integer"} for key in CAPTURE_KEYS},
                          "required": list(CAPTURE_KEYS)},
            },
            "regions": {
                "type": "array",
                "description": "classify: each undecided region's id and category; none needs a one-sentence "
                               "reason. Motion timing and font smoothing are not categories.",
                "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "category": {"type": "string", "enum": list(CATEGORIES)},
                    "reason": {"type": "string"}}, "required": ["id", "category"]},
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="design_compare",
    toolset="design_vision",
    schema=DESIGN_COMPARE_SCHEMA,
    handler=lambda args, **kw: design_compare(args, task_id=kw.get("task_id")),
    emoji="🎨",
)
