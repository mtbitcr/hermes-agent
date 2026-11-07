"""design_compare, the reviewer's comparison of a website card with its pinned prototype (comparison
review plan, card C4).

Every case runs under the environment the dispatcher gives a worker: ``HERMES_KANBAN_DB`` and
``HERMES_KANBAN_BOARD`` pinned to the worker's own board, ``HERMES_KANBAN_TASK``,
``HERMES_KANBAN_RUN_ID`` and ``HERMES_KANBAN_CLAIM_LOCK`` set, and ``HERMES_HOME`` a profile home
under the root. Every call goes through the registry. Bundles are synthetic
design_compare_bundle_v1 documents with small PNG pictures: no browser runs, except in the one opt-in
real-Chromium test (``HERMES_E2E_BROWSER=1``, ``-m integration`` and a ``chromium`` on PATH).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import toolsets
from tools import design_compare_tool, kanban_tools  # noqa: F401  (registers the tool and kanban_attach)
from tools.design_access import (
    BOARD_NOT_LISTED, BUILDER, DESIGNER, NOT_IN_TASK, REVIEW_RUN_ONLY, REVIEWER, check_design_access,
)
from tools.design_compare_judge import finding_id
from tools.registry import registry

KIT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "design_compare"
KIT = hashlib.sha256((KIT_DIR / "capture.mjs").read_bytes() + (KIT_DIR / "probe.js").read_bytes()).hexdigest()
ENV = ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
       "HERMES_KANBAN_CLAIM_LOCK")
TITLE = "Home page hero"
PROTOTYPE = b"<!doctype html><title>chosen</title><h1>Hello</h1>\n"
HEIGHT = 240
REFUSED = "design compare refused: "
MISMATCH = (f"The chosen design file for '{TITLE}' is not the one this card was created with. "
            "Which design should Raphael compare against?")
CAPTURE_KEYS = ("candidate_bundle", "candidate_image", "prototype_bundle", "prototype_image")
STYLE = {"font-family": '"Geist", sans-serif', "font-face": "geist", "font-size": 16, "font-weight": 400,
         "line-height": 24, "color": [17, 17, 17, 1], "background-color": None, "border-top-color": None,
         "border-right-color": None, "border-bottom-color": None, "border-left-color": None,
         "text-decoration-color": None, "fill": None, "stroke": None}


def stop_question(rounds: int, count: int) -> str:
    return (f"The website change '{TITLE}' still differs from the chosen design after {rounds} review rounds: "
            f"{count} differences, with pictures on the card. Should Raphael keep the design exactly and try "
            "once more, or accept the change as it is now?")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def el(tag, text, box, landmarks=(("main", ""),), **style):
    return {"landmarks": [list(mark) for mark in landmarks], "tag": tag, "role": tag, "label": "", "alt": "",
            "text": text, "parent": None, "fixed": False, "box": dict(zip(("x", "y", "width", "height"), box)),
            "style": {**STYLE, **{k.replace("_", "-"): v for k, v in style.items()}}}


def elements(width, change=None):
    """The page at one width; ``change`` plants one difference on the candidate side."""
    els = {
        "header": el("header", "", (0, 0, width, 60), landmarks=()),
        "h1": el("h1", "Hello", (20, 10, 200, 40), landmarks=(("header", "Hello"),), font_size=32,
                 font_weight=700, line_height=40),
        "p": el("p", "Welcome to the site", (20, 80, width - 40, 24)),
        "link": el("a", "Start", (20, 120, 80, 24), color=[0, 90, 200, 1]),
        "footer": el("footer", "", (0, 200, width, 40), landmarks=()),
        "contact": el("a", "Contact", (20, 208, 80, 24), landmarks=(("footer", ""),)),
    }
    if change == "layout":
        els["header"]["box"]["y"] = 6
    elif change == "type":
        els["h1"]["style"]["font-weight"] = 500
    elif change == "colour":
        els["link"]["style"]["color"] = [0, 130, 200, 1]
    elif change == "missing":
        del els["contact"]
    elif change == "extra":
        els["badge"] = el("span", "New", (120, 120, 40, 20))
    elif change == "text":
        els["link"]["text"] = "Begin"
    elif change == "moved-text":
        els["link"]["text"], els["link"]["box"]["x"] = "Begin", 250
    return list(els.values())


def png(width, *blocks) -> bytes:
    image = Image.new("RGB", (width, HEIGHT), "white")
    for block in filter(None, blocks):  # red, unless a colour follows the corners
        ImageDraw.Draw(image).rectangle(block[:4], fill=block[4:] or (200, 0, 0))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def write_config(**design) -> None:
    """``config.yaml`` in the worker's own HERMES_HOME (JSON is YAML)."""
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(json.dumps({"design_browser": design}),
                                                              encoding="utf-8")


class Kanban:
    """Real boards under the root, a done design card holding the prototype, its child build card,
    and the dispatcher's environment for the build card's current run."""

    def __init__(self, kb, root: Path, monkeypatch) -> None:
        self.kb, self.root, self.mp = kb, root, monkeypatch

    def start(self, body="{pin}", *, assignee=BUILDER, review=True, data=PROTOTYPE, filename="chosen.html",
              pinned=PROTOTYPE, on_card=False) -> dict:
        kb = self.kb
        for var in ENV:
            self.mp.delenv(var, raising=False)
        with contextlib.closing(kb.connect(board="web")) as conn:
            design = kb.create_task(conn, title="Design options", assignee=DESIGNER)
            prototype = kb.store_attachment_bytes(conn, design, filename, data, uploaded_by="agent", board="web")
            kb.claim_task(conn, design)
            kb.complete_task(conn, design, summary="chosen")
            build = kb.create_task(conn, title=TITLE, assignee=assignee, parents=[design])
            if on_card:
                prototype = kb.store_attachment_bytes(conn, build, filename, data, uploaded_by="agent", board="web")
            pin = f"Prototype pin: attachment={prototype} sha256={sha(pinned)} file=chosen.html\nCompare pages: /"
            conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (body.format(pin=pin), build))
            conn.commit()
            first = kb.claim_task(conn, build)
            if review:
                kb.request_review(conn, build, summary="ready", reviewer=REVIEWER,
                                  expected_run_id=first.current_run_id)
                kb.claim_review_task(conn, build, claimer=f"{REVIEWER}:1")
            task = kb.get_task(conn, build)
            from hermes_cli.kanban_review_record import record_candidate
            record = record_candidate(conn, build) if review else None
        for key, value in {
            "HERMES_KANBAN_DB": str(self.db("web")), "HERMES_KANBAN_BOARD": "web", "HERMES_KANBAN_TASK": build,
            "HERMES_KANBAN_RUN_ID": str(task.current_run_id), "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock or "",
        }.items():
            self.mp.setenv(key, value)
        return {"task": build, "run": task.current_run_id, "prototype": prototype,
                "digest": record.digest if record else None}

    def db(self, board: str) -> Path:
        return self.root / "kanban" / "boards" / board / "kanban.db"

    def raw(self, sql: str, *params, board="web"):
        """A separate read-only connection to the board's own file."""
        with contextlib.closing(sqlite3.connect(self.db(board).as_uri() + "?mode=ro", uri=True)) as conn:
            return conn.execute(sql, params).fetchall()

    def export(self, ctx, change=None, *, block=None, paint=None, widths=(1280, 375), **override) -> list:
        """Store each width's two bundles and two pictures on the card, as the reviewer's artifact export
        does, and return the judge's captures. ``block`` is drawn on the candidate at 1280 px, ``paint``
        on its ``candidate`` or ``prototype`` side at both widths."""
        captures, paint = [], paint or {}
        with contextlib.closing(self.kb.connect()) as conn:
            for width in widths:
                common = {"schema": "design_compare_bundle_v1", "kit_sha256": KIT,
                          "chromium_version": "153.0.8010.0", "run_id": str(ctx["run"]),
                          "candidate_digest": ctx["digest"], "page": "/", "width": width, "page_height": HEIGHT,
                          "image": f"index-{width}.png", "blocked_requests": [], "masks": []}
                files = (
                    (f"cand-{width}.json", {**common, "elements": elements(width, change), **override}),
                    (f"cand-{width}.png", png(width, block if width == 1280 else None, paint.get("candidate"))),
                    (f"proto-{width}.json", {**common, "elements": elements(width)}),
                    (f"proto-{width}.png", png(width, paint.get("prototype"))),
                )
                captures.append({key: self.kb.store_attachment_bytes(
                    conn, ctx["task"], name, data if isinstance(data, bytes) else json.dumps(data).encode(),
                    uploaded_by="agent", board="web", expected_run_id=ctx["run"])
                    for key, (name, data) in zip(CAPTURE_KEYS, files)})
        return captures

    def handed_back(self, ctx, *ids) -> dict:
        """An earlier round whose findings carry ``ids``, handed back natively, then a fresh review run."""
        return self.next_review(ctx, [{"severity": "major", "file": "page / at 1280 px", "lines": "earlier",
                                       "impact": "earlier", "problem": f"layout difference in box.y [dc:{i}]",
                                       "smallest_fix": "fix it", "candidate_digest": ctx["digest"]} for i in ids])

    def next_review(self, ctx, findings) -> dict:
        """The native typed handback of ``findings``, the builder's next run and a fresh review run."""
        kb = self.kb
        with contextlib.closing(kb.connect()) as conn:
            result = kb.submit_review_findings(conn, ctx["task"], findings=findings, candidate_digest=ctx["digest"],
                                               expected_run_id=ctx["run"])
            assert result["outcome"] == "handed_back", result
            first = kb.claim_task(conn, ctx["task"])
        return self.review({**ctx, "run": first.current_run_id})

    def review(self, ctx) -> dict:
        """The builder's review request from its run ``ctx``, the reviewer's claim and that review run."""
        kb = self.kb
        with contextlib.closing(kb.connect()) as conn:
            assert kb.request_review(conn, ctx["task"], summary="next candidate", reviewer=REVIEWER,
                                     expected_run_id=ctx["run"])
            task = kb.claim_review_task(conn, ctx["task"], claimer=f"{REVIEWER}:2")
            from hermes_cli.kanban_review_record import record_candidate
            digest = record_candidate(conn, ctx["task"]).digest
        self.mp.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
        self.mp.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock or "")
        return {**ctx, "run": task.current_run_id, "digest": digest}


@pytest.fixture
def kanban(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "root"
    home = root / "profiles" / REVIEWER
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_DELEGATED_CHILD_CONTEXT", *ENV):
        monkeypatch.delenv(var, raising=False)
    for board in ("web", "elsewhere"):
        kb.create_board(board)
        kb.connect(board=board).close()
    write_config(boards=["web", "elsewhere"])
    return Kanban(kb, root, monkeypatch)


def call(args: dict):
    return registry.dispatch("design_compare", dict(args), task_id="design-compare-test")


def ok(result) -> dict:
    assert isinstance(result, str), result
    payload = json.loads(result)
    assert "error" not in payload, payload
    return payload


def error(result) -> str:
    assert isinstance(result, str), result
    return json.loads(result)["error"]


def judge(captures):
    return ok(call({"action": "judge", "captures": captures}))


def attachments(kanban, task):
    return kanban.raw("SELECT id, filename FROM task_attachments WHERE task_id = ? ORDER BY id", task)


def stored_in_run(kanban, task, filename) -> list:
    """The run ids recorded by the kernel for each attachment of that name (its ``attached`` receipt)."""
    return [run for (run,) in kanban.raw(
        "SELECT e.run_id FROM task_events e JOIN task_attachments a ON a.id = json_extract(e.payload, "
        "'$.attachment_id') WHERE e.kind = 'attached' AND a.task_id = ? AND a.filename = ?", task, filename)]


def stored_image(kanban, filename) -> Image.Image:
    (path,), = kanban.raw("SELECT stored_path FROM task_attachments WHERE filename = ?", filename)
    return Image.open(path).convert("RGB")


def delivered(kanban, task) -> int:
    """The kernel's review handbacks on the card (its ``review_findings_delivered`` events)."""
    (count,), = kanban.raw("SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = "
                           "'review_findings_delivered'", task)
    return count


def upload_findings(ctx, tag, index) -> None:
    """A findings document the builder uploads through kanban_attach, as any agent may; ``index`` varies its
    text, so the kernel stores each one."""
    finding = {"severity": "major", "file": "page / at 1280 px", "lines": "example", "impact": f"example {index}",
               "problem": f"colour difference in pixels [dc:{tag}]", "smallest_fix": "match the design",
               "candidate_digest": "example"}
    result = json.loads(registry.dispatch("kanban_attach", {
        "task_id": ctx["task"], "filename": "review_findings.json", "content_type": "application/json",
        "content_text": json.dumps({"schema_version": 1, "candidate_digest": "example", "findings": [finding]}),
        "technical_artifact": False}, task_id="design-compare-test"))
    assert result.get("ok") is True, result


# ── Access: the board setting and the role gate, then the reviewer alone ─────


def test_the_tool_is_in_design_vision_only():
    assert "design_compare" in toolsets.TOOLSETS["design_vision"]["tools"]
    assert registry.get_entry("design_compare").toolset == "design_vision"
    assert not [name for name, ts in toolsets.TOOLSETS.items() if "design_vision" in ts.get("includes", [])]


@pytest.mark.parametrize("case", ["outside-a-task", "builder", "designer", "unlisted-board", "not-a-review-run"])
def test_refused_without_the_reviewer_on_a_review_run_on_a_listed_board(kanban, monkeypatch, case):
    config, reason = {"boards": ["web"]}, "only the reviewer may use it, in a review run"
    if case == "outside-a-task":
        kanban.start()
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        reason = NOT_IN_TASK
    elif case == "builder":  # a builder run the gate lets in: its own preview exists
        ctx = kanban.start(review=False)
        with contextlib.closing(kanban.kb.connect()) as conn:
            kanban.kb.set_branch_name(conn, ctx["task"], f"web/{ctx['task']}")
        config["allowed_hosts"] = ["*.raphael-first-project.pages.dev"]
    elif case == "designer":
        kanban.start(assignee=DESIGNER, review=False)
    elif case == "unlisted-board":
        kanban.start()
        config["boards"], reason = ["elsewhere"], BOARD_NOT_LISTED
    else:
        kanban.start(assignee=REVIEWER, review=False)
        reason = REVIEW_RUN_ONLY
    write_config(**config)
    assert check_design_access(config).allowed is (case in ("builder", "designer"))
    before = kanban.raw("SELECT COUNT(*) FROM task_attachments")
    for args in ({"action": "prepare"}, {"action": "judge", "captures": []},
                 {"action": "classify", "regions": [{"id": "0" * 16, "category": "none", "reason": "same."}]}):
        assert error(call(args)) == REFUSED + reason
    assert kanban.raw("SELECT COUNT(*) FROM task_attachments") == before


# ── prepare: the pin ─────────────────────────────────────────────────────────


def test_prepare_verifies_the_pin_and_stores_the_hashed_kit(kanban):
    ctx = kanban.start()
    result = ok(call({"action": "prepare"}))
    assert result["prototype"] == {"attachment_id": ctx["prototype"], "sha256": sha(PROTOTYPE)}
    assert result["kit_sha256"] == KIT
    assert (result["pages"], result["widths"]) == (["/"], [1280, 375])
    assert (result["run_id"], result["candidate_digest"]) == (str(ctx["run"]), ctx["digest"])
    for entry in result["kit"]:
        data = (KIT_DIR / entry["filename"]).read_bytes()
        assert entry["sha256"] == sha(data)
        assert stored_in_run(kanban, ctx["task"], entry["filename"]) == [ctx["run"]]
    assert sorted(e["filename"] for e in result["kit"]) == ["capture.mjs", "probe.js"]


@pytest.mark.parametrize("case", ["no-pin", "malformed-pin", "not-html", "on-the-build-card", "one-byte"])
def test_prepare_refuses_a_bad_pin_and_compares_nothing(kanban, case):
    if case == "no-pin":
        ctx = kanban.start("Compare pages: /")
    elif case == "malformed-pin":
        ctx = kanban.start(f"Prototype pin: attachment=1 sha256={sha(PROTOTYPE)[:63]} file=chosen.html\n"
                           "Compare pages: /")
    elif case == "not-html":
        ctx = kanban.start(filename="chosen.txt")
    elif case == "on-the-build-card":
        ctx = kanban.start(on_card=True)
    else:
        ctx = kanban.start(data=PROTOTYPE[:-2] + b"!\n")
    before = attachments(kanban, ctx["task"])
    for args in ({"action": "prepare"}, {"action": "judge", "captures": kanban.export(ctx)}):
        if case in ("no-pin", "malformed-pin"):
            assert error(call(args)) == "this card names no chosen design file"
        else:
            assert ok(call(args)) == {"action": args["action"], "verdict": "stop", "question": MISMATCH}
        before = attachments(kanban, ctx["task"])  # the export above added only the reviewer's files
    assert attachments(kanban, ctx["task"]) == before


# ── judge ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("override,word", [
    ({"kit_sha256": "0" * 64}, "capture kit"), ({"run_id": "1"}, "review run"),
    ({"candidate_digest": "record:999999"}, "candidate"), ({"chromium_version": "153.0.8010.1"}, "Chromium"),
])
def test_judge_refuses_bundles_from_another_kit_run_candidate_or_browser(kanban, override, word):
    ctx = kanban.start()
    before = attachments(kanban, ctx["task"])
    captures = kanban.export(ctx, "layout", **override)
    message = error(call({"action": "judge", "captures": captures}))
    assert message.startswith("judge refused: ") and word in message
    assert len(attachments(kanban, ctx["task"])) == len(before) + 8  # no crop was stored


def test_judge_refuses_captures_that_miss_a_width(kanban):
    ctx = kanban.start()
    assert "375" in error(call({"action": "judge", "captures": kanban.export(ctx, widths=(1280,))}))


@pytest.mark.parametrize("case", ["width", "height", "no-page-height"])
def test_judge_refuses_pictures_that_contradict_their_bundles(kanban, case):
    ctx = kanban.start()
    if case == "no-page-height":
        captures = kanban.export(ctx, "layout", page_height=None)
    else:  # both pictures 1 px wide, or 1 px high, at each width
        captures = kanban.export(ctx, "layout")
        with contextlib.closing(kanban.kb.connect()) as conn:
            for capture, width in zip(captures, (1280, 375)):
                data = io.BytesIO()
                Image.new("RGB", (1, HEIGHT) if case == "width" else (width, 1), "white").save(data, "PNG")
                for key in ("candidate_image", "prototype_image"):
                    capture[key] = kanban.kb.store_attachment_bytes(
                        conn, ctx["task"], f"{case}-{key}-{width}.png", data.getvalue(), uploaded_by="agent",
                        board="web", expected_run_id=ctx["run"])
    before = attachments(kanban, ctx["task"])
    payload = json.loads(call({"action": "judge", "captures": captures}))
    assert "error" in payload, payload
    assert payload["error"].startswith("judge refused: ") and "page_height" in payload["error"]
    assert attachments(kanban, ctx["task"]) == before  # no crop was stored


def test_equal_bundles_at_both_widths_give_no_findings(kanban):
    ctx = kanban.start()
    assert judge(kanban.export(ctx)) == {"action": "judge", "verdict": "findings", "round": 1,
                                        "candidate_digest": ctx["digest"], "findings": []}


@pytest.mark.parametrize("change,kind,prop,key", [
    ("layout", "layout", "box.y", '/header:""#1'),
    ("type", "type", "font-weight", 'header:hello/h1:"hello"#1'),
    ("colour", "colour", "color", 'main/a:"start"#1'),
    ("missing", "element", "presence", 'footer/a:"contact"#1'),
    ("extra", "element", "presence", 'main/span:"new"#1'),
])
def test_each_planted_category_is_one_finding_per_width_with_two_stored_crops(kanban, change, kind, prop, key):
    ctx = kanban.start()
    result = judge(kanban.export(ctx, change))
    findings = result["findings"]
    assert len(findings) == 2
    for finding, width in zip(findings, (1280, 375)):
        fid = finding_id("/", width, kind, key, prop)
        crops = [f"dc-r1-{fid}-{width}-candidate.png", f"dc-r1-{fid}-{width}-prototype.png"]
        assert finding == {
            "severity": "blocking" if kind == "element" else "major", "file": f"page / at {width} px",
            "lines": key, "problem": f"{kind} difference in {prop} [dc:{fid}]",
            "impact": finding["impact"], "smallest_fix": f"make {prop} of this element match the prototype",
            "candidate_digest": ctx["digest"],
        }
        scaled = ", scaled" if (kind, width) == ("layout", 1280) else ""  # the header spans the full width
        assert finding["impact"].startswith("measured ") and finding["impact"].endswith(
            f"; crops {crops[0]} and {crops[1]}{scaled}")
        for name in crops:
            assert stored_in_run(kanban, ctx["task"], name) == [ctx["run"]]
            (path, size), = kanban.raw("SELECT stored_path, size FROM task_attachments WHERE filename = ?", name)
            assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and size <= 1024 * 1024
    if change == "layout":
        assert findings[0]["impact"].startswith("measured 6 px, expected 0 px, tolerance plus or minus 2 px; ")
    kanban.kb.build_review_findings_document(findings, candidate_digest=ctx["digest"])  # the kernel accepts them


@pytest.mark.parametrize("change,paint,key,prop", [
    ("missing", {"prototype": (20, 208, 99, 231)}, 'footer/a:"contact"#1', "presence"),
    ("extra", {"candidate": (120, 120, 159, 139)}, 'main/span:"new"#1', "presence"),
    ("text", {"candidate": (24, 124, 70, 139)}, 'main/a:"start"#1', "text"),
])
def test_pixels_an_element_finding_explains_are_not_judged_again(kanban, change, paint, key, prop):
    ctx = kanban.start()
    result = ok(call({"action": "judge", "captures": kanban.export(ctx, change, paint=paint)}))  # not undecided
    ids = [finding_id("/", width, "element", key, prop) for width in (1280, 375)]
    assert [f["problem"] for f in result["findings"]] == [f"element difference in {prop} [dc:{fid}]" for fid in ids]
    for finding, fid, width in zip(result["findings"], ids, (1280, 375)):
        crops = [f"dc-r1-{fid}-{width}-candidate.png", f"dc-r1-{fid}-{width}-prototype.png"]
        assert finding["impact"].endswith(f"; crops {crops[0]} and {crops[1]}")
        assert [stored_in_run(kanban, ctx["task"], name) for name in crops] == [[ctx["run"]]] * 2


def test_a_text_pair_finding_crops_the_union_of_both_boxes(kanban):
    ctx = kanban.start('{pin}\nAllowed text change: "Start" becomes "Begin"')
    paint = {"candidate": (250, 120, 329, 143), "prototype": (20, 120, 99, 143)}  # the link on each side
    findings = judge(kanban.export(ctx, "moved-text", paint=paint))["findings"]
    assert [(f["lines"], f["problem"].split(" [")[0]) for f in findings] == [
        ('main/a:"start"#1', "layout difference in box.x")] * 2
    for finding in findings:
        names = finding["impact"].split("; crops ")[1].split(" and ")
        candidate, prototype = (stored_image(kanban, name) for name in names)
        assert candidate.size == prototype.size == (354, 72)  # page x 0 to 354, y 96 to 168
        assert candidate.getpixel((250, 24)) == candidate.getpixel((329, 47)) == (200, 0, 0)
        assert prototype.getpixel((20, 24)) == prototype.getpixel((99, 47)) == (200, 0, 0)


def test_allowed_text_is_read_from_the_body_never_from_comments(kanban):
    line = 'Allowed text change: "Start" becomes "Begin"'
    ctx = kanban.start("{pin}\n" + line)
    assert judge(kanban.export(ctx, "text"))["findings"] == []

    ctx = kanban.start()
    with contextlib.closing(kanban.kb.connect()) as conn:
        kanban.kb.add_comment(conn, ctx["task"], "raphael-planner", line)
    findings = judge(kanban.export(ctx, "text"))["findings"]
    assert [f["problem"].split(" [")[0] for f in findings] == ["element difference in text"] * 2


# ── The stop rule ────────────────────────────────────────────────────────────


def test_the_same_finding_in_two_rounds_stops_with_pictures_on_the_card(kanban):
    ctx = kanban.start()
    fid = finding_id("/", 1280, "layout", '/header:""#1', "box.y")
    ctx = kanban.handed_back(ctx, fid)
    result = judge(kanban.export(ctx, "layout"))
    assert result == {"action": "judge", "verdict": "stop", "round": 2, "question": stop_question(2, 2)}
    assert stored_in_run(kanban, ctx["task"], f"dc-r2-{fid}-1280-candidate.png") == [ctx["run"]]
    assert delivered(kanban, ctx["task"]) == 1


def test_a_pixel_difference_never_recurs_round_three_stops_and_a_classification_decides_only_its_region(kanban):
    ctx, ids, handed = kanban.start(), [], []
    for rounds in (1, 2, 3):  # the same 12 x 12 px difference inside the paragraph, in each round's review run
        captures = kanban.export(ctx, block=(100, 84, 111, 95))
        (region,) = json.loads(call({"action": "judge", "captures": captures})["text_summary"])["regions"]
        ok(call({"action": "classify", "regions": [{"id": region["id"], "category": "colour"}]}))
        ids.append(region["id"])
        result = judge(captures)  # the unchanged-capture retry keeps its classification
        if rounds < 3:  # no stop by recurrence: handed back natively, then a fresh review run
            assert (result["verdict"], result["round"]) == ("findings", rounds)
            handed += result["findings"]
            ctx = kanban.next_review(ctx, result["findings"])
    assert result == {"action": "judge", "verdict": "stop", "round": 3, "question": stop_question(3, 1)}
    assert len(set(ids)) == 3
    assert [(f["lines"], f["problem"]) for f in handed] == [
        ("region 100,84 12x12", "pixel_judged difference in pixels [dc:pixel]")] * 2
    assert stored_in_run(kanban, ctx["task"], f"dc-r3-{ids[2]}-1280-candidate.png") == [ctx["run"]]
    for block, box in (((100, 84, 113, 95), {**region["box"], "width": 14}),  # a wider rectangle
                       ((100, 84, 111, 95, 0, 0, 200), region["box"])):  # another colour at the same rectangle
        shown = call({"action": "judge", "captures": kanban.export(ctx, block=block)})
        assert isinstance(shown, dict), shown  # a different region of the same element: shown again as undecided
        (again,) = json.loads(shown["text_summary"])["regions"]
        assert (again["box"], again["id"] != region["id"]) == (box, True)


def test_round_three_with_findings_stops(kanban):
    ctx = kanban.handed_back(kanban.handed_back(kanban.start(), "0" * 16), "1" * 16)
    result = judge(kanban.export(ctx, "type"))
    assert (result["verdict"], result["question"], "findings" in result) == ("stop", stop_question(3, 2), False)


def test_round_three_with_zero_findings_passes(kanban):
    ctx = kanban.handed_back(kanban.handed_back(kanban.start(), "0" * 16), "1" * 16)
    result = judge(kanban.export(ctx))
    assert (result["verdict"], result["round"], result["findings"]) == ("findings", 3, [])


@pytest.mark.parametrize("case", ["fake-element-recurrence", "fake-element-round-limit", "fake-pixel-round-limit"])
def test_builder_attachment_is_not_a_review_round(kanban, case):
    ctx = kanban.start(review=False)  # the builder's own run uploads them
    fid = finding_id("/", 1280, "layout", '/header:""#1', "box.y")
    tags = {"fake-element-recurrence": [fid], "fake-element-round-limit": ["0" * 16, "1" * 16],
            "fake-pixel-round-limit": ["pixel", "pixel"]}[case]
    for index, tag in enumerate(tags):
        upload_findings(ctx, tag, index)
    uploads = kanban.raw("SELECT filename, uploaded_by FROM task_attachments WHERE task_id = ? ORDER BY id",
                         ctx["task"])
    assert uploads == [("review_findings.json", "agent"), ("review_findings (1).json", "agent")][:len(tags)]
    ctx = kanban.review(ctx)
    result = judge(kanban.export(ctx, "layout" if case == "fake-element-recurrence" else "type"))
    assert (result["verdict"], result["round"]) == ("findings", 1), result
    assert (len(result["findings"]), delivered(kanban, ctx["task"])) == (2, 0)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Review control</title><style>
html,body{margin:0;padding:0;font:16px Arial,sans-serif;color:#111;background:#fff}body{height:300px}
main{position:relative;height:300px}
h1{position:absolute;left:20px;top:10px;margin:0;width:300px;height:40px;font:700 32px/40px Arial,sans-serif}
p{position:absolute;left:20px;top:80px;margin:0;width:300px;height:40px}
</style></head><body><main><h1>Hello</h1><p>Welcome</p></main></body></html>"""


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
                    reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in")
@pytest.mark.skipif(not shutil.which("chromium"), reason="Chromium not installed")
def test_real_chromium_builder_pixel_documents_leave_the_first_review_in_round_one(kanban, tmp_path):
    ctx = kanban.start(review=False, data=PAGE.encode(), pinned=PAGE.encode())
    for index in (0, 1):
        upload_findings(ctx, "pixel", index)
    ctx = kanban.review(ctx)
    for side, html in (("candidate", PAGE.replace("</style>", "h1{font-weight:500}</style>")), ("prototype", PAGE)):
        (tmp_path / side).mkdir()
        for name in ("index.html", "chosen.html"):
            (tmp_path / side / name).write_text(html)
        kit = subprocess.run(  # the unchanged kit, which refuses any Chromium but its pinned build
            ["node", str(KIT_DIR / "capture.mjs"), "--side", side, "--site", str(tmp_path / side), "--prototype",
             str(tmp_path / side / "chosen.html"), "--pages", "/", "--out", str(tmp_path / side / "out"),
             "--run-id", str(ctx["run"]), "--candidate-digest", ctx["digest"], "--chromium", shutil.which("chromium")],
            capture_output=True, text=True, timeout=60)
        assert kit.returncode == 0, kit.stderr
    captures = []
    with contextlib.closing(kanban.kb.connect()) as conn:
        for width in (1280, 375):
            files = [next((tmp_path / side / "out").glob(f"*-{width}.{ext}")) for side in ("candidate", "prototype")
                     for ext in ("json", "png")]
            captures.append({key: kanban.kb.store_attachment_bytes(
                conn, ctx["task"], path.name, path.read_bytes(), uploaded_by="agent", board="web",
                expected_run_id=ctx["run"]) for key, path in zip(CAPTURE_KEYS, files)})
    result = judge(captures)
    assert (result["verdict"], result["round"]) == ("findings", 1), result
    assert (len(result["findings"]), delivered(kanban, ctx["task"])) == (2, 0)


# ── Undecided regions and classify ───────────────────────────────────────────


def test_undecided_regions_are_shown_then_classified(kanban):
    ctx = kanban.start()
    captures = kanban.export(ctx, "layout", block=(100, 84, 111, 95))  # inside the paragraph, which matches
    shown = call({"action": "judge", "captures": captures})
    assert isinstance(shown, dict) and shown["_multimodal"] is True
    summary = json.loads(shown["text_summary"])
    assert (summary["verdict"], summary["remaining"]) == ("undecided", 0)
    (region,) = summary["regions"]
    assert (region["page"], region["width"], region["box"]) == ("/", 1280, {"x": 100, "y": 84, "width": 12,
                                                                           "height": 12})
    images = [part["image_url"]["url"] for part in shown["content"] if part["type"] == "image_url"]
    assert len(images) == 2
    assert all(base64.b64decode(url.split(",", 1)[1])[:4] == b"\x89PNG" for url in images)

    for bad in ({"id": region["id"], "category": "motion"}, {"id": region["id"], "category": "none"}):
        assert "error" in json.loads(call({"action": "classify", "regions": [bad]}))
    layout = [finding_id("/", width, "layout", '/header:""#1', "box.y") for width in (1280, 375)]
    ok(call({"action": "classify", "regions": [
        {"id": region["id"], "category": "none", "reason": "A rendering speck."},
        {"id": layout[0], "category": "none", "reason": "Measured findings cannot be dismissed."}]}))
    assert [f["problem"] for f in judge(captures)["findings"]] == [
        f"layout difference in box.y [dc:{fid}]" for fid in layout]

    ok(call({"action": "classify", "regions": [{"id": region["id"], "category": "colour"}]}))
    findings = judge(captures)["findings"]
    assert findings[-1]["severity"] == "major"
    assert findings[-1]["problem"] == "pixel_judged difference in pixels [dc:pixel]"
    assert stored_in_run(kanban, ctx["task"], f"dc-r1-{region['id']}-1280-prototype.png") == [ctx["run"]]


# ── The dispatcher's worker environment ──────────────────────────────────────


def test_under_the_dispatcher_environment_everything_lands_on_the_workers_board_under_the_root(kanban):
    ctx = kanban.start()
    home = Path(os.environ["HERMES_HOME"])
    assert home.parent == kanban.root / "profiles"
    assert os.environ["HERMES_KANBAN_DB"] == str(kanban.db("web"))
    assert check_design_access({"boards": ["web", "elsewhere"]}).allowed
    kit = {e["filename"]: e["sha256"] for e in ok(call({"action": "prepare"}))["kit"]}
    findings = judge(kanban.export(ctx, "colour"))["findings"]
    crops = [name for f in findings for name in f["impact"].split("; crops ")[1].removesuffix(", scaled").split(" and ")]

    folder = kanban.root / "kanban" / "boards" / "web" / "attachments" / ctx["task"]
    rows = {name: (Path(path), by) for name, path, by in kanban.raw(
        "SELECT filename, stored_path, uploaded_by FROM task_attachments WHERE task_id = ?", ctx["task"])}
    for name in [*kit, *crops]:
        path, by = rows[name]
        assert path.parent == folder and path.is_file()
        assert stored_in_run(kanban, ctx["task"], name) == [ctx["run"]]
    assert {name: sha(rows[name][0].read_bytes()) for name in kit} == kit
    assert not (home / "kanban").exists() and not (home / "kanban.db").exists()
    assert kanban.raw("SELECT COUNT(*) FROM task_attachments", board="elsewhere") == [(0,)]
    assert {"web", "elsewhere"} <= {b["slug"] for b in kanban.kb.list_boards()}
