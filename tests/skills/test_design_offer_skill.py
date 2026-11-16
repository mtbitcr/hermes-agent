"""Tests for optional-skills/creative/design-offer: the designer offers one picture per
option (1280 px and 375 px side by side), then writes and checks one full-page file.

stdlib + pytest, no network, no browser, no Pillow. The pictures are composed by the
fixed browser script compose_offer.mjs (tests-js/design-offer-compose.test.ts); this
file pins the skill text and the two Python helpers, embed_fonts.py and
check_full_page.py, with fake website folders and fake capture bundles.
"""

import ast
import base64
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "creative" / "design-offer"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS = SKILL_DIR / "scripts"
OLD_COMPOSER = SCRIPTS / "compose_offer.py"
COMPOSER = SCRIPTS / "compose_offer.mjs"
EMBED = SCRIPTS / "embed_fonts.py"
CHECK = SCRIPTS / "check_full_page.py"
JUDGE = REPO / "tools" / "design_compare_judge.py"
CAPTURE_MJS = REPO / "scripts" / "design_compare" / "capture.mjs"

GEIST = "@fontsource-variable/geist"
GEIST_MONO = "@fontsource-variable/geist-mono"
GEIST_FILE = "geist-latin-wght-normal.woff2"
MONO_FILE = "geist-mono-latin-wght-normal.woff2"
GEIST_BYTES = b"wOF2" + bytes(range(256)) * 3
MONO_BYTES = b"wOF2" + bytes(range(255, -1, -1)) * 2
PAGE_TEXT = "Secret page text that must never be printed"
PAGE = f"<!doctype html><html><head><title>t</title></head><body><p>{PAGE_TEXT}</p></body></html>"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path, name: str):
    assert path.is_file(), f"{path.name} is missing"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The scripts of the skill
# ---------------------------------------------------------------------------


def test_the_pillow_composer_is_replaced_by_the_browser_script():
    assert not OLD_COMPOSER.exists(), "compose_offer.py must be deleted"
    for path in (COMPOSER, EMBED, CHECK):
        assert path.is_file(), f"{path.name} is missing"
    assert sorted(p.name for p in SCRIPTS.iterdir() if p.is_file()) == [
        "check_full_page.py", "compose_offer.mjs", "embed_fonts.py"]


@pytest.mark.parametrize("script", [EMBED, CHECK], ids=lambda p: p.name)
def test_python_helpers_use_only_the_standard_library_and_the_judge(script):
    assert script.is_file(), f"{script.name} is missing"
    imported = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"design_compare_judge"}, imported
    assert "PIL" not in script.read_text(encoding="utf-8")


def test_no_script_of_the_skill_uses_pillow():
    for path in (p for p in SCRIPTS.iterdir() if p.is_file() and p.suffix in (".py", ".mjs")):
        assert not re.search(r"\bPIL\b|Pillow", path.read_text(encoding="utf-8")), path.name


# ---------------------------------------------------------------------------
# embed_fonts.py
# ---------------------------------------------------------------------------


def _website(tmp_path, geist_pin="5.3.0", mono_pin="5.3.0", geist_version="5.3.0", mono_version="5.3.0",
             installed=True):
    site = tmp_path / "website"
    site.mkdir()
    (site / "package.json").write_text(json.dumps({
        "name": "site", "dependencies": {GEIST: geist_pin, GEIST_MONO: mono_pin}}))
    if installed:
        for pkg, version, name, data in ((GEIST, geist_version, GEIST_FILE, GEIST_BYTES),
                                         (GEIST_MONO, mono_version, MONO_FILE, MONO_BYTES)):
            folder = site / "node_modules" / pkg
            (folder / "files").mkdir(parents=True)
            (folder / "package.json").write_text(json.dumps({"name": pkg, "version": version}))
            (folder / "files" / name).write_bytes(data)
    return site


def _page(tmp_path, text=PAGE):
    page = tmp_path / "full-page.html"
    page.write_text(text, encoding="utf-8")
    return page


def _faces(css: str) -> dict:
    faces = {}
    for block in re.findall(r"@font-face\s*\{[^}]*\}", css):
        family = re.search(r"font-family:\s*'([^']+)'", block).group(1)
        weight = re.search(r"font-weight:\s*([0-9 ]+?)\s*;", block).group(1)
        url = re.search(r"url\(\s*data:font/woff2;base64,([A-Za-z0-9+/=]+)\s*\)", block)
        assert url, block
        faces[family] = (weight, base64.b64decode(url.group(1)))
    return faces


def test_embed_fonts_writes_two_data_url_faces_under_the_websites_family_names(tmp_path):
    embed = _load(EMBED, "embed_fonts")
    page = _page(tmp_path)
    embed.embed(_website(tmp_path), page)
    out = page.read_text(encoding="utf-8")
    assert _faces(out) == {"Geist": ("400 500", GEIST_BYTES), "Geist Mono": ("400", MONO_BYTES)}
    assert out.index("@font-face") < out.index("</head>")
    assert f"<p>{PAGE_TEXT}</p>" in out and "<title>t</title>" in out


def test_embed_fonts_command_line_prints_each_fonts_sha256_and_no_page_text(tmp_path):
    page = _page(tmp_path)
    done = subprocess.run([sys.executable, str(EMBED), "--website", str(_website(tmp_path)), "--page", str(page)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert _sha(GEIST_BYTES) in done.stdout and _sha(MONO_BYTES) in done.stdout
    assert PAGE_TEXT not in done.stdout + done.stderr
    assert "@font-face" in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("variant", [
    {"geist_version": "5.3.1"}, {"mono_version": "5.2.9"},
    {"geist_pin": "^5.3.0"}, {"mono_pin": "5.3.1"}, {"geist_pin": "~5.3.0"}, {"installed": False},
], ids=lambda v: next(iter(v)) + "=" + str(next(iter(v.values()))))
def test_embed_fonts_refuses_any_pin_or_installed_version_other_than_5_3_0(tmp_path, variant):
    embed = _load(EMBED, "embed_fonts")
    page = _page(tmp_path)
    with pytest.raises(embed.EmbedError):
        embed.embed(_website(tmp_path, **variant), page)
    assert page.read_text(encoding="utf-8") == PAGE


@pytest.mark.parametrize("outside", [
    "<link rel='stylesheet' href='https://fonts.example/a.css'>",
    "<img src='//cdn.example/x.png'>",
    "<style>@import url(http://x.example/a.css);</style>",
    "<div style=\"background:url(https://x.example/i.png)\"></div>",
], ids=["https-link", "protocol-relative", "http-import", "css-url"])
def test_embed_fonts_refuses_a_page_that_names_an_outside_address(tmp_path, outside):
    embed = _load(EMBED, "embed_fonts")
    text = PAGE.replace("</head>", outside + "</head>")
    page = _page(tmp_path, text)
    with pytest.raises(embed.EmbedError):
        embed.embed(_website(tmp_path), page)
    assert page.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("outside", [
    "<img src=\"https:&#47;&#47;example.invalid/private\">",
    "<img src=\"https&colon;&sol;&sol;example.invalid/private\">",
    "<img src=\"&#47;&#47;example.invalid/private\">",
    "<img src=\"https:example.invalid/private\">",
    "<img src=\"HTTP:example.invalid/private\">",
    "<div style=\"background:url(\\68ttps://example.invalid/p)\"></div>",
    "<style>@import url(h\\74 tps://example.invalid/a.css);</style>",
    "<style>@import url(https\\3a //example.invalid/a.css);</style>",
    "<img src=\"https:\\\\example.invalid/private\">",
    "<img src=\"https:/\\example.invalid/private\">",
    "<img src=\"\\\\example.invalid/private\">",
    "<style>@import url(https:\\5c \\5c example.invalid/a.css);</style>",
    "<img src=\"ht&#9;tps://example.invalid/private\">",
    "<img src=\"ht\ntps:\n//example.invalid/private\">",
    "<img src=\"https:&Tab;&NewLine;//example.invalid/private\">",
], ids=["entity-slashes", "named-entities", "entity-protocol-relative", "no-slashes", "no-slashes-upper",
        "css-escaped-scheme", "css-escaped-scheme-space", "css-escaped-colon", "backslashes", "slash-backslash",
        "backslash-protocol-relative", "css-escaped-backslashes", "entity-tab", "raw-newlines", "named-tab-newline"])
def test_embed_fonts_refuses_an_outside_address_written_in_another_form(tmp_path, outside):
    embed = _load(EMBED, "embed_fonts")
    text = PAGE.replace("</head>", outside + "</head>")
    page = _page(tmp_path, text)
    with pytest.raises(embed.EmbedError):
        embed.embed(_website(tmp_path), page)
    assert page.read_bytes() == text.encode("utf-8")


def test_embed_fonts_accepts_inline_svg_namespaces_and_relative_or_data_addresses(tmp_path):
    embed = _load(EMBED, "embed_fonts")
    body = ("<svg xmlns='http://www.w3.org/2000/svg'><use xlink:href='#a'/></svg><a href='/privacy'>p</a>"
            "<img src='data:image/png;base64,AAAA'><a href='docs/https:x'>d</a><img src='img/a&#47;b.png'>"
            "<svg xmlns=\"http&#58;//www.w3.org/2000/svg\"></svg><p>read https and http</p>")
    page = _page(tmp_path, PAGE.replace("</body>", body + "</body>"))
    embed.embed(_website(tmp_path), page)
    assert "@font-face" in page.read_text(encoding="utf-8")


def test_embed_fonts_command_line_exits_non_zero_on_a_refusal(tmp_path):
    assert EMBED.is_file(), "embed_fonts.py is missing"
    page = _page(tmp_path)
    done = subprocess.run([sys.executable, str(EMBED), "--website", str(_website(tmp_path, geist_version="5.3.1")),
                           "--page", str(page)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 1
    assert "refused" in done.stderr
    assert page.read_text(encoding="utf-8") == PAGE


# ---------------------------------------------------------------------------
# check_full_page.py
# ---------------------------------------------------------------------------

COLOURS = ("color", "background-color", "border-top-color", "border-right-color", "border-bottom-color",
           "border-left-color", "text-decoration-color", "fill", "stroke")


def _element(text, *, font="Geist", face="Geist 400", size=16.0, y=0.0):
    style = {"font-family": font, "font-face": face, "font-size": size, "font-weight": 400.0,
             "line-height": 24.0, **{c: None for c in COLOURS}}
    return {"landmarks": [["main", ""]], "tag": "p", "role": "", "label": "", "alt": "", "text": text,
            "parent": 0, "fixed": False, "box": {"x": 0.0, "y": y, "width": 100.0, "height": 24.0}, "style": style}


def _bundle(width, elements, *, blocked=(), page="/privacy"):
    return {"schema": "design_compare_bundle_v1", "kit_sha256": "a" * 64, "chromium_version": "153.0.8010.0",
            "run_id": "design-self-check", "candidate_digest": "none", "page": page, "width": width,
            "page_height": 900, "image": f"privacy-{width}.png", "blocked_requests": list(blocked),
            "masks": [], "elements": elements}


def _write_side(folder: Path, bundles: dict):
    folder.mkdir(parents=True, exist_ok=True)
    for width, bundle in bundles.items():
        (folder / f"privacy-{width}.json").write_text(json.dumps(bundle))
    return folder


def _both_widths(elements, **kw):
    return {w: _bundle(w, elements, **kw) for w in (1280, 375)}


SAME = [_element("Privacy"), _element("We keep little", y=40.0)]
DESIGNED_CHANGE = [_element("Privacy"), _element("We keep very little", y=40.0)]
FONT_DRIFT = [_element("Privacy", font="Arial", face="Arial 400"), _element("We keep little", y=40.0)]


def _kit(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    shutil.copy(CHECK, kit / CHECK.name)
    shutil.copy(JUDGE, kit / JUDGE.name)
    return kit


def _compare(tmp_path, candidate, prototype):
    module = _load(_kit(tmp_path) / CHECK.name, "check_full_page")
    return module, module.compare_folders(
        _write_side(tmp_path / "cand", candidate), _write_side(tmp_path / "proto", prototype))


def test_check_counts_font_findings_and_other_findings_per_width(tmp_path):
    candidate = {1280: _bundle(1280, SAME), 375: _bundle(375, SAME)}
    prototype = {1280: _bundle(1280, FONT_DRIFT), 375: _bundle(375, DESIGNED_CHANGE)}
    _, report = _compare(tmp_path, candidate, prototype)
    wide, narrow = report["widths"]["1280"], report["widths"]["375"]
    assert wide["font_findings"] == 2 and wide["findings"] == 2
    assert narrow["font_findings"] == 0 and narrow["findings"] == 1
    assert wide["by_kind"] == {"type/font-family": 1, "type/font-face": 1}
    assert narrow["by_kind"] == {"element/text": 1}
    assert report["passed"] is False


def test_check_passes_only_with_zero_font_findings_and_zero_blocked_requests(tmp_path):
    _, report = _compare(tmp_path, _both_widths(SAME), _both_widths(DESIGNED_CHANGE))
    assert report["passed"] is True
    assert report["widths"]["1280"]["findings"] == 1
    assert report["widths"]["375"]["blocked_requests"] == 0


def test_check_fails_on_one_font_finding_at_one_width(tmp_path):
    prototype = {1280: _bundle(1280, SAME), 375: _bundle(375, FONT_DRIFT)}
    _, report = _compare(tmp_path, _both_widths(SAME), prototype)
    assert report["widths"]["1280"]["font_findings"] == 0
    assert report["widths"]["375"]["font_findings"] == 2
    assert report["passed"] is False


@pytest.mark.parametrize("side", ["candidate", "prototype"])
def test_check_fails_on_a_blocked_request_even_with_no_findings(tmp_path, side):
    clean, blocked = _both_widths(SAME), {1280: _bundle(1280, SAME, blocked=["http://x.example/a.css"]),
                                          375: _bundle(375, SAME)}
    _, report = _compare(tmp_path, *((blocked, clean) if side == "candidate" else (clean, blocked)))
    assert report["widths"]["1280"]["findings"] == 0
    assert report["widths"]["1280"]["blocked_requests"] == 1
    assert report["passed"] is False


FAKE_CAPTURE = """\
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
const { values: v } = parseArgs({ options: { side: { type: 'string' }, site: { type: 'string' },
  prototype: { type: 'string' }, pages: { type: 'string' }, out: { type: 'string' }, 'run-id': { type: 'string' },
  'candidate-digest': { type: 'string' }, chromium: { type: 'string' } } })
const plan = JSON.parse(readFileSync(process.env.FAKE_BUNDLES, 'utf8'))[v.side]
mkdirSync(v.out, { recursive: true })
for (const [width, bundle] of Object.entries(plan)) writeFileSync(join(v.out, `privacy-${width}.json`), JSON.stringify(bundle))
writeFileSync(join(v.out, 'args.json'), JSON.stringify(process.argv.slice(2)))
"""


def _full_page_with_fonts(tmp_path):
    def face(family, weight, data):
        return (f"@font-face{{font-family:'{family}';font-weight:{weight};"
                f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode()}) format('woff2')}}")
    css = face("Geist", "400 500", GEIST_BYTES) + face("Geist Mono", "400", MONO_BYTES)
    return _page(tmp_path, PAGE.replace("</head>", f"<style>{css}</style></head>"))


def _run_check(tmp_path, candidate, prototype, *, full_page=None, capture=FAKE_CAPTURE, extra_env=None):
    kit = _kit(tmp_path)
    (kit / "capture.mjs").write_text(capture)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"candidate": {str(w): b for w, b in candidate.items()},
                                "prototype": {str(w): b for w, b in prototype.items()}}))
    site = tmp_path / "dist"
    site.mkdir()
    out = tmp_path / "out"
    full_page = full_page or _full_page_with_fonts(tmp_path)
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "FAKE_BUNDLES": str(plan), **(extra_env or {})}
    done = subprocess.run(
        [sys.executable, str(kit / CHECK.name), "--site", str(site), "--full-page", str(full_page),
         "--pages", "/privacy", "--chromium", "/fake/chromium", "--out", str(out)],
        capture_output=True, text=True, timeout=120, env=env)
    return done, out, full_page, site


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_check_command_line_captures_both_sides_and_writes_self_check_json(tmp_path):
    done, out, full_page, site = _run_check(tmp_path, _both_widths(SAME), _both_widths(DESIGNED_CHANGE))
    assert done.returncode == 0, done.stderr
    report = json.loads((out / "self-check.json").read_text())
    assert report["passed"] is True
    assert report["full_page"] == {"file": "full-page.html", "sha256": _sha(full_page.read_bytes())}
    assert report["fonts"] == [{"family": "Geist", "sha256": _sha(GEIST_BYTES)},
                               {"family": "Geist Mono", "sha256": _sha(MONO_BYTES)}]
    assert report["kit_sha256"] == "a" * 64 and report["chromium_version"] == "153.0.8010.0"
    assert set(report["widths"]) == {"1280", "375"}
    sides = {side: json.loads(args.read_text()) for side, args in
             (("candidate", out / "candidate" / "args.json"), ("prototype", out / "full-page" / "args.json"))}
    for side, args in sides.items():
        assert args[args.index("--side") + 1] == side
        assert args[args.index("--pages") + 1] == "/privacy"
        assert args[args.index("--run-id") + 1] == "design-self-check"
        assert args[args.index("--candidate-digest") + 1] == "none"
        assert args[args.index("--chromium") + 1] == "/fake/chromium"
    assert sides["candidate"][sides["candidate"].index("--site") + 1] == str(site)
    assert sides["prototype"][sides["prototype"].index("--prototype") + 1] == str(full_page)
    assert "self-check passed: 0 font findings at 1280 px and 375 px" in done.stdout
    assert "self-check.json" in done.stdout
    assert PAGE_TEXT not in done.stdout + done.stderr and "We keep" not in done.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_check_command_line_fails_on_font_findings_and_still_writes_the_report(tmp_path):
    prototype = {1280: _bundle(1280, FONT_DRIFT), 375: _bundle(375, SAME)}
    done, out, _, _ = _run_check(tmp_path, _both_widths(SAME), prototype)
    assert done.returncode == 1
    assert json.loads((out / "self-check.json").read_text())["passed"] is False
    assert "self-check failed" in done.stdout
    assert "Arial" not in done.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_check_command_line_refuses_a_page_without_the_two_embedded_faces(tmp_path):
    done, out, _, _ = _run_check(tmp_path, _both_widths(SAME), _both_widths(SAME), full_page=_page(tmp_path))
    assert done.returncode == 1
    assert "refused" in done.stderr
    assert not (out / "self-check.json").exists()


FAILING_CAPTURE = """\
const side = process.argv[process.argv.indexOf('--side') + 1]
if (side === process.env.FAIL_SIDE) {
  process.stderr.write('requested: https://example.invalid/PRIVATE_PAGE_TEXT_4931\\n')
  process.stdout.write('page text PRIVATE_PAGE_TEXT_4931\\n')
  process.exit(3)
}
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("side", ["candidate", "prototype"])
def test_check_command_line_refuses_a_failed_capture_without_printing_what_the_kit_wrote(tmp_path, side):
    done, out, _, _ = _run_check(tmp_path, {}, {}, capture=FAILING_CAPTURE, extra_env={"FAIL_SIDE": side})
    assert done.returncode == 1
    assert done.stderr.strip() == f"refused: the {side} capture failed (exit 3)"
    assert "PRIVATE_PAGE_TEXT_4931" not in done.stdout + done.stderr
    assert "example.invalid" not in done.stdout + done.stderr
    assert not (out / "self-check.json").exists()


# ---------------------------------------------------------------------------
# SKILL.md
# ---------------------------------------------------------------------------


def _skill_text() -> str:
    assert SKILL_MD.is_file(), "SKILL.md is missing"
    return SKILL_MD.read_text(encoding="utf-8")


def _section(name: str, until: str) -> str:
    text = _skill_text()
    return text[text.index(f"## {name}"):text.index(f"## {until}")]


def _frontmatter() -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", _skill_text(), re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_description_is_at_most_sixty_characters_and_ends_with_a_period():
    description = _frontmatter()["description"]
    assert len(description) <= 60
    assert description.endswith(".")


def test_name_matches_the_directory():
    assert _frontmatter()["name"] == "design-offer"


def test_modern_section_order():
    headings = re.findall(r"^## (.+)$", _skill_text(), re.M)
    wanted = ["When to Use", "Prerequisites", "How to Run", "Quick Reference", "Procedure",
              "Pitfalls", "Verification"]
    positions = [headings.index(h) for h in wanted]
    assert positions == sorted(positions)
    assert "Return to Claude Design" in headings


def test_standing_rules_stay_in_the_skill():
    text = _skill_text()
    for needle in ("PNG, JPEG or WebP", "1280 px", "375 px", "two or three", "kanban_offer_design_options",
                   "waits for the owner's choice"):
        assert needle in text, needle
    summary = re.search(r"After the owner's choice[^\n]*", text)
    assert summary, "the summary rule is missing"
    for needle in ("number of the option", "title", "attachment of its picture"):
        assert needle in summary.group(0), needle
    assert re.search(r"one picture (for each|per) option", text, re.I)


def test_pillow_is_not_a_prerequisite_and_designer_kit_is():
    assert not re.search(r"pillow|\bPIL\b", _skill_text(), re.I)
    prerequisites = _section("Prerequisites", "How to Run")
    assert "design_compare" in prerequisites and "designer_kit" in prerequisites


def test_procedure_names_the_tools_the_plan_requires_in_order():
    text = _section("Procedure", "Pitfalls")
    order = ["option-<n>.html", "kanban_attach", "raphael_sandbox_provision", "designer_kit",
             "raphael_sandbox_artifact", "CHROMIUM_PIN.package", "scripts/design_compare/capture.mjs",
             "compose_offer.mjs", "kanban_view_image", "kanban_offer_design_options"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
    for needle in ("--side prototype", "--pages /", "--run-id design-offer", "--candidate-digest none",
                   "--chromium", "inspect", "import", "export", "no credential vault",
                   "no credential proxy", "empty ownership scope"):
        assert needle in text, needle


def test_procedure_names_the_kit_scripts_and_website_steps_in_order():
    text = _section("Procedure", "Pitfalls")
    order = ["designer_kit", "compose_offer.mjs", "--kit", "embed_fonts.py", "check_full_page.py",
             "npm ci", "npm run build", "1280 px", "375 px"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions), dict(zip(order, positions))
    assert "SHA-256" in text[text.index("designer_kit"):text.index("compose_offer.mjs")]


def test_after_the_choice_commands_run_in_the_order_of_the_plan():
    text = _section("Procedure", "Pitfalls")
    order = ["npm ci", "embed_fonts.py --website", "npm run build", "check_full_page.py --site", "design_pin"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions), dict(zip(order, positions))


def test_design_pin_record_and_its_four_fields_are_named_exactly():
    text = _skill_text()
    anchor = re.search(r"^.*`design_pin`.*exactly four fields.*$", text, re.M)
    assert anchor, "the skill must say that design_pin has exactly four fields"
    fields = re.findall(r"^\s+- `(\w+)`:", text[anchor.end():].split("\n\n")[0], re.M)
    assert fields == ["attachment_id", "sha256", "file", "pages"]
    block = text[anchor.end():].split("\n\n")[0]
    assert "full-page.html" in block and "64 hex characters" in block and "list" in block
    assert re.search(r"summary[^\n]*is not enough|not enough[^\n]*summary", text, re.I)
    assert re.search(r"After the owner's choice[^\n]*completion metadata holds `design_pin`", text)


def test_rules_forbid_foreign_pages_readbacks_and_outside_services():
    text = _skill_text().lower()
    assert "attached yourself in this run" in text
    assert "never read" in text and "bundle" in text
    assert "never use the design browser" in text
    assert "outside design service" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_skill_names_the_real_capture_kit_pin_field():
    # Import the real module in one Node process (no browser is launched) and
    # read the exported pin, instead of matching the kit's source text.
    done = subprocess.run(
        ["node", "--input-type=module", "-e",
         "const m = await import(process.argv[1]);"
         "console.log(JSON.stringify(m.CHROMIUM_PIN));",
         CAPTURE_MJS.as_uri()],
        capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    pin = json.loads(done.stdout)
    assert isinstance(pin.get("package"), str) and pin["package"]
    assert "CHROMIUM_PIN.package" in _skill_text()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
