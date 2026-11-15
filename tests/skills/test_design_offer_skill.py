"""Tests for optional-skills/creative/design-offer: the designer renders its own
option pages into one picture per option (1280 px and 375 px side by side).

stdlib + pytest + Pillow, no network, no browser: the capture kit's output is
faked with plain pictures and bundle files of the kit's own shape.
"""

import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "creative" / "design-offer"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT = SKILL_DIR / "scripts" / "compose_offer.py"
CAPTURE_MJS = REPO / "scripts" / "design_compare" / "capture.mjs"

FOUR_MIB = 4 * 1024 * 1024
KIT = "a" * 64
CHROMIUM = "153.0.8010.0"
LEFT_COLOR = (200, 30, 30)
RIGHT_COLOR = (30, 30, 200)


def _load():
    assert SCRIPT.is_file(), "compose_offer.py is missing"
    spec = importlib.util.spec_from_file_location("compose_offer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_capture(folder: Path, *, left_height=900, right_height=1400, kit=KIT,
                  chromium=CHROMIUM, blocked=(), noisy=False, slug="index-0a1b2c3d"):
    """A capture folder as the kit writes it for one page at both widths."""
    folder.mkdir(parents=True, exist_ok=True)
    for width, height, color in ((1280, left_height, LEFT_COLOR), (375, right_height, RIGHT_COLOR)):
        if noisy:
            rng = random.Random(width)
            image = Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3))
        else:
            image = Image.new("RGB", (width, height), color)
        image.save(folder / f"{slug}-{width}.png")
        bundle = {
            "schema": "design_compare_bundle_v1", "kit_sha256": kit, "chromium_version": chromium,
            "run_id": "design-offer", "candidate_digest": "none", "page": "/", "width": width,
            "page_height": height, "image": f"{slug}-{width}.png", "blocked_requests": list(blocked),
            "masks": [], "elements": [],
        }
        (folder / f"{slug}-{width}.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return folder


def _page(tmp_path: Path, name: str) -> Path:
    page = tmp_path / "pages" / name
    page.parent.mkdir(exist_ok=True)
    page.write_text(f"<!doctype html><title>{name}</title><p>option page</p>")
    return page


def _options(tmp_path: Path, count: int, **kw):
    return [
        (_page(tmp_path, f"option-{n}.html"), _fake_capture(tmp_path / f"cap-{n}", **kw))
        for n in range(1, count + 1)
    ]


def _color_box(image: Image.Image, color):
    r, g, b = image.convert("RGB").split()
    masks = [
        ch.point(lambda v, c=c: 255 if abs(v - c) < 8 else 0) for ch, c in zip((r, g, b), color)
    ]
    return ImageChops.multiply(ImageChops.multiply(masks[0], masks[1]), masks[2]).getbbox()


@pytest.fixture(scope="module")
def mod():
    return _load()


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [2, 3])
def test_options_compose_into_one_picture_each_with_both_widths(mod, tmp_path, count):
    out = tmp_path / "out"
    mod.compose(_options(tmp_path, count), out)

    for n in range(1, count + 1):
        pictures = [p for p in out.glob(f"option-{n}-1280-375.*")]
        assert len(pictures) == 1
        picture = pictures[0]
        assert picture.suffix in (".png", ".jpg")
        assert 0 < picture.stat().st_size <= FOUR_MIB
        image = Image.open(picture)
        left = _color_box(image, LEFT_COLOR)
        right = _color_box(image, RIGHT_COLOR)
        assert left and right
        assert (left[2] - left[0], left[3] - left[1]) == (1280, 900)
        assert (right[2] - right[0], right[3] - right[1]) == (375, 1400)
        assert left[2] <= right[0], "1280 px picture sits left of the 375 px picture"
        assert left[1] == right[1], "both start at the same height"
        assert left[1] > 0, "room above the pictures for the labels"
        for box in (left, right):
            header = image.convert("RGB").crop((box[0], 0, box[2], box[1]))
            background = image.convert("RGB").getpixel((0, 0))
            assert ImageChops.difference(header, Image.new("RGB", header.size, background)).getbbox(), \
                "a label sits above each picture"
    assert (out / "design-offer-manifest.json").is_file()
    assert len(list(out.glob("option-*-1280-375.*"))) == count


def test_manifest_hashes_equal_independent_hashes(mod, tmp_path):
    options = _options(tmp_path, 2)
    out = tmp_path / "out"
    mod.compose(options, out)

    manifest = json.loads((out / "design-offer-manifest.json").read_text())
    assert manifest["kit_sha256"] == KIT
    assert manifest["chromium_version"] == CHROMIUM
    assert len(manifest["options"]) == 2
    for entry, (page, folder) in zip(manifest["options"], options):
        n = entry["n"]
        assert entry["page"] == {"file": f"option-{n}.html", "sha256": _sha(page)}
        captures = {c["width"]: c for c in entry["captures"]}
        assert set(captures) == {1280, 375}
        for width, capture in captures.items():
            assert capture["sha256"] == _sha(folder / capture["file"])
            assert capture["file"].endswith(f"-{width}.png")
        picture = out / entry["picture"]["file"]
        assert entry["picture"]["sha256"] == _sha(picture)
        assert entry["picture"]["bytes"] == picture.stat().st_size


def test_a_very_tall_page_still_lands_under_four_mib(mod, tmp_path):
    options = [
        (_page(tmp_path, "option-1.html"),
         _fake_capture(tmp_path / "cap-1", left_height=8000, right_height=8000, noisy=True)),
        (_page(tmp_path, "option-2.html"), _fake_capture(tmp_path / "cap-2")),
    ]
    out = tmp_path / "out"
    mod.compose(options, out)

    manifest = json.loads((out / "design-offer-manifest.json").read_text())
    picture = out / manifest["options"][0]["picture"]["file"]
    assert 0 < picture.stat().st_size <= FOUR_MIB
    assert Image.open(picture).format in ("PNG", "JPEG")
    assert manifest["options"][0]["picture"]["sha256"] == _sha(picture)


def test_pictures_pass_the_offer_tools_own_picture_rules(mod, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import owner_design_choices as odc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB",
                "HERMES_KANBAN_BOARD", "HERMES_KANBAN_ATTACHMENTS_ROOT"):
        monkeypatch.delenv(key, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        card = kb.create_task(conn, title="privacy page heading", assignee="designer")
        out = tmp_path / "out"
        mod.compose(_options(tmp_path, 2), out)
        manifest = json.loads((out / "design-offer-manifest.json").read_text())
        for n, entry in enumerate(manifest["options"], start=1):
            picture = out / entry["picture"]["file"]
            media = "image/png" if picture.suffix == ".png" else "image/jpeg"
            attachment = kb.store_attachment_bytes(
                conn, card, picture.name, picture.read_bytes(), content_type=media,
                uploaded_by="agent", board=kb.DEFAULT_BOARD,
            )
            stored = odc._picture(
                conn, card, n,
                {"attachment_id": str(attachment), "alt": "The page at two widths."},
                board=kb.DEFAULT_BOARD,
            )
            assert stored["media_type"] == media
            assert stored["bytes"] <= odc.MAX_PICTURE_BYTES
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _refused(mod, options, out):
    with pytest.raises(mod.OfferError):
        mod.compose(options, out)
    assert not out.exists() or not any(out.iterdir()), "a refusal writes nothing"


@pytest.mark.parametrize("count", [0, 1, 4])
def test_refuses_zero_one_or_four_options(mod, tmp_path, count):
    options = _options(tmp_path, min(count, 3))
    if count == 4:
        options.append((_page(tmp_path, "option-4.html"), _fake_capture(tmp_path / "cap-4")))
    _refused(mod, options, tmp_path / "out")


@pytest.mark.parametrize("bad", [
    "page-1.html", "option-4.html", "option-0.html", "option-1.htm", "Option-1.html",
    "option-01.html", "option-1.html.txt", "option-a.html",
])
def test_refuses_a_page_not_named_option_n_html(mod, tmp_path, bad):
    options = _options(tmp_path, 2)
    options[0] = (_page(tmp_path, bad), options[0][1])
    _refused(mod, options, tmp_path / "out")


def test_refuses_duplicate_or_gapped_option_numbers(mod, tmp_path):
    options = _options(tmp_path, 2)
    options[1] = (_page(tmp_path, "option-3.html"), options[1][1])
    _refused(mod, options, tmp_path / "gap")
    options = _options(tmp_path, 2)
    options[1] = (options[0][0], options[1][1])
    _refused(mod, options, tmp_path / "dup")


def test_refuses_a_missing_page_file(mod, tmp_path):
    options = _options(tmp_path, 2)
    options[0][0].unlink()
    _refused(mod, options, tmp_path / "out")


def test_refuses_bundles_with_different_kit_hashes(mod, tmp_path):
    options = _options(tmp_path, 2)
    options[1] = (options[1][0], _fake_capture(tmp_path / "other", kit="b" * 64))
    _refused(mod, options, tmp_path / "out")


def test_refuses_bundles_with_different_chromium_versions(mod, tmp_path):
    options = _options(tmp_path, 2)
    options[1] = (options[1][0], _fake_capture(tmp_path / "other", chromium="152.0.0.0"))
    _refused(mod, options, tmp_path / "out")


def test_refuses_two_widths_of_one_page_that_disagree(mod, tmp_path):
    options = _options(tmp_path, 2)
    path = next(options[0][1].glob("*-375.json"))
    bundle = json.loads(path.read_text())
    bundle["kit_sha256"] = "c" * 64
    path.write_text(json.dumps(bundle))
    _refused(mod, options, tmp_path / "out")


def test_refuses_a_bundle_with_a_blocked_request(mod, tmp_path):
    options = _options(tmp_path, 2)
    options[0] = (options[0][0], _fake_capture(tmp_path / "blocked", blocked=["http://example.test/x"]))
    _refused(mod, options, tmp_path / "out")


def test_refuses_a_picture_that_does_not_match_its_bundle_width(mod, tmp_path):
    options = _options(tmp_path, 2)
    path = next(options[0][1].glob("*-375.png"))
    Image.new("RGB", (400, 1400), RIGHT_COLOR).save(path)
    _refused(mod, options, tmp_path / "out")


def test_refuses_a_bundle_naming_a_file_outside_its_folder(mod, tmp_path):
    options = _options(tmp_path, 2)
    path = next(options[0][1].glob("*-375.json"))
    bundle = json.loads(path.read_text())
    bundle["image"] = "../cap-2/" + next(options[1][1].glob("*-375.png")).name
    path.write_text(json.dumps(bundle))
    _refused(mod, options, tmp_path / "out")


def test_refuses_to_overwrite_an_earlier_result(mod, tmp_path):
    options = _options(tmp_path, 2)
    out = tmp_path / "out"
    mod.compose(options, out)
    with pytest.raises(mod.OfferError):
        mod.compose(options, out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_command_line_composes_and_prints_only_file_names(mod, tmp_path):
    options = _options(tmp_path, 2)
    out = tmp_path / "out"
    args = [sys.executable, str(SCRIPT), "--out", str(out)]
    for page, folder in options:
        args += ["--option", str(page), str(folder)]
    done = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert (out / "design-offer-manifest.json").is_file()
    assert "option page" not in done.stdout


def test_command_line_exits_non_zero_on_a_refusal(tmp_path):
    assert SCRIPT.is_file(), "compose_offer.py is missing"
    options = _options(tmp_path, 1)
    out = tmp_path / "out"
    args = [sys.executable, str(SCRIPT), "--out", str(out)]
    for page, folder in options:
        args += ["--option", str(page), str(folder)]
    done = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1
    assert "refused" in done.stderr
    assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------------------
# SKILL.md
# ---------------------------------------------------------------------------


def _skill_text() -> str:
    assert SKILL_MD.is_file(), "SKILL.md is missing"
    return SKILL_MD.read_text(encoding="utf-8")


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


def test_procedure_names_the_tools_the_plan_requires_in_order():
    text = _skill_text()
    text = text[text.index("## Procedure"):text.index("## Pitfalls")]
    order = ["option-<n>.html", "kanban_attach", "raphael_sandbox_provision", "raphael_sandbox_artifact",
             "CHROMIUM_PIN.package", "scripts/design_compare/capture.mjs",
             "optional-skills/creative/design-offer/scripts/compose_offer.py",
             "kanban_view_image", "kanban_offer_design_options"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
    for needle in ("--side prototype", "--pages /", "--run-id design-offer", "--candidate-digest none",
                   "--chromium", "inspect", "import", "export", "no credential vault",
                   "no credential proxy", "empty ownership scope"):
        assert needle in text, needle


def test_rules_forbid_foreign_pages_readbacks_and_outside_services():
    text = _skill_text().lower()
    assert "attached yourself in this run" in text
    assert "never read" in text and "bundle" in text
    assert "never use the design browser" in text
    assert "outside design service" in text


def test_skill_names_the_real_capture_kit_pin_field():
    assert re.search(r"CHROMIUM_PIN\s*=\s*\{\s*package:", CAPTURE_MJS.read_text())
    assert "CHROMIUM_PIN.package" in _skill_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
