#!/usr/bin/env python3
"""Check a full-page design file against the built website at 1280 px and 375 px.

    check_full_page.py --site DIST --full-page full-page.html --pages /privacy --chromium BINARY --out DIR

Run it from the kit folder: it uses ``capture.mjs`` and ``design_compare_judge.py`` that sit next
to it. The kit captures the built website (candidate side) and the full-page file (prototype
side) for the same routes at both widths; the judge compares each pair. The check passes only
with zero blocked requests and zero font-family and font-face findings at both widths; other
findings are the designed change and are counted. ``DIR/self-check.json`` records the file's
SHA-256, the SHA-256 of the two embedded fonts, the kit hash, the Chromium version and the
counts per width.

Standard library plus the judge. Prints counts and file names only, never page text.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WIDTHS = (1280, 375)
FONT_PROPERTIES = ("font-family", "font-face")
FONT_FACE = re.compile(r"@font-face\s*\{([^}]*)\}", re.IGNORECASE)
FAMILY = re.compile(r"font-family\s*:\s*(['\"]?)([^'\";]+?)\1\s*;", re.IGNORECASE)
DATA_URL = re.compile(r"url\(\s*['\"]?data:font/woff2;base64,([A-Za-z0-9+/=]+)['\"]?\s*\)", re.IGNORECASE)
EMBEDDED_FAMILIES = ("Geist", "Geist Mono")
REPORT = "self-check.json"
RUN_ID = "design-self-check"
CAPTURE_SECONDS = 600


class CheckError(Exception):
    """The check cannot run; nothing was reported."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def embedded_fonts(page_text: str) -> list[dict]:
    """The SHA-256 of the font bytes in the page's data: faces named 'Geist' and 'Geist Mono'."""
    found = {}
    for block in FONT_FACE.findall(page_text):
        family, url = FAMILY.search(block), DATA_URL.search(block)
        if family and url and family.group(2) in EMBEDDED_FAMILIES:
            found[family.group(2)] = _sha256(base64.b64decode(url.group(1)))
    missing = [f for f in EMBEDDED_FAMILIES if f not in found]
    if missing:
        raise CheckError(f"the full-page file has no embedded data: face for {', '.join(missing)}")
    return [{"family": family, "sha256": found[family]} for family in EMBEDDED_FAMILIES]


def _read_side(folder: Path) -> dict:
    bundles = {}
    for width in WIDTHS:
        for path in sorted(folder.glob(f"*-{width}.json")):
            try:
                bundle = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise CheckError(f"{folder.name}/{path.name} cannot be read") from None
            if not isinstance(bundle, dict) or bundle.get("width") != width or not isinstance(bundle.get("page"), str):
                raise CheckError(f"{folder.name}/{path.name} is not a capture bundle of {width} px")
            if (bundle["page"], width) in bundles:
                raise CheckError(f"{folder.name} holds two {width} px bundles for {bundle['page']}")
            bundles[bundle["page"], width] = bundle
    return bundles


def compare_folders(candidate_dir, prototype_dir) -> dict:
    """Compare the bundles of two capture folders with the judge; count findings per width."""
    import design_compare_judge as judge

    candidate, prototype = _read_side(Path(candidate_dir)), _read_side(Path(prototype_dir))
    if not candidate or set(candidate) != set(prototype):
        raise CheckError("the two sides captured different pages or widths")
    widths = {str(w): {"findings": 0, "font_findings": 0, "blocked_requests": 0, "by_kind": Counter()} for w in WIDTHS}
    kits, versions = set(), set()
    for key in sorted(candidate):
        cand, proto = candidate[key], prototype[key]
        row = widths[str(key[1])]
        try:
            comparison = judge.compare_bundles(cand, proto)
        except judge.BundleError as exc:
            raise CheckError(f"a bundle pair cannot be compared: {exc}") from None
        for finding in comparison.findings:
            row["findings"] += 1
            row["by_kind"][f"{finding.kind}/{finding.property}"] += 1
            if finding.property in FONT_PROPERTIES:
                row["font_findings"] += 1
        for bundle in (cand, proto):
            row["blocked_requests"] += len(bundle.get("blocked_requests") or [])
            kits.add(bundle.get("kit_sha256"))
            versions.add(bundle.get("chromium_version"))
    for row in widths.values():
        row["by_kind"] = dict(row["by_kind"])
    return {
        "kit_sha256": kits.pop() if len(kits) == 1 else None,
        "chromium_version": versions.pop() if len(versions) == 1 else None,
        "pages": sorted({page for page, _ in candidate}),
        "widths": widths,
        "passed": all(r["font_findings"] == 0 and r["blocked_requests"] == 0 for r in widths.values()),
    }


def _capture(side: str, folder: Path, extra: list[str], pages: str, chromium: str) -> None:
    command = ["node", str(HERE / "capture.mjs"), "--side", side, *extra, "--pages", pages, "--out", str(folder),
               "--run-id", RUN_ID, "--candidate-digest", "none", "--chromium", chromium]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=CAPTURE_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckError(f"the {side} capture could not run: {type(exc).__name__}") from None
    if done.returncode != 0:
        raise CheckError(f"the {side} capture failed (exit {done.returncode})")


def run(site, full_page, pages, chromium, out) -> dict:
    site, full_page, out = Path(site), Path(full_page), Path(out)
    routes = [p for p in re.split(r"[\s,]+", pages) if p]
    if not routes or not all(p.startswith("/") for p in routes):
        raise CheckError("pages must be routes that start with /")
    if not site.is_dir():
        raise CheckError("the built website folder is missing")
    if full_page.is_symlink() or not full_page.is_file():
        raise CheckError("the full-page file is missing")
    data = full_page.read_bytes()
    fonts = embedded_fonts(data.decode("utf-8", errors="replace"))
    if (out / REPORT).exists():
        raise CheckError(f"refusing to overwrite {out / REPORT}")
    route_text = " ".join(routes)
    _capture("candidate", out / "candidate", ["--site", str(site)], route_text, chromium)
    _capture("prototype", out / "full-page", ["--prototype", str(full_page)], route_text, chromium)
    outside = sum(len(bundle.get("blocked_requests") or []) for bundle in _read_side(out / "full-page").values())
    if outside:
        raise CheckError(f"the full-page file asked for {outside} request(s) beyond the page file")
    report = compare_folders(out / "candidate", out / "full-page")
    report = {"full_page": {"file": full_page.name, "sha256": _sha256(data)}, "fonts": fonts, **report}
    with open(out / REPORT, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True, help="the built website folder (dist)")
    parser.add_argument("--full-page", required=True, help="the full-page file, fonts embedded")
    parser.add_argument("--pages", required=True, help="the route(s) to compare, for example /privacy")
    parser.add_argument("--chromium", required=True, help="the pinned Chromium binary")
    parser.add_argument("--out", required=True, help="folder for the captures and self-check.json")
    args = parser.parse_args(argv)
    try:
        report = run(args.site, args.full_page, args.pages, args.chromium, args.out)
    except CheckError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    others = 0
    for width in WIDTHS:
        row = report["widths"][str(width)]
        others += row["findings"] - row["font_findings"]
        print(f"{width} px: {row['findings']} findings, {row['font_findings']} font findings, "
              f"{row['blocked_requests']} blocked requests")
    if report["passed"]:
        print(f"self-check passed: 0 font findings at {WIDTHS[0]} px and {WIDTHS[1]} px; "
              f"{others} other findings are the designed change")
    else:
        print("self-check failed: fix the full-page file (fonts, outside requests) and run it again")
    print(REPORT)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
