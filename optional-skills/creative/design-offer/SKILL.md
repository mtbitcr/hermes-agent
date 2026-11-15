---
name: design-offer
description: Render option pages into pictures for a design offer.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, offer, screenshots, options, owner-choice, prototype, html]
    related_skills: [sketch]
---

# Design Offer Skill

The designer writes each design option as one self-contained web page, renders the pages in its own short-lived build machine with the capture kit already in the repository, and offers one picture per option so the owner chooses with one tap. It needs no outside design service and no credential. It does not build the chosen design; that is a later card.

## When to Use

- The owner needs to choose between two or three designs and wants to see them as pictures.
- The card is the designer's own, and the choice is held for the owner.
- Not for a quick drawing: use the SVG drawing tool for sketches. Not for looking at live sites: use the design browser for that.

## Prerequisites

- `kanban_attach`, `raphael_sandbox_provision`, `raphael_sandbox_artifact`, the build machine's command tool, `kanban_view_image` and `kanban_offer_design_options`.
- A build machine provisioned with no credential vault, no credential proxy, an empty ownership scope and default-deny outbound traffic, plus the package registry allowance the review runs already have.
- Pillow in the machine's Python (already a dependency of the repository).

## How to Run

Follow the Procedure in order. The standing rules of an offer:

- One picture for each option, of type PNG, JPEG or WebP, at most 4 MiB.
- Each option shows at 1280 px and at 375 px wide, side by side in that one picture.
- Two or three options per offer, offered with the offer tool; the card then waits for the owner's choice.
- After the owner's choice, the summary names the number of the option, its title and the attachment of its picture.

## Quick Reference

| Step | Tool or file |
|---|---|
| Page files | `option-<n>.html`, n from 1 to 3 |
| Render | `scripts/design_compare/capture.mjs`, prototype side, route `/` |
| Compose | `scripts/compose_offer.py` of this skill |
| Outputs | `option-<n>-1280-375.png` (or `.jpg`) and `design-offer-manifest.json` |

## Procedure

1. Write each option as one self-contained HTML file named `option-<n>.html`: inline styles, system or embedded fonts, no outside address, no animation.
2. Attach each file with `kanban_attach` to your own card.
3. Call `raphael_sandbox_provision`. Check the receipt shows no credential vault, no credential proxy and an empty ownership scope. Stop and say so if it does not.
4. For each page, call `raphael_sandbox_artifact` with `inspect`, then with `import` and its SHA-256.
5. In the machine, install the pinned Chromium package named by `CHROMIUM_PIN.package` in `scripts/design_compare/capture.mjs`, unpack its browser binary (the package's `executablePath()` does this), then for each page run `node scripts/design_compare/capture.mjs --side prototype --prototype <page> --pages / --out <folder>/option-<n> --run-id design-offer --candidate-digest none --chromium <pinned binary>`. The kit refuses any other binary and any page that asks for something outside its own file. If the package cannot be installed under the machine's outbound policy, stop and report it; do not ask for wider access.
6. Run `python optional-skills/creative/design-offer/scripts/compose_offer.py --out <folder>/offer --option <folder>/option-1.html <folder>/option-1 --option <folder>/option-2.html <folder>/option-2` (a third `--option` for a third page). It refuses fewer than two or more than three options, page names other than `option-<n>.html`, bundles from different kit or browser versions, and any capture that blocked a request.
7. Call `raphael_sandbox_artifact` with `export` for each combined picture and for `design-offer-manifest.json`.
8. Look at each picture with `kanban_view_image`.
9. Call `kanban_offer_design_options` with one picture for each option: one plain sentence per option and the attachment of its picture. The card then waits for the owner's choice.
10. Rules: render only pages you attached yourself in this run. Never read bundle JSON or page text back into the conversation. Never use the design browser or any outside design service for this.

## Pitfalls

- A page that loads a font, script, image or style from an outside address fails the capture. Put everything inline.
- A page that moves or fades gives unstable pictures; leave animation out.
- A very tall page makes a large picture; the helper shrinks it evenly and may switch to JPEG to stay under 4 MiB.
- The helper never overwrites: use a fresh output folder for a second attempt.

## Verification

- The provision receipt shows no credential vault, no credential proxy and an empty ownership scope.
- `design-offer-manifest.json` lists, for each option, the SHA-256 of its page file, both captures and its picture, and one kit hash and one Chromium version.
- Each picture shows the 1280 px page on the left and the 375 px page on the right, with both labels.
- The card is held for the owner's choice and nothing is built.

## Return to Claude Design

The Claude Design connection is switched off, not deleted. To return, switch the `mcp_servers.claude-design.enabled` setting of the designer back on and follow the old Claude Design procedure; this skill stays installed.
