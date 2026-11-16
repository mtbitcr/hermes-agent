---
name: design-offer
description: Render option pages into pictures for a design offer.
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, offer, screenshots, options, owner-choice, prototype, html]
    related_skills: [sketch]
---

# Design Offer Skill

The designer writes each design option as one self-contained web page, renders the pages in its own short-lived build machine with the capture kit it receives on its card, and offers one picture per option so the owner chooses with one tap. After the choice it writes one full-page file of the chosen option, checks it against the built website and leaves a pin of that file on its card. It needs no outside design service and no credential. It does not build the chosen design; that is a later card.

## When to Use

- The owner needs to choose between two or three designs and wants to see them as pictures.
- The card is the designer's own, and the choice is held for the owner.
- Not for a quick drawing: use the SVG drawing tool for sketches. Not for looking at live sites: use the design browser for that.

## Prerequisites

- `kanban_attach`, `raphael_sandbox_provision`, `raphael_sandbox_artifact`, the build machine's command tool, `kanban_view_image`, `kanban_offer_design_options` and `design_compare` with the action `designer_kit`.
- A build machine provisioned with no credential vault, no credential proxy, an empty ownership scope and default-deny outbound traffic, plus the package registry allowance the review runs already have. It holds the website tree and needs only Node and the Python standard library.

## How to Run

Follow the Procedure in order. The standing rules of an offer:

- One picture for each option, of type PNG, JPEG or WebP, at most 4 MiB.
- Each option shows at 1280 px and at 375 px wide, side by side in that one picture.
- Two or three options per offer, offered with the offer tool; the card then waits for the owner's choice.
- After the owner's choice, the summary names the number of the option, its title and the attachment of its picture, and the completion metadata holds `design_pin`.

## Quick Reference

| Step | Tool or file |
|---|---|
| Page files | `option-<n>.html`, n from 1 to 3 |
| Kit | `design_compare` action `designer_kit`, every file imported by SHA-256 into one kit folder |
| Render | `<kit>/capture.mjs`, prototype side, route `/` |
| Compose | `<kit>/compose_offer.mjs --kit <kit>` |
| Outputs | `option-<n>-1280-375.png` (or `.jpg`) and `design-offer-manifest.json` |
| After the choice | `<kit>/embed_fonts.py`, `<kit>/check_full_page.py`, `full-page.html`, `self-check.json` |
| Record | `design_pin` in the completion metadata |

## Procedure

1. Write each option as one self-contained HTML file named `option-<n>.html`: inline styles, system or embedded fonts, no outside address, no animation.
2. Attach each file with `kanban_attach` to your own card.
3. Call `raphael_sandbox_provision`. Check the receipt shows no credential vault, no credential proxy and an empty ownership scope. Stop and say so if it does not.
4. On your own card, outside a review run, call `design_compare` with the action `designer_kit`. It stores the capture kit, the judge and the scripts of this skill on your card, each file with its own SHA-256.
5. For each page, and for every file `designer_kit` lists, call `raphael_sandbox_artifact` with `inspect`, then with `import` and its SHA-256. Import the kit files under their own names into one folder, the kit folder `<kit>`.
6. In the machine, install the pinned Chromium package named by `CHROMIUM_PIN.package` in `scripts/design_compare/capture.mjs`, unpack its browser binary (the package's `executablePath()` does this), then for each page run `node <kit>/capture.mjs --side prototype --prototype <page> --pages / --out <folder>/option-<n> --run-id design-offer --candidate-digest none --chromium <pinned binary>`. The kit refuses any other binary and any page that asks for something outside its own file. If the package cannot be installed under the machine's outbound policy, stop and report it; do not ask for wider access.
7. Run `node <kit>/compose_offer.mjs --kit <kit> --chromium <pinned binary> --out <folder>/offer --option <folder>/option-1.html <folder>/option-1 --option <folder>/option-2.html <folder>/option-2` (a third `--option` for a third page). It uses the kit's own browser check, refuses fewer than two or more than three options, page names other than `option-<n>.html`, bundles from different kit or browser versions, and any capture that blocked a request, checks its own pictures after drawing them, and writes nothing when it refuses. The kit folder also holds `embed_fonts.py` and `check_full_page.py`, which the steps after the owner's choice use.
8. Call `raphael_sandbox_artifact` with `export` for each combined picture and for `design-offer-manifest.json`.
9. Look at each picture with `kanban_view_image`.
10. Call `kanban_offer_design_options` with one picture for each option: one plain sentence per option and the attachment of its picture. The card then waits for the owner's choice.
11. Rules: render only pages you attached yourself in this run. Never read bundle JSON or page text back into the conversation. Never use the design browser or any outside design service for this.
12. After the owner's choice, in the website tree of the machine, run `npm ci`. The registry is reachable and npm checks each package against the lock file.
13. Write the chosen option for the compared route as `full-page.html`, then run `python <kit>/embed_fonts.py --website <website> --page full-page.html`. It refuses unless the website pins both font packages and both installed packages are exactly 5.3.0, and it refuses a page that names an outside address. It writes the two font files into the page as data addresses and prints the SHA-256 of each. Never handle the font bytes yourself.
14. In the website tree, run `npm run build`. The built folder is `dist`.
15. Run `python <kit>/check_full_page.py --site <website>/dist --full-page full-page.html --pages /<route> --chromium <pinned binary> --out <folder>/self-check`. It captures the built website and the full-page file with the kit at 1280 px and at 375 px, compares each pair with the judge, and writes `self-check.json` with the file's SHA-256, the SHA-256 of both fonts, the kit hash, the Chromium version and the finding counts per width. It passes only with zero blocked requests and zero font findings at both widths; other findings are the designed change and are counted. If it fails, rebuild `full-page.html` from the chosen option, embed again and run it with a fresh `--out` folder.
16. Call `raphael_sandbox_artifact` with `export` for `full-page.html` and for `self-check.json`.
17. Complete the card with the record `design_pin` in the completion metadata, exactly four fields, because a line in the summary is not enough: the comparison reads this record from the designer card.
   - `attachment_id`: the number of the full-page file on the designer card.
   - `sha256`: the 64 hex characters of that file.
   - `file`: the name `full-page.html`.
   - `pages`: a list that holds the compared route, for example `["/privacy"]`.

   The summary also names the number of the option, its title and the attachment of its picture, the attachment of the full-page file, and the pass line of the check, for example "self-check passed: 0 font findings at 1280 px and 375 px; 6 element findings are the designed change".

## Pitfalls

- A page that loads a font, script, image or style from an outside address fails the capture. Put everything inline.
- A page that moves or fades gives unstable pictures; leave animation out.
- A very tall page makes a large picture; the composer shrinks it evenly and may switch to JPEG to stay under 4 MiB.
- The composer and the check never overwrite: use a fresh output folder for a second attempt.
- Use the kit files from `designer_kit` of this card, not a copy from an earlier card; the scripts change between releases.
- A `design_pin` that names another file than the one on the card, or a hash that does not match it, fails the later comparison.

## Verification

- The provision receipt shows no credential vault, no credential proxy and an empty ownership scope.
- `design-offer-manifest.json` lists, for each option, the SHA-256 of its page file, both captures and its picture, and one kit hash and one Chromium version.
- Each picture shows the 1280 px page on the left and the 375 px page on the right, with both labels.
- The card is held for the owner's choice and nothing is built.
- After the choice, `self-check.json` shows zero font findings and zero blocked requests at both widths, and `design_pin` in the completion metadata holds the attachment number, the SHA-256, the name `full-page.html` and the compared route.

## Return to Claude Design

The Claude Design connection is switched off, not deleted. To return, switch the `mcp_servers.claude-design.enabled` setting of the designer back on and follow the old Claude Design procedure; this skill stays installed.
