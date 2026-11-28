---
name: handing-build-work
description: Follow this skill whenever you compose the instructions of a build task handed to the implementation boundary.
---
# Build Task Handoff Skill

Copy the next two sections into the build task in full, filled in with its goal, owned paths and the names of the attached analysis files and their sections; soften, shorten or drop nothing. The kernel owns the mechanical steps, so the coordinator stays thin.

## The coordinator's six steps

Do these six things, in this order, and nothing else.

1. Provision the isolated build machine with the platform's provision tool, `raphael_sandbox_provision`, and check its receipt: the machine id, the source commit and the `/workspace/task` and `/workspace/baseline` trees. Then import each attachment that the card names: call `raphael_sandbox_artifact` with `inspect` to get its SHA-256, then with `import` and that SHA-256. Each file lands at `/workspace/inputs/<id>/<name>`.
2. Write the coding tool's prompt to one file inside `/workspace/task`, carrying every item of the prompt-file list below.
3. Install the coding tool with one call, `npm install -g @anthropic-ai/claude-code@2.1.280`; the machine image does not carry it. Then launch it once from `/workspace/task` as `claude` in print mode, reading that prompt file on standard input, with `--permission-mode acceptEdits --allowed-tools Bash Read Edit Write Glob Grep TodoWrite` and JSON output, and with `CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000` in its environment: the machine's credential proxy signs a request only up to 1 MiB, about 330,000 tokens, and a larger request fails with 401 "Not logged in". One tool call ends after 120 seconds and a coding session runs far longer, so write the launch line into a small script under `/tmp` with the file tool, start that script detached with `setsid -f`, input from `/dev/null` and output to a result file outside the owned paths (a plain `nohup … &` has died when its tool call returned), check with `ps` a few seconds later that it runs, and wait in single calls of about 110 seconds each until the result file holds the final JSON result. Add a turn limit above 60 only if that version's help lists one. Never pass `--model` or `--effort`: the machine's environment carries the approved model and effort. If the tool stops on a 401 "Not logged in", make one short probe call and, if it answers, write a continuation prompt file that starts with one line (compare `/workspace/task` with the untouched `/workspace/baseline`, for example with `diff -ru`, since the task tree has no git history; keep what is already done and continue from there) followed by the original prompt, and launch a fresh session with it the same way, because a resumed session is refused at once; up to three fresh sessions in all; a fourth 401 goes into the completion line.
4. Collect the tool's JSON result and the diff of the owned paths, `/workspace/baseline` against `/workspace/task`, written to one file inside `/workspace/task`.
5. Attach that diff as a new attachment of the same run with the platform's artifact tool, `raphael_sandbox_artifact`, in the `export` direction, giving the file's absolute path and its SHA-256.
6. Complete with `kanban_complete`, setting its patch attachment to that attachment's id, before iteration 100 or two hours after the first launch, whichever comes first, whether or not the tool finished everything, with one line saying what is missing. Only a completion makes the kernel apply the patch and hand the card to its reviewer; a review request alone leaves the branch without the change.

Run no tests and no self-review, and never edit the owned files: the coding tool runs only the focused tests it wrote, and the wider gate is continuous integration and the independent reviews.

Make every tool call one command, under 300 characters, with no raw line break or tab inside any argument, no chained command lists and no heredocs; one violation crashes the run. File and attachment text is exempt from the length limit; send its line breaks and tabs escaped, never raw.

## The prompt-file list

- State the task's goal.
- Add "Touch nothing outside these exact owned paths:" and list each path literally, never as an area or a description.
- List the path of each imported file and the sections of it that apply, then add "Read these files first. Make the smallest change they call for, never wider." Copy only the card's own goal and rules into the prompt.
- Add "Write the focused tests first and see them fail before the change; run only those tests, never the whole suite."
- Add "The coding tool may run commands of any length; the one-command rule binds only the coordinator's own platform calls."
- Add "Each check you report means no failure caused by this change; continuous integration judges the whole suite."
- Add "Add a command, file or flag only when this prompt names it with its reason." and copy the card's new-parts list after it.
- When the change adds or touches a kernel tool that resolves a store path (a board, the project registry, an attachment), add: "Write one test under the exact environment the dispatcher injects into a worker: `HERMES_KANBAN_DB` and `HERMES_KANBAN_BOARD` pinned to the worker's own board, `HERMES_KANBAN_TASK` set, and `HERMES_HOME` pointing at a profile home under the root. From there the tool must still reach every other board and the root registry. Its production proof is read against an independent count."

## The card's tier, new parts and box

- Name the risk tier on the first line of the task text. Tier 0: text, documentation or settings without data writes. Tier 1: tests only, owner pages, data reads, and code changes near 100 lines or fewer. Tier 2: the merge and release paths, access, credentials or money, and larger code changes that write data. When two tiers fit, take the higher. Tier 1 work runs on the routine execution tier.
- List each new command, file or flag the change adds, with a one-line reason. A fix adds one only when the owner's request gives the reason; quote it.
- The box is two hours of work: step 6 above carries it.
- Keep the change near 500 changed lines or less. A larger change becomes two or more cards, and each card delivers one complete piece of behaviour that works on its own.

## Edges decided before the build

Every build card holds these three parts, decided before the build. A card with no input form, or with no actors that work at one time, says so in one line.

1. the closed list of supported input forms;
2. the closed list of supported orderings of concurrent actors, each with its named interleaving test;
3. "Any other form or ordering is refused before any state change, with a named refusal test."

When the facts sheet asks for "every form" of an open format or "any order" of concurrent actors, propose the closed list as one owner decision in the plan, before the build card is approved.

## Before sending the build task

- Confirm both sections are copied in full, with the install line, the launch flags, the compaction window and the 401 relaunch sentence exactly as this skill gives them and no `--model` or `--effort`.
- Confirm the owned paths are literal paths.
- Confirm the analysis is attached and the card names the file and the sections that apply.
- Confirm the store-path test is present when the change resolves a store path.
- Confirm the tier line and the new-parts list are present.
- Confirm the change fits about 500 changed lines, or that it is split into cards that each work on their own.
