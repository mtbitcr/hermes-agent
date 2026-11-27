---
name: running-build-work
description: Follow this skill whenever you run a build card as the implementation boundary; it gives the six steps from build machine to handover, the prompt-file list and the one-command rule.
---
# Build Task Run Skill

Your card gives the goal, the literal owned paths and the attachments that hold the analysis. The coding tool reads those attachments as files, so read only the card text yourself. Run the card with the six steps below, keeping to the card's tier and its new-parts list. The kernel owns the mechanical steps, so the coordinator stays thin.

## The coordinator's six steps

Do these six things, in this order, and nothing else.

1. Provision the isolated build machine with the platform's provision tool, `raphael_sandbox_provision`, and check its receipt: the machine id, the source commit and the `/workspace/task` and `/workspace/baseline` trees. Then import each attachment that the card names: call `raphael_sandbox_artifact` with `inspect` to get its SHA-256, then with `import` and that SHA-256. Each file lands at `/workspace/inputs/<id>/<name>`.
2. Write the coding tool's prompt to one file inside `/workspace/task`, carrying every item of the prompt-file list below.
3. Install the coding tool with one call, `npm install -g @anthropic-ai/claude-code@2.1.280`; the machine image does not carry it. Then launch it once from `/workspace/task` as `claude` in print mode, reading that prompt file on standard input, with `--permission-mode acceptEdits --allowed-tools Bash Read Edit Write Glob Grep TodoWrite` and JSON output, and with `CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000` in its environment: the machine's credential proxy signs a request only up to 1 MiB, about 330,000 tokens, and a larger request fails with 401 "Not logged in". One tool call ends after 120 seconds and a coding session runs far longer, so write the launch line into a small script under `/tmp` with the file tool, start that script detached with `setsid -f`, input from `/dev/null` and output to a result file outside the owned paths (a plain `nohup … &` has died when its tool call returned), check with `ps` a few seconds later that it runs, and wait in single calls of about 110 seconds each until the result file holds the final JSON result. Add a turn limit above 60 only if that version's help lists one. Never pass `--model` or `--effort`: the machine's environment carries the approved model and effort. If the tool stops on a 401 "Not logged in", make one short probe call and, if it answers, write a continuation prompt file that starts with one line (compare `/workspace/task` with the untouched `/workspace/baseline`, for example with `diff -ru`, since the task tree has no git history; keep what is already done and continue from there) followed by the original prompt, and launch a fresh session with it the same way, because a resumed session is refused at once; up to three fresh sessions in all; a fourth 401 goes into the completion line.
   At each progress check in this step, collect the current diff of only the card's exact owned paths, as in step 4. A final JSON result is not required for this progress export. Export that diff as a new attachment of the current run with step 5 before the next wait. Keep the earlier attachments. If the coding tool stops and no further continuation is taken under the existing recoverable-401 rule, record the actual stop reason and hand over the latest eligible patch under step 6. Keep the existing run deadline and completion limit. If that diff is still empty 40 minutes after a launch, end that session's process and launch one fresh session the same way, with the prompt plus a first line: "Write the failing tests now; read only what the notes file names." A second launch whose diff stays empty for 40 minutes goes into the completion line.

4. Collect the tool's JSON result and the diff of the owned paths, `/workspace/baseline` against `/workspace/task`, written to one file inside `/workspace/task`.
5. Attach that diff as a new attachment of the same run with the platform's artifact tool, `raphael_sandbox_artifact`, in the `export` direction, giving the file's absolute path and its SHA-256.
6. Complete with `kanban_complete`, setting its patch attachment to that attachment's id, before iteration 100 or two hours after the first launch, whichever comes first, whether or not the tool finished everything, with one line saying what is missing. Only a completion makes the kernel apply the patch and hand the card to its reviewer; a review request alone leaves the branch without the change. Before the completion, export `/workspace/task/POINT_TABLE.md` as one more attachment of the same run with the call of step 5, and name its attachment id in the completion metadata as `point_table`; when the file does not exist, set `point_table` to `none`.

Run no tests and no self-review, and never edit the owned files: the coding tool runs only the focused tests it wrote, and the wider gate is continuous integration and the independent reviews.

Make every tool call one command, under 300 characters, with no raw line break or tab inside any argument, no chained command lists and no heredocs; one violation crashes the run. File and attachment text is exempt from the length limit; send its line breaks and tabs escaped, never raw.

## The prompt-file list

- State the task's goal.
- Add "Touch nothing outside these exact owned paths:" and list each path literally, never as an area or a description.
- List the path of each imported file and the sections of it that apply, then add "Read these files first. Make the smallest change they call for, never wider." Copy only the card's own goal and rules into the prompt.
- Add "Read each named file one time. In a file of more than 300 lines, search first and read a range of at most 120 lines. When a step needs one more function, find it by search and read only that function."
- Add "Keep the notes file `/workspace/BUILD_NOTES.md`: after each step, write what is done and the names and line numbers that the next step needs. After a context compaction, read the notes file first and continue from it."
- Add "Write the focused tests first and see them fail before the change; run only those tests, never the whole suite."
- Add "The coding tool may run commands of any length; the one-command rule binds only the coordinator's own platform calls."
- Add "Each check you report means no failure caused by this change; continuous integration judges the whole suite."
- Add "Add a command, file or flag only when this prompt names it with its reason." and copy the card's new-parts list after it.
- When the change adds or touches a kernel tool that resolves a store path (a board, the project registry, an attachment), add: "Write one test under the exact environment the dispatcher injects into a worker: `HERMES_KANBAN_DB` and `HERMES_KANBAN_BOARD` pinned to the worker's own board, `HERMES_KANBAN_TASK` set, and `HERMES_HOME` pointing at a profile home under the root. From there the tool must still reach every other board and the root registry. Its production proof is read against an independent count."
- Add "Before the handover, write `/workspace/task/POINT_TABLE.md`: each numbered point of the facts file and of the card's acceptance, each with the name of the test that holds it or 'not held: <reason>'. A test that holds the opposite of a point is a defect to fix before the handover."
- Add "For each new input or read, write a focused test for each case that applies: negative, zero, empty, missing, a read that raises, a write that fails after a successful open, multibyte and NUL text, and a record stored by the previous version. List each case that does not apply in the point table."

## When the card needs the owner

When the card cannot go on without a decision of the owner, do not complete and do not guess. That is the case when the card or its facts file says that the build stops, when two rules of the card contradict each other, or when a file outside the owned paths must change. First write the technical detail, such as the name of the test or of the file, as a comment on the card. Then call `kanban_ask_owner` one time: the question in plain words for an owner who is not a programmer, with what stopped the work; at most four choices, each one line that says what happens next; no file path, no card id and no model name in the question or in a choice. The call holds the card for the owner, so end the turn after it. Use `kanban_block` only for a stop that is not a question to the owner.

## Pitfalls seen on the build machine

- The sandbox command and file tools are deferred: call them through `tool_call`, with `connect_if_missing=true` on the first call.
- The image carries no test environment. For a Python repository, install uv at the version pinned in `.github/workflows/tests.yml` from PyPI, check its SHA-256, run `uv sync --locked --no-install-project --python /usr/bin/python3 --extra all --extra dev` with `UV_PROJECT_ENVIRONMENT=/opt/venv`, and tell the coding tool to run its tests as `HERMES_PYTHON=/opt/venv/bin/python scripts/run_tests.sh <files>`; they run on the image's Python 3.12, and the completion line says so. For a Node repository, tell the coding tool to run `npm ci` once and to keep scratch files under `/tmp`. The image has no pip: read `https://pypi.org/pypi/uv/<version>/json`, download the manylinux x86_64 wheel, compare its SHA-256 with the digest in that JSON, and extract `uv` from the wheel's `scripts` folder into `/usr/local/bin`.
- The machine runs as root, so a test that needs an unreadable file fails there; the completion line names it as known.
- Tell the coding tool to write its test logs under `/tmp`, outside the task tree, so a continuation session can check them without redoing the work.
- If `kanban_show` spills to a file that you cannot read, read the card body and its text attachments with the deferred `kanban_read` tool, up to 10,000 characters for each call.
- A session at max effort can show only seconds of CPU over many minutes, because the model thinks on the provider's side. That is not a stall: new scratch files under `/tmp` or a larger task diff show progress.
- Build the patch one owned path at a time, with `diff -u --label a/<path> --label b/<path> /workspace/baseline/<path> /workspace/task/<path>` in a small script, so test residue (`.pytest_cache`, `test_durations.json`, `__pycache__`) and the prompt file never enter it. Check it with `git apply --check` against copies of the baseline files.
- The provision receipt marks manual machine management as blocked and cleanup as automatic. Do not stop the machine yourself, and say so in the completion metadata.
- An update card (its body hands over an approved head that the platform brought up to date on main): run one focused check of the owned test files, then hand over the unchanged updated head. Do not rebuild the base tree or verify the whole change again; the update's own review does that. When a review returned findings on this card, this line does not apply: build the change that the findings and the owner rules ask for, write its tests first, and hand over.

## Website builds

When the card builds or changes a website, the owner's standing rules of 2 November 2026 apply:

- Before the handover, run Lighthouse on every built page, once for desktop and once for mobile, with the headless Chromium that takes the screenshots (`CHROME_PATH`). Performance, Accessibility, Best Practices and SEO each reach 100. Attach the scores of every page as one JSON file, and name each score under 100 in the completion line as unfinished work.
- Attach screenshots of every page at 1280 px and 375 px, in each theme, so that the owner's side can compare them with the design prototypes.
- The first view shows at once, without waiting for a script or an animation, and the reduced-motion setting shows every section in place.
