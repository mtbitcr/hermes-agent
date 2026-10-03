"""Delivery fence core: pure decisions for publishing, rerunning, lens requests and merging.

Each function decides from the facts its caller passes in and does nothing else: no file,
network, process, environment, clock or database. The reviewed policy file is the only flaky
list; the caller reads the installed file and passes its text to load_policy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from hermes_cli.kanban_pr_acceptance import _classify

# The owner keeps these out of the policy file; each is the CI's literal text, so a renamed
# job or a new slice count is refused until a reviewed change updates it here. The jobs API
# names a called workflow's job "<calling job> / <called job>": tests.yml's slices carry
# ci.yaml's "Python tests" prefix, their steps and ci.yaml's own summary job carry none.
SUMMARY_CHECK = "All required checks pass"
TEST_JOB_STEPS = {f"Python tests / Run tests slice {n}/12": f"Run tests (slice {n}/12)" for n in range(1, 13)}
# The plan names no lenses, only "both lens verdicts", so the merge counts distinct lenses on H.
LENS_COUNT = 2

_PASSING = frozenset({"success", "skipped", "neutral"})
# GitHub turns a dismissed review's state into DISMISSED, so a dismissal decides like a verdict.
_DECISIVE_REVIEWS = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})
_SHA = re.compile(r"[0-9a-f]{40}")
# kanban_pr_acceptance._REPO's shape; the owner allows only _classify to be imported from there.
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# One exact pytest node id: a .py path and at least one name, with no glob or parameter brackets.
_NODE_ID = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*[.]py(?:::[A-Za-z_][A-Za-z0-9_]*)+")
_WILDCARDS = frozenset("*?[]")


@dataclass(frozen=True)
class Decision:
    """A fence's answer. allowed is False only for a refusal: not_needed, already_published,
    already_requested and already_merged are allowed with nothing left to do."""

    allowed: bool
    code: str
    detail: str
    matches: list = field(default_factory=list)


@dataclass(frozen=True)
class Policy:
    """The reviewed policy per repository: required check names, and exact flaky test ids
    mapped to the finding that listed them."""

    required_checks: dict[str, tuple[str, ...]]
    flaky_tests: dict[str, dict[str, int]]


def _refuse(code: str, detail: str) -> Decision:
    return Decision(False, code, detail)


def _independent(reviewer: str | None, implementer: str | None) -> bool:
    # A missing identity cannot be shown to differ, so it fails closed.
    return bool(reviewer) and bool(implementer) and reviewer != implementer


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    # json silently keeps the last copy of a repeated key; the reviewed file must say one thing.
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("the policy repeats a key")
    return dict(pairs)


def load_policy(text: str) -> Policy:
    """Parse the text the caller read from the installed policy file; any other shape is a
    ValueError, so nothing but the reviewed entries can ever count as required or flaky."""
    document = json.loads(text, object_pairs_hook=_unique_keys)
    if not isinstance(document, dict):
        raise ValueError("the policy must be an object keyed by repository")
    required_checks, flaky_tests = {}, {}
    for repo, entry in document.items():
        if not _REPO.fullmatch(repo):
            raise ValueError("a policy key is not an owner/name repository")
        if not isinstance(entry, dict) or set(entry) != {"required_checks", "flaky_tests"}:
            raise ValueError(f"{repo}: an entry holds exactly required_checks and flaky_tests")
        required_checks[repo] = _required_check_names(repo, entry["required_checks"])
        flaky_tests[repo] = _flaky_test_findings(repo, entry["flaky_tests"])
    return Policy(required_checks, flaky_tests)


def _required_check_names(repo: str, names: object) -> tuple[str, ...]:
    if not isinstance(names, list) or not names:
        raise ValueError(f"{repo}: required_checks must be a non-empty list")
    if not all(isinstance(name, str) and name and not _WILDCARDS & set(name) for name in names):
        raise ValueError(f"{repo}: a required check is not an exact check name")
    if len(set(names)) != len(names):
        raise ValueError(f"{repo}: a required check is listed twice")
    return tuple(names)


def _flaky_test_findings(repo: str, entries: object) -> dict[str, int]:
    if not isinstance(entries, list):
        raise ValueError(f"{repo}: flaky_tests must be a list")
    findings = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"test", "finding"}:
            raise ValueError(f"{repo}: a flaky entry holds exactly test and finding")
        test, finding = entry["test"], entry["finding"]
        # Matching is exact equality, so an entry must name one test: no pattern, file or parameter set.
        if not isinstance(test, str) or not _NODE_ID.fullmatch(test):
            raise ValueError(f"{repo}: a flaky entry is not one exact test id")
        if type(finding) is not int or finding < 1:
            raise ValueError(f"{repo}: a flaky entry's finding is not a finding number")
        if test in findings:
            raise ValueError(f"{repo}: a flaky test is listed twice")
        findings[test] = finding
    return findings


def classify_check(check: dict, head: str) -> str:
    """Classify one check run or legacy status for H exactly as collect_acceptance does."""
    is_run = "conclusion" in check
    return _classify(check, head, check.get("conclusion") if is_run else check["state"], is_run)


def decide_publish(approval: dict, ledger: dict, remote: dict) -> Decision:
    """T1: may the kernel-approved head H become the card's one pull request?

    approval: source_done, approved_by_review_lane, head_commit (H), review_head (the latest
    review head provenance), bound_head, reviewer, implementer, reopened_after_approval, and
    git's head_exists, base and base_is_ancestor. ledger: the recorded pull_request (None when
    none), its head and state, and head_is_ancestor (git: that head is an ancestor of H).
    remote: branch_head (None when the branch is absent) and open_pulls [{number, head}].
    """
    head = approval["head_commit"]
    if not (approval["source_done"] and approval["approved_by_review_lane"]):
        return _refuse("not_approved", "the source card is not done with an approving review-lane completion")
    if not isinstance(head, str) or not _SHA.fullmatch(head):
        return _refuse("invalid_head", "head_commit is not a 40-character lowercase hex SHA")
    if not _independent(approval["reviewer"], approval["implementer"]):
        return _refuse("reviewer_not_independent", "the reviewer is missing or is the implementer")
    if approval["reopened_after_approval"]:
        return _refuse("reopened_after_approval", "the card was reopened after the approval")
    if head != approval["review_head"] or head != approval["bound_head"]:
        return _refuse("head_moved", f"{head} is not both the reviewed head and the bound head")
    if not (approval["head_exists"] and approval["base_is_ancestor"] and approval["base"] != head):
        return _refuse("head_unconfirmed", "git does not confirm H with the base as a distinct ancestor")
    recorded, branch, pulls = ledger["pull_request"], remote["branch_head"], remote["open_pulls"]
    if any(pull["number"] != recorded for pull in pulls):
        return _refuse("second_pull_request", "an open pull request other than the recorded one exists")
    if recorded is None:
        if branch is None:
            return Decision(True, "push_and_create", f"push {head} to the absent branch and open the pull request")
        if branch == head:
            return Decision(True, "adopt_and_create", f"the branch already holds {head}; open the pull request")
        return _refuse("head_moved", "the branch holds a head other than H")
    if not pulls:
        return _refuse("pull_request_not_open", "the recorded pull request is not open")
    old, pull_head = ledger["head"], pulls[0]["head"]
    if old == head:
        if pull_head == head and branch == head:
            return Decision(True, "already_published", f"the recorded pull request is open at {head}")
        return _refuse("head_moved", "the pull request or the branch is not at the recorded head H")
    # An older recorded head is allowed only after a return for changes, as a fast-forward to H.
    if ledger["state"] != "returned_for_changes":
        return _refuse("head_moved", "the ledger records another head without a return for changes")
    if not ledger["head_is_ancestor"]:
        return _refuse("not_fast_forward", "the recorded head is not an ancestor of H")
    if pull_head == old and branch == old:
        return Decision(True, "push_fast_forward", f"push {head} with a lease expecting exactly {old}")
    if pull_head == head and branch == head:
        return Decision(True, "adopt_fast_forward", f"the pull request already holds {head}; record it")
    return _refuse("head_moved", "the branch or the pull request is at neither the recorded head nor H")


def decide_rerun(policy: Policy, repo: str, head: str, jobs: list[dict], ledger: dict) -> Decision:
    """T3: may the failed jobs of H's workflow run be rerun once, as listed flaky failures?

    jobs: every job of the run with id, name, head_sha, status, conclusion, run_attempt and
    steps [{name, conclusion}]; a failed slice also carries failed_tests (the test ids its log
    names) and failed_count (the count its pytest summary reports). ledger: {"reruns": [heads
    already rerun]}. All or nothing: one ineligible failed job refuses the whole rerun.
    """
    if repo not in policy.flaky_tests:
        return _refuse("unknown_repository", "the policy has no entry for this repository")
    if any(job["head_sha"] != head for job in jobs):
        return _refuse("head_moved", f"a job ran on a head other than {head}")
    if any(job["status"] != "completed" for job in jobs):
        return _refuse("not_completed", "a job of the run has not completed")
    failed = [job for job in jobs if job["conclusion"] not in _PASSING]
    if not failed:
        return Decision(True, "not_needed", "no job failed")
    if head in ledger["reruns"] or any(job["run_attempt"] > 1 for job in jobs):
        return _refuse("rerun_used", "H already had its one rerun")
    # Owner decision 2: the summary check is a summary only beside other failed jobs that are all
    # eligible. It is then not rerun itself; GitHub reruns it after the jobs it waits on. Only its
    # ordinary failure summarises them: a cancelled, timed-out, missing or unknown end stops.
    if any(job["conclusion"] != "failure" for job in failed if job["name"] == SUMMARY_CHECK):
        return _refuse("summary_not_failure", "the summary check ended in something other than failure")
    others = [job for job in failed if job["name"] != SUMMARY_CHECK]
    if not others:
        return _refuse("summary_failed_alone", "the summary check failed and no other job did")
    flaky, matches = policy.flaky_tests[repo], []
    for job in others:
        if not _is_test_job(job):
            return _refuse("not_test_job", f"job {job['id']} did not fail only at its test step")
        named = job.get("failed_tests") or []
        if not named:
            return _refuse("no_failed_test", f"job {job['id']} names no failed test")
        count = job.get("failed_count")
        if type(count) is not int or count != len(named) or len(set(named)) != len(named):
            return _refuse("count_mismatch", f"job {job['id']} does not prove its count of failed tests")
        # Exact equality only: a test is flaky when this repository's list holds exactly that id.
        if any(test not in flaky for test in named):
            return _refuse("not_flaky", f"job {job['id']} failed a test the policy does not list as flaky")
        matches += [{"job_id": job["id"], "test": test, "list_entry": {"test": test, "finding": flaky[test]}}
                    for test in named]
    ids = [job["id"] for job in others]
    return Decision(True, "rerun", f"rerun jobs {ids} once; GitHub reruns the summary check after them", matches)


def _is_test_job(job: dict) -> bool:
    # Test shape: a slice job that failed with exactly one failed step, the slice's test step.
    failed_steps = [step for step in job["steps"] if step["conclusion"] not in _PASSING]
    return (job["conclusion"] == "failure" and len(failed_steps) == 1
            and failed_steps[0]["conclusion"] == "failure"
            and failed_steps[0]["name"] == TEST_JOB_STEPS.get(job["name"]))


def decide_lens_request(head: str, pr_head: str, evidence: list[dict], existing_lens_cards: list[dict],
                        reviewer: str | None, implementer: str | None) -> Decision:
    """T4: may read-only lens review cards bound to H be requested?

    evidence: the T2 and T3 records [{id, kind ("checks" or "rerun"), head}], where a checks
    record also carries overall, digest (the attachment's recorded SHA-256) and reread_digest
    (its SHA-256 as just re-read). reviewer: from policy_resolved_reviewer().
    existing_lens_cards: [{lens, task_id, head}].
    """
    if pr_head != head:
        return _refuse("head_moved", f"the pull request head is not {head}")
    records = [record for record in evidence if record["head"] == head]
    if not records:
        return _refuse("no_evidence", f"no check evidence is recorded for {head}")
    latest = max(records, key=lambda record: record["id"])
    if latest["kind"] != "checks":
        return _refuse("evidence_before_rerun", "H was rerun after its latest check evidence")
    if latest["overall"] != "success":
        return _refuse("checks_not_passing", "the latest check evidence for H is not success")
    if not latest["digest"] or latest["reread_digest"] != latest["digest"]:
        return _refuse("evidence_unverified", "the evidence attachment does not re-read to its recorded SHA-256")
    if not _independent(reviewer, implementer):
        return _refuse("reviewer_not_independent", "the reviewer is missing or is the implementer")
    # Idempotent on (source card, H, lens): existing cards for H are returned, never added to.
    cards = [card for card in existing_lens_cards if card["head"] == head]
    if cards:
        return Decision(True, "already_requested", "lens cards for H exist; create no more", cards)
    return Decision(True, "request", f"request read-only lens cards bound to {head}, carrying the evidence")


def decide_merge(policy: Policy, repo: str, head: str, pr: dict, checks: dict, verdicts: list[dict],
                 github_reviews: list[dict], security_reviewer: str | None) -> Decision:
    """T6: may the pull request be merged at exactly H?

    pr: state, merged, head, ledger_head, mergeable, mergeable_state. checks: GitHub's latest
    check runs and legacy statuses for H, {total_count, check_runs, statuses}, every one on H,
    with two kernel record ids: evidence_id (the check evidence they were recorded under) and
    last_rerun_id (the last rerun recorded for H, None when H was never rerun; required).
    verdicts: lens verdicts [{id, lens, head, verdict}], id being the verdict's kernel record
    id. As in decide_lens_request, a larger id was recorded later: the evidence must follow the
    last rerun, and every approval for H the evidence. github_reviews: the pull request's
    reviews as GitHub returns them. security_reviewer: the security reviewer's GitHub login;
    the plan does not say who supplies it, so a missing one refuses.
    """
    if repo not in policy.required_checks:
        return _refuse("unknown_repository", "the policy has no entry for this repository")
    if pr["merged"] and pr["head"] == head:
        return Decision(True, "already_merged", f"the pull request is already merged at {head}")
    if pr["state"] != "open":
        return _refuse("pull_request_not_open", "the pull request is not open")
    if pr["head"] != head or pr["ledger_head"] != head:
        return _refuse("head_moved", f"the pull request head and the ledger head are not both {head}")
    if len({run["id"] for run in checks["check_runs"]}) != checks["total_count"]:
        return _refuse("incomplete_checks", "the check runs read do not match GitHub's total_count")
    # Section 4 binds every check read to H, required or not, its head read as _classify reads it.
    supplied = [*checks["check_runs"], *checks["statuses"]]
    if any(check.get("head_sha", check.get("sha")) != head for check in supplied):
        return _refuse("check_stale", f"a check run or status read is not on {head}")
    # Only an explicit None says H was never rerun; a missing last_rerun_id proves nothing.
    evidence_id, last_rerun_id = checks.get("evidence_id"), checks.get("last_rerun_id", False)
    if type(evidence_id) is not int or not (last_rerun_id is None or type(last_rerun_id) is int):
        return _refuse("order_unproven", "the check evidence or the last rerun has no kernel record id")
    if last_rerun_id is not None and evidence_id <= last_rerun_id:
        return _refuse("evidence_before_rerun", "H was rerun after its check evidence was recorded")
    for name in policy.required_checks[repo]:
        outcome = _required_check(name, head, checks)
        if outcome != "success":
            return _refuse(f"check_{outcome}", f"required check {name!r} is {outcome} for H")
    on_head = [verdict for verdict in verdicts if verdict["head"] == head]
    if any(type(verdict.get("id")) is not int for verdict in on_head):
        return _refuse("order_unproven", "a lens verdict for H has no kernel record id")
    if any(verdict["verdict"] != "approve" for verdict in on_head):
        return _refuse("changes_verdict", "a lens verdict for H is not approve")
    if any(verdict["id"] <= evidence_id for verdict in on_head):
        return _refuse("verdict_before_evidence", "a lens approval for H was recorded before its check evidence")
    if len({verdict["lens"] for verdict in on_head}) != LENS_COUNT:
        return _refuse("lens_approval_missing", f"H lacks approve verdicts from {LENS_COUNT} distinct lenses")
    if not security_reviewer:
        return _refuse("no_security_reviewer", "no security reviewer identity was supplied")
    # Owner change, no approval mirror: the security reviewer's own latest decisive review decides.
    own = [review for review in github_reviews if review["state"] in _DECISIVE_REVIEWS
           and (review.get("user") or {}).get("login") == security_reviewer]
    latest = max(own, key=lambda review: review["id"], default=None)
    if latest is None or latest["state"] != "APPROVED" or latest["commit_id"] != head:
        return _refuse("security_approval_missing", f"the security reviewer's own latest review is not an approval of {head}")
    if pr["mergeable"] is not True or pr["mergeable_state"] != "clean":
        return _refuse("not_mergeable", "GitHub does not report the pull request as mergeable and clean")
    return Decision(True, "merge", f"merge with sha={head}; request no bypass")


def _required_check(name: str, head: str, checks: dict) -> str:
    # collect_acceptance's selection for an unpinned context (the policy pins no app): every check
    # run with the name plus the latest legacy status with that context; none at all is missing.
    selected = [run for run in checks["check_runs"] if run["name"] == name]
    legacy = [status for status in checks["statuses"] if status["context"] == name]
    selected += [max(legacy, key=lambda status: status["id"])] if legacy else []
    outcomes = [classify_check(check, head) for check in selected] or ["missing"]
    return next((outcome for outcome in outcomes if outcome != "success"), "success")
