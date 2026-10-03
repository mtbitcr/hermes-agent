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
_DIGEST = re.compile(r"[0-9a-f]{64}")
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


# Owner rule 1: one small check per fact-type family. A fact without its documented type refuses
# with invalid_fact, so it can never pass as truthy, falsy, equal or distinct.
def _is_bool(value: object) -> bool:
    return type(value) is bool


def _is_positive_int(value: object) -> bool:
    # Ids, counts and attempts; Python's bool is an int, a fence's never is.
    return type(value) is int and value >= 1


def _has_unique_ids(records: list[dict]) -> bool:
    ids = [record.get("id") for record in records]
    return all(_is_positive_int(record_id) for record_id in ids) and len(set(ids)) == len(ids)


def _is_sha(value: object) -> bool:
    # Heads and commit ids.
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _is_text(value: object) -> bool:
    # Identities and lens names.
    return isinstance(value, str) and value != ""


def _is_digest(value: object) -> bool:
    # Owner rule 5: a SHA-256 digest is 64 lowercase hex characters before any comparison.
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


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


def classify_check(check: dict, head: str, is_run: bool) -> str:
    """Classify one check run (is_run) or legacy status for H exactly as collect_acceptance does.
    Owner rule 2: the kind comes from the list the check was read from, never from its fields."""
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
    # Owner rule 1: the approval's flags are bools, its base a SHA and its identities non-empty strs.
    # A None or "" identity still refuses below as missing, the code the existing tests pin.
    flags = ("source_done", "approved_by_review_lane", "reopened_after_approval", "head_exists", "base_is_ancestor")
    if not (all(_is_bool(approval[flag]) for flag in flags) and _is_sha(approval["base"])
            and all(_is_text(approval[who]) or approval[who] in (None, "") for who in ("reviewer", "implementer"))):
        return _refuse("invalid_fact", "an approval flag, the base or an identity does not have its documented type")
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
    # Owner rule 1: H and the rerun heads are SHAs and the reruns a list, and every run_attempt is a
    # positive int, so a null head matches no null job head and a bool or float is no first attempt.
    reruns = ledger["reruns"]
    if not (_is_sha(head) and isinstance(reruns, list) and all(_is_sha(rerun_head) for rerun_head in reruns)
            and all(_is_positive_int(job.get("run_attempt")) for job in jobs)):
        return _refuse("invalid_fact", "H, the rerun ledger or a job's run_attempt does not have its documented type")
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
    # Owner rule 3: it counts as a summary only when its only failed step is ci.yaml's "Evaluate job
    # results"; failed at any other step it is one more failed job, refused under the rule above.
    # One with no failed step at all stays a summary: the existing tests' summary fixture has none.
    others = [job for job in failed if job["name"] != SUMMARY_CHECK
              or [(step["name"], step["conclusion"]) for step in job["steps"] if step["conclusion"] not in _PASSING]
              not in ([], [("Evaluate job results", "failure")])]
    if not others:
        return _refuse("summary_failed_alone", "the summary check failed and no other job did")
    flaky, matches = policy.flaky_tests[repo], []
    for job in others:
        if not _is_test_job(job):
            return _refuse("not_test_job", f"job {job['id']} did not fail only at its test step")
        named = job.get("failed_tests")
        # Owner rule 1: failed_tests is the documented list, so a dict of listed ids proves nothing.
        # None still refuses below as naming no test, the code the existing tests pin.
        if named is not None and not isinstance(named, list):
            return _refuse("invalid_fact", f"job {job['id']}'s failed_tests is not a list")
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
    # Test shape: a slice job on the TEST_JOB_STEPS allowlist that failed with exactly one failed
    # step, the slice's test step. The name must be a str key and the step's name a non-empty str
    # equal to its value, so a job off the list never matches through a failed step with no name.
    failed_steps = [step for step in job["steps"] if step["conclusion"] not in _PASSING]
    return (isinstance(job["name"], str) and job["name"] in TEST_JOB_STEPS
            and job["conclusion"] == "failure" and len(failed_steps) == 1
            and failed_steps[0]["conclusion"] == "failure"
            and isinstance(failed_steps[0]["name"], str) and failed_steps[0]["name"] != ""
            and failed_steps[0]["name"] == TEST_JOB_STEPS[job["name"]])


def decide_lens_request(head: str, pr_head: str, evidence: list[dict], existing_lens_cards: list[dict],
                        reviewer: str | None, implementer: str | None) -> Decision:
    """T4: may read-only lens review cards bound to H be requested?

    evidence: the T2 and T3 records [{id, kind ("checks" or "rerun"), head}], where a checks
    record also carries overall, digest (the attachment's recorded SHA-256) and reread_digest
    (its SHA-256 as just re-read). reviewer: from policy_resolved_reviewer().
    existing_lens_cards: [{lens, task_id, head}].
    """
    # Owner rule 1: H is a SHA and the identities non-empty strs; policy_resolved_reviewer() may
    # return None, and a None or "" reviewer still refuses below as missing, as the existing tests pin.
    if not (_is_sha(head) and _is_text(implementer) and (_is_text(reviewer) or reviewer in (None, ""))):
        return _refuse("invalid_fact", "H or an identity does not have its documented type")
    if pr_head != head:
        return _refuse("head_moved", f"the pull request head is not {head}")
    records = [record for record in evidence if record["head"] == head]
    if not records:
        return _refuse("no_evidence", f"no check evidence is recorded for {head}")
    # Owner rule 1: ids order the records only as unique positive ints; no str or tied id picks the latest.
    if not _has_unique_ids(evidence):
        return _refuse("invalid_fact", "an evidence record id is not a unique positive int")
    latest = max(records, key=lambda record: record["id"])
    if latest["kind"] != "checks":
        return _refuse("evidence_before_rerun", "H was rerun after its latest check evidence")
    if latest["overall"] != "success":
        return _refuse("checks_not_passing", "the latest check evidence for H is not success")
    if not (_is_digest(latest["digest"]) and _is_digest(latest["reread_digest"])) or latest["reread_digest"] != latest["digest"]:
        return _refuse("evidence_unverified", "the evidence attachment does not re-read to its recorded SHA-256")
    if not _independent(reviewer, implementer):
        return _refuse("reviewer_not_independent", "the reviewer is missing or is the implementer")
    # Idempotent on (source card, H, lens): existing cards for H are returned, never added to.
    cards = [card for card in existing_lens_cards if card["head"] == head]
    if cards:
        return Decision(True, "already_requested", "lens cards for H exist; create no more", cards)
    return Decision(True, "request", f"request read-only lens cards bound to {head}, carrying the evidence")


def decide_merge(policy: Policy, repo: str, head: str, pr: dict, checks: dict, verdicts: list[dict],
                 github_reviews: list[dict], security_reviewer: str | None,
                 rerun_attempt: int | None = None) -> Decision:
    """T6: may the pull request be merged at exactly H?

    pr: state, merged, head, ledger_head, mergeable, mergeable_state. checks: GitHub's latest
    check runs and legacy statuses for H, {total_count, check_runs, statuses}, every one on H,
    with two kernel record ids: evidence_id (the check evidence they were recorded under) and
    last_rerun_id (the last rerun recorded for H, None when H was never rerun; required).
    verdicts: lens verdicts [{id, lens, head, verdict}], id being the verdict's kernel record
    id. As in decide_lens_request, a larger id was recorded later: the evidence must follow the
    last rerun, and every approval for H the evidence. github_reviews: the pull request's
    reviews as GitHub returns them. security_reviewer: the security reviewer's GitHub login;
    the plan does not say who supplies it, so a missing one refuses. rerun_attempt: when H was
    rerun, the rerun attempt number, the run_attempt H's jobs had when their rerun was requested;
    every required check run must then carry an int run_attempt greater than it, or the merge
    refuses. None, the default, checks no attempt, so callers that do not pass it keep working.
    """
    if repo not in policy.required_checks:
        return _refuse("unknown_repository", "the policy has no entry for this repository")
    # Owner rule 1: H is a SHA, so a null head matches no null pull request, check, verdict or review head.
    if not _is_sha(head):
        return _refuse("invalid_fact", "H is not a 40-character lowercase hex SHA")
    if pr["merged"] and pr["head"] == head:
        return Decision(True, "already_merged", f"the pull request is already merged at {head}")
    if pr["state"] != "open":
        return _refuse("pull_request_not_open", "the pull request is not open")
    if pr["head"] != head or pr["ledger_head"] != head:
        return _refuse("head_moved", f"the pull request head and the ledger head are not both {head}")
    # Owner rule 1: total_count is a count, never a bool, and the check run ids are unique positive ints.
    # A total_count of 0 (no check run) stays allowed: the existing tests pin it.
    total = checks["total_count"]
    if not ((_is_positive_int(total) or (total == 0 and type(total) is int)) and _has_unique_ids(checks["check_runs"])):
        return _refuse("invalid_fact", "total_count or a check run id does not have its documented type")
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
    # Owner rule 1: the latest legacy status is chosen only among unique positive int ids.
    if not _has_unique_ids(checks["statuses"]):
        return _refuse("invalid_fact", "a legacy status id is not a unique positive int")
    # Owner rule 4: re-reading an old success does not make it fresh; after H's rerun every required
    # check run must come from a later attempt.
    if rerun_attempt is not None:
        required = [run for run in checks["check_runs"] if run["name"] in policy.required_checks[repo]]
        if not (_is_positive_int(rerun_attempt) and all(_is_positive_int(run.get("run_attempt")) for run in required)):
            return _refuse("invalid_fact", "the rerun attempt or a required check run's run_attempt is not a positive int")
        if any(run["run_attempt"] <= rerun_attempt for run in required):
            return _refuse("check_stale", f"a required check run is not from an attempt after {rerun_attempt}")
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
    # Owner rule 1: lens names are non-empty strs, so None and "" never count as two lenses.
    if not all(_is_text(verdict["lens"]) for verdict in on_head):
        return _refuse("invalid_fact", "a lens verdict for H has no lens name")
    if len({verdict["lens"] for verdict in on_head}) != LENS_COUNT:
        return _refuse("lens_approval_missing", f"H lacks approve verdicts from {LENS_COUNT} distinct lenses")
    # Owner rule 1: a supplied security reviewer is a non-empty str, so no number matches a login.
    # None or "" still refuses as missing, the code the existing tests pin.
    if security_reviewer not in (None, "") and not _is_text(security_reviewer):
        return _refuse("invalid_fact", "the security reviewer identity is not a non-empty str")
    if not security_reviewer:
        return _refuse("no_security_reviewer", "no security reviewer identity was supplied")
    # Owner rule 1: the latest review is chosen only among unique positive int ids.
    if not _has_unique_ids(github_reviews):
        return _refuse("invalid_fact", "a review id is not a unique positive int")
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
    # Owner rule 2: each check keeps the kind of the list it came from.
    selected = [(run, True) for run in checks["check_runs"] if run["name"] == name]
    legacy = [status for status in checks["statuses"] if status["context"] == name]
    selected += [(max(legacy, key=lambda status: status["id"]), False)] if legacy else []
    outcomes = [classify_check(check, head, is_run) for check, is_run in selected] or ["missing"]
    return next((outcome for outcome in outcomes if outcome != "success"), "success")
