"""Delivery fence core: pure fence decisions over facts the caller supplies.

Tests 1 to 14 are plan section 8's, adapted as the owner directs: load_policy takes the
policy text, and the merge reads the security reviewer's own approval (no mirror). The
four tests after them show owner cases that tests 1 to 14 do not show in full, the next
four are regressions for the independent review's findings 1 to 4, and the last six are
the third pass's: the next review's null-step regression, a run with no jobs, and, per
decision, the owner's rule that every fact it reads refuses when missing or malformed.
"""
import ast
import copy
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_delivery_fences as fences

H = "a" * 40
OLD = "b" * 40
OTHER = "c" * 40
BASE = "d" * 40
DIGEST = "e" * 64
PLATFORM = "mtbitcr/hermes-agent"
WORKSPACE = "mtbitcr/raphael-workspace"
GATE = "All required checks pass"
FLAKY_46 = ("tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::"
            "test_timed_out_child_keeps_relay_session_until_its_turn_exits")
FLAKY_100 = ("tests/tui_gateway/test_compute_host_turn_protocol.py::"
             "test_turn_start_streams_deltas_then_turn_end_with_history_identity")
UNLISTED = "tests/tools/test_example.py::test_unlisted"
SECURITY = "security-reviewer"
# Test data in the policy file's shape; only test 13 reads the real file.
POLICY_TEXT = json.dumps({
    PLATFORM: {"required_checks": [GATE], "flaky_tests": [
        {"test": FLAKY_46, "finding": 46}, {"test": FLAKY_100, "finding": 100}]},
    WORKSPACE: {"required_checks": ["quality"], "flaky_tests": []},
})
# Kernel record ids order the merge facts: H's check evidence (20, see checks) precedes both approvals.
LENSES = ({"id": 30, "lens": "first-lens", "head": H, "verdict": "approve"},
          {"id": 31, "lens": "second-lens", "head": H, "verdict": "approve"})


def policy():
    return fences.load_policy(POLICY_TEXT)


def approval(**changes):
    return {"source_done": True, "approved_by_review_lane": True, "head_commit": H,
            "review_head": H, "bound_head": H, "reviewer": "reviewer-profile",
            "implementer": "worker-profile", "reopened_after_approval": False,
            "head_exists": True, "base": BASE, "base_is_ancestor": True, **changes}


def ledger(pull_request=None, head=None, state=None, head_is_ancestor=False):
    return {"pull_request": pull_request, "head": head, "state": state,
            "head_is_ancestor": head_is_ancestor}


def remote(branch_head=None, *open_pulls):
    return {"branch_head": branch_head,
            "open_pulls": [{"number": number, "head": head} for number, head in open_pulls]}


def job(job_id, name, conclusion="success", **facts):
    return {"id": job_id, "name": name, "head_sha": H, "status": "completed",
            "conclusion": conclusion, "run_attempt": 1,
            "steps": [{"name": "Set up job", "conclusion": "success"}], **facts}


def slice_job(job_id, n, tests, failed_count=None, step=None):
    steps = [{"name": "Checkout code", "conclusion": "success"},
             {"name": step or f"Run tests (slice {n}/12)", "conclusion": "failure"},
             {"name": "Upload per-slice durations", "conclusion": "skipped"}]
    # The jobs API names a called workflow's job "<calling job> / <called job>" (review finding 1).
    return job(job_id, f"Python tests / Run tests slice {n}/12", "failure", steps=steps, failed_tests=list(tests),
               failed_count=len(tests) if failed_count is None else failed_count)


def rerun(jobs, reruns=(), repo=PLATFORM):
    return fences.decide_rerun(policy(), repo, H, list(jobs), {"reruns": list(reruns)})


def evidence(record_id, overall="success", head=H, **changes):
    return {"id": record_id, "kind": "checks", "head": head, "overall": overall,
            "digest": DIGEST, "reread_digest": DIGEST, **changes}


def rerun_record(record_id, head=H):
    return {"id": record_id, "kind": "rerun", "head": head}


def lens_request(records, pr_head=H, cards=(), reviewer="reviewer-profile"):
    return fences.decide_lens_request(H, pr_head, list(records), list(cards), reviewer, "worker-profile")


def pr(**changes):
    return {"state": "open", "merged": False, "head": H, "ledger_head": H,
            "mergeable": True, "mergeable_state": "clean", **changes}


def check_run(run_id=1, name=GATE, **changes):
    return {"id": run_id, "name": name, "head_sha": H, "status": "completed",
            "conclusion": "success", **changes}


def status(status_id, state):
    return {"id": status_id, "context": GATE, "state": state, "sha": H}


def checks(*runs, statuses=(), total_count=None, evidence_id=20, last_rerun_id=None):
    return {"total_count": len(runs) if total_count is None else total_count,
            "check_runs": list(runs), "statuses": list(statuses),
            "evidence_id": evidence_id, "last_rerun_id": last_rerun_id}


def without(facts, key):
    return {name: value for name, value in facts.items() if name != key}


def review(review_id, state, commit_id=H, login=SECURITY):
    return {"id": review_id, "user": {"login": login}, "state": state, "commit_id": commit_id}


def merge(pr_facts=None, check_facts=None, verdicts=LENSES, reviews=None,
          security_reviewer=SECURITY, repo=PLATFORM):
    if reviews is None:
        reviews = [review(10, "APPROVED")]
    return fences.decide_merge(policy(), repo, H, pr_facts or pr(), check_facts or checks(check_run()),
                               list(verdicts), reviews, security_reviewer)


# The owner's rule, fault by fault: each value is one that a fact of that kind must never hold.
# REMOVED drops the key instead; an argument or a list item has no key, so it is never dropped.
REMOVED = object()
NOT_A_DICT = (REMOVED, None, "x", [], ["x"])
RECORD = (*NOT_A_DICT, {})
RECORDS = (REMOVED, None, "x", {}, [None], ["x"], [[]])
FLAG = (REMOVED, None, "", 0, 1, "x", [])
NUMBER = (REMOVED, None, "", True, 2.0, "7", [])
TEXT = (REMOVED, None, "", 5, ["x"])
SHA = (*TEXT, H.upper(), H[:39])


def nullable(faults):
    # None is a documented value of the fact, but its key must still be present.
    return tuple(value for value in faults if value is not None)


def faulted(facts, path, value):
    """A deep copy of facts with the fact at path set to value, or its key removed for REMOVED."""
    facts = copy.deepcopy(facts)
    *parents, key = path
    target = facts
    for step in parents:
        target = target[step]
    if value is REMOVED:
        del target[key]
    else:
        target[key] = copy.deepcopy(value)
    return facts


def not_refused(decide, facts, faults):
    """Inject each fault alone into a copy of facts, the decision's arguments, and describe every
    call that raised, allowed, or refused with a code other than the expected one."""
    failures = []
    for path, values, code in faults:
        for value in values:
            if value is REMOVED and (len(path) == 1 or isinstance(path[-1], int)):
                continue
            fault = f"{path} {'removed' if value is REMOVED else repr(value)}"
            try:
                d = decide(**faulted(facts, path, value))
            except Exception as error:
                failures.append(f"{fault}: raised {type(error).__name__}")
                continue
            if (d.allowed, d.code) != (False, code):
                failures.append(f"{fault}: gave {d.allowed}, {d.code} instead of False, {code}")
    return failures


def test_publish_refuses_when_the_head_moved():
    moved = (False, "head_moved")
    for changes in ({"review_head": OLD}, {"bound_head": OLD}):
        d = fences.decide_publish(approval(**changes), ledger(), remote())
        assert (d.allowed, d.code) == moved
    # No pull request recorded: the branch is absent (push) or already holds H (adopt).
    d = fences.decide_publish(approval(), ledger(), remote())
    assert (d.allowed, d.code) == (True, "push_and_create")
    d = fences.decide_publish(approval(), ledger(), remote(H))
    assert (d.allowed, d.code) == (True, "adopt_and_create")
    d = fences.decide_publish(approval(), ledger(), remote(OTHER))
    assert (d.allowed, d.code) == moved
    # The ledger holds another head without a fast-forward return, or GitHub is not at H.
    d = fences.decide_publish(approval(), ledger(7, OLD, "published"), remote(OLD, (7, OLD)))
    assert (d.allowed, d.code) == moved
    for branch, pull_head in ((H, OTHER), (OTHER, H)):
        d = fences.decide_publish(approval(), ledger(7, H, "published"), remote(branch, (7, pull_head)))
        assert (d.allowed, d.code) == moved
    # A fast-forward pushes only over the ledger's old head, with a lease on exactly that SHA.
    returned = ledger(7, OLD, "returned_for_changes", head_is_ancestor=True)
    d = fences.decide_publish(approval(), returned, remote(OLD, (7, OLD)))
    assert (d.allowed, d.code) == (True, "push_fast_forward") and OLD in d.detail
    d = fences.decide_publish(approval(), returned, remote(H, (7, H)))
    assert (d.allowed, d.code) == (True, "adopt_fast_forward")
    for branch, pull_head in ((OTHER, OLD), (OLD, OTHER)):
        d = fences.decide_publish(approval(), returned, remote(branch, (7, pull_head)))
        assert (d.allowed, d.code) == moved
    # T6: the pull request head and the ledger head both equal H, and the merge carries sha=H.
    d = merge()
    assert (d.allowed, d.code) == (True, "merge") and H in d.detail
    for changes in ({"head": OLD}, {"ledger_head": OLD}):
        d = merge(pr(**changes))
        assert (d.allowed, d.code) == moved
    d = merge(pr(state="closed", merged=True))
    assert (d.allowed, d.code) == (True, "already_merged")
    for changes in ({"state": "closed", "merged": True, "head": OLD}, {"state": "closed"}):
        d = merge(pr(**changes))
        assert (d.allowed, d.code) == (False, "pull_request_not_open")


def test_publish_refuses_a_second_pull_request():
    recorded = ledger(7, H, "published")
    d = fences.decide_publish(approval(), recorded, remote(H, (7, H)))
    assert (d.allowed, d.code) == (True, "already_published")
    for facts, pulls in ((ledger(), [(8, H)]), (recorded, [(7, H), (8, H)])):
        d = fences.decide_publish(approval(), facts, remote(H, *pulls))
        assert (d.allowed, d.code) == (False, "second_pull_request")
    d = fences.decide_publish(approval(), recorded, remote(H))
    assert (d.allowed, d.code) == (False, "pull_request_not_open")
    # Recorded at an unrelated head: returned for changes, but that head is not an ancestor of H.
    unrelated = ledger(7, OTHER, "returned_for_changes")
    d = fences.decide_publish(approval(), unrelated, remote(OTHER, (7, OTHER)))
    assert (d.allowed, d.code) == (False, "not_fast_forward")


def test_publish_refuses_a_head_not_approved_by_the_kernel():
    assert fences.decide_publish(approval(), ledger(), remote()).allowed
    upper = dict.fromkeys(("head_commit", "review_head", "bound_head"), H.upper())
    short = dict.fromkeys(("head_commit", "review_head", "bound_head"), H[:39])
    cases = [({"source_done": False}, "not_approved"),
             ({"approved_by_review_lane": False}, "not_approved"),
             (upper, "invalid_head"),
             (short, "invalid_head"),
             ({"reviewer": "worker-profile"}, "reviewer_not_independent"),
             ({"reviewer": None}, "reviewer_not_independent"),
             ({"implementer": ""}, "reviewer_not_independent"),
             ({"reopened_after_approval": True}, "reopened_after_approval"),
             ({"head_exists": False}, "head_unconfirmed"),
             ({"base_is_ancestor": False}, "head_unconfirmed"),
             ({"base": H}, "head_unconfirmed")]
    for changes, code in cases:
        d = fences.decide_publish(approval(**changes), ledger(), remote())
        assert (d.allowed, d.code) == (False, code), changes


def test_rerun_refuses_when_any_failed_test_is_off_the_flaky_list():
    for jobs in ([slice_job(3, 3, [FLAKY_46, UNLISTED])],
                 [slice_job(3, 3, [FLAKY_46]), slice_job(7, 7, [UNLISTED])]):
        d = rerun(jobs)
        assert (d.allowed, d.code, d.matches) == (False, "not_flaky", [])
        assert UNLISTED not in d.detail


def test_rerun_refuses_a_failed_job_that_names_no_test():
    assert rerun([slice_job(3, 3, [FLAKY_46, FLAKY_100])]).allowed
    for facts in (slice_job(3, 3, []), {**slice_job(3, 3, []), "failed_tests": None}):
        d = rerun([facts])
        assert (d.allowed, d.code) == (False, "no_failed_test")
    for facts in (slice_job(3, 3, [FLAKY_46], failed_count=2),
                  slice_job(3, 3, [FLAKY_46, FLAKY_100], failed_count=1),
                  slice_job(3, 3, [FLAKY_46, FLAKY_46]),
                  {**slice_job(3, 3, [FLAKY_46]), "failed_count": None}):
        d = rerun([facts])
        assert (d.allowed, d.code) == (False, "count_mismatch")


def test_rerun_refuses_a_second_rerun():
    flaky = slice_job(3, 3, [FLAKY_46])
    assert rerun([flaky], reruns=[OLD]).allowed
    d = rerun([flaky], reruns=[H])
    assert (d.allowed, d.code) == (False, "rerun_used")
    # The label-triggered "gh run rerun --failed" leaves run_attempt 2 on the jobs of H.
    for jobs in ([{**flaky, "run_attempt": 2}], [flaky, job(5, "Python lints", run_attempt=2)]):
        d = rerun(jobs)
        assert (d.allowed, d.code) == (False, "rerun_used")


def test_rerun_refuses_any_other_failed_job():
    flaky = slice_job(3, 3, [FLAKY_46])
    test_step = {"name": "Run tests (slice 3/12)", "conclusion": "failure"}
    others = [job(5, "Python lints", "failure"),
              {**flaky, "conclusion": "cancelled"},
              {**flaky, "conclusion": "timed_out"},
              slice_job(3, 3, [FLAKY_46], step="Install dependencies"),
              slice_job(3, 3, [FLAKY_46], step="Run tests (slice 4/12)"),
              {**flaky, "steps": [test_step, {"name": "Upload per-slice durations", "conclusion": "failure"}]},
              {**flaky, "steps": [{**test_step, "conclusion": "cancelled"}]}]
    for other in others:
        d = rerun([other])
        assert (d.allowed, d.code) == (False, "not_test_job"), other


def test_rerun_with_several_failed_jobs_is_all_or_nothing():
    eligible = [slice_job(3, 3, [FLAKY_46]), slice_job(7, 7, [FLAKY_100])]
    d = rerun(eligible + [job(1, "Python lints")])
    assert (d.allowed, d.code) == (True, "rerun")
    assert d.matches == [
        {"job_id": 3, "test": FLAKY_46, "list_entry": {"test": FLAKY_46, "finding": 46}},
        {"job_id": 7, "test": FLAKY_100, "list_entry": {"test": FLAKY_100, "finding": 100}}]
    for ineligible in (job(5, "Python lints", "failure"), slice_job(9, 9, [UNLISTED]), slice_job(9, 9, [])):
        d = rerun(eligible + [ineligible])
        assert (d.allowed, d.matches) == (False, [])


def test_rerun_gate_job_follows_the_owner_decision():
    # Beside eligible slices the gate is their summary: it neither stops the rerun nor is rerun.
    d = rerun([slice_job(3, 3, [FLAKY_46]), slice_job(7, 7, [FLAKY_100]), job(99, GATE, "failure")])
    assert (d.allowed, d.code) == (True, "rerun")
    assert {match["job_id"] for match in d.matches} == {3, 7}
    d = rerun([job(3, "Python tests / Run tests slice 3/12"), job(99, GATE, "failure")])
    assert (d.allowed, d.code, d.matches) == (False, "summary_failed_alone", [])


def test_lens_request_refuses_without_check_evidence():
    d = lens_request([evidence(1)])
    assert (d.allowed, d.code) == (True, "request")
    cases = [([], "no_evidence"),
             ([evidence(1, head=OLD)], "no_evidence"),
             ([evidence(1, "failure")], "checks_not_passing"),
             ([evidence(1, "pending")], "checks_not_passing"),
             ([evidence(1), rerun_record(2)], "evidence_before_rerun"),
             ([rerun_record(2), evidence(1)], "evidence_before_rerun"),
             ([evidence(1, reread_digest="f" * 64)], "evidence_unverified"),
             ([evidence(1, reread_digest=None)], "evidence_unverified"),
             ([evidence(1, digest=None, reread_digest=None)], "evidence_unverified")]
    for records, code in cases:
        d = lens_request(records)
        assert (d.allowed, d.code) == (False, code), records
    # Success evidence recorded after the rerun counts; another head's records never do.
    assert lens_request([evidence(1, "failure"), rerun_record(2), evidence(3)]).allowed
    assert lens_request([evidence(1), rerun_record(2, head=OLD)]).allowed
    # The other T4 fences: the pull request head, an independent reviewer, no second set of cards.
    d = lens_request([evidence(1)], pr_head=OLD)
    assert (d.allowed, d.code) == (False, "head_moved")
    for reviewer in ("worker-profile", None, ""):
        d = lens_request([evidence(1)], reviewer=reviewer)
        assert (d.allowed, d.code) == (False, "reviewer_not_independent")
    cards = [{"lens": "first-lens", "task_id": "t1", "head": H},
             {"lens": "second-lens", "task_id": "t2", "head": H}]
    d = lens_request([evidence(1)], cards=cards)
    assert (d.allowed, d.code, d.matches) == (True, "already_requested", cards)
    d = lens_request([evidence(1)], cards=[{**card, "head": OLD} for card in cards])
    assert (d.allowed, d.code) == (True, "request")


def test_merge_refuses_without_both_approvals_on_the_same_head():
    first, second = LENSES
    for verdicts in ([first], [first, first], [first, {**second, "head": OLD}]):
        d = merge(verdicts=verdicts)
        assert (d.allowed, d.code) == (False, "lens_approval_missing")
    d = merge(verdicts=[first, second, {**second, "verdict": "changes"}])
    assert (d.allowed, d.code) == (False, "changes_verdict")
    assert merge(verdicts=[first, second, {**second, "head": OLD, "verdict": "changes"}]).allowed
    # No mirror (owner change): the security reviewer's own latest decisive review approves exactly H.
    refused = [[],
               [review(10, "APPROVED", commit_id=OLD)],
               [review(10, "APPROVED", commit_id=OLD), review(11, "COMMENTED")],
               [review(10, "APPROVED"), review(11, "APPROVED", commit_id=OLD)],
               [review(10, "APPROVED", login="another-reviewer")],
               [{**review(10, "APPROVED"), "user": None}],
               [review(10, "APPROVED"), review(11, "CHANGES_REQUESTED")],
               [review(10, "DISMISSED")],
               [review(10, "COMMENTED")]]
    for reviews in refused:
        d = merge(reviews=reviews)
        assert (d.allowed, d.code) == (False, "security_approval_missing"), reviews
    for reviews in ([review(10, "APPROVED", commit_id=OLD), review(11, "APPROVED")],
                    [review(11, "APPROVED"), review(10, "CHANGES_REQUESTED")],
                    [review(10, "APPROVED"), review(11, "COMMENTED")]):
        assert merge(reviews=reviews).allowed, reviews
    for identity in (None, ""):
        d = merge(security_reviewer=identity)
        assert (d.allowed, d.code) == (False, "no_security_reviewer")


def test_merge_refuses_when_a_required_check_is_not_passing():
    cases = [(checks(check_run(conclusion="failure")), "check_failure"),
             (checks(check_run(status="in_progress", conclusion=None)), "check_pending"),
             (checks(), "check_missing"),
             (checks(check_run(name="Python lints")), "check_missing"),
             (checks(check_run(head_sha=OLD)), "check_stale"),
             (checks(check_run(conclusion="cancelled")), "check_infra"),
             (checks(check_run(conclusion="skipped")), "check_infra"),
             (checks(check_run(conclusion="neutral")), "check_infra"),
             (checks(check_run(), check_run(2, conclusion="failure")), "check_failure"),
             (checks(check_run(), statuses=[status(1, "failure")]), "check_failure"),
             (checks(check_run(), total_count=2), "incomplete_checks")]
    for facts, code in cases:
        d = merge(check_facts=facts)
        assert (d.allowed, d.code) == (False, code), facts
    # As in collect_acceptance, the latest legacy status can satisfy an unpinned check.
    assert merge(check_facts=checks(statuses=[status(1, "failure"), status(2, "success")])).allowed
    # The required checks are the policy's, per repository.
    d = merge(repo=WORKSPACE)
    assert (d.allowed, d.code) == (False, "check_missing")
    d = merge(repo="mtbitcr/unlisted-repo")
    assert (d.allowed, d.code) == (False, "unknown_repository")
    # T6 (e): GitHub reports the pull request as mergeable and clean.
    for changes in ({"mergeable": False}, {"mergeable": None},
                    {"mergeable_state": "blocked"}, {"mergeable_state": "unstable"}):
        d = merge(pr(**changes))
        assert (d.allowed, d.code) == (False, "not_mergeable")


def test_policy_file_loads_and_matches_exactly():
    text = Path(fences.__file__).with_name("kanban_delivery_policy.json").read_text(encoding="utf-8")
    loaded = fences.load_policy(text)
    assert loaded.required_checks == {PLATFORM: (GATE,), WORKSPACE: ("quality",)}
    assert loaded.flaky_tests == {PLATFORM: {FLAKY_46: 46, FLAKY_100: 100}, WORKSPACE: {}}
    seeded = json.loads(POLICY_TEXT)
    entry = seeded[PLATFORM]
    first = {"test": FLAKY_46, "finding": 46}
    invalid = [[seeded], {**seeded, "version": 1}, {"mtbitcr/*": entry},
               {PLATFORM: {**entry, "gate_job": GATE}},
               {PLATFORM: {**entry, "required_checks": []}},
               {PLATFORM: {**entry, "required_checks": [GATE, GATE]}},
               {PLATFORM: {**entry, "required_checks": ["Run tests slice */12"]}},
               {PLATFORM: {**entry, "flaky_tests": [{**first, "pattern": True}]}},
               {PLATFORM: {**entry, "flaky_tests": [{"test": FLAKY_46}]}},
               {PLATFORM: {**entry, "flaky_tests": [{**first, "finding": "46"}]}},
               {PLATFORM: {**entry, "flaky_tests": [{**first, "finding": 0}]}},
               {PLATFORM: {**entry, "flaky_tests": [first, {**first, "finding": 47}]}}]
    patterns = [FLAKY_46 + "*", FLAKY_46 + "[1]", FLAKY_46.replace("Delegation", "?elegation"),
                "tests/tools/*", "tests/tools/test_zombie_process_cleanup.py::*",
                "tests/tools/test_zombie_process_cleanup.py", FLAKY_46.rsplit("::", 1)[1], "re:" + FLAKY_46]
    invalid += [{PLATFORM: {**entry, "flaky_tests": [{"test": test, "finding": 46}]}} for test in patterns]
    for document in invalid:
        with pytest.raises(ValueError):
            fences.load_policy(json.dumps(document))
    body = json.dumps(seeded[WORKSPACE])
    with pytest.raises(ValueError):
        fences.load_policy(f'{{"{WORKSPACE}": {body}, "{WORKSPACE}": {body}}}')


def test_fences_have_no_io():
    tree = ast.parse(Path(fences.__file__).read_text(encoding="utf-8"))
    imports = {"json", "re"}
    from_imports = {"__future__": {"annotations"}, "dataclasses": {"dataclass", "field"},
                    "hermes_cli.kanban_pr_acceptance": {"_classify"}}
    # Builtins that do input or output, or run or reach code dynamically.
    forbidden = {"__builtins__", "__import__", "breakpoint", "compile", "delattr", "eval", "exec",
                 "exit", "getattr", "globals", "help", "input", "locals", "open", "print", "quit",
                 "setattr", "vars"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= imports
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and {alias.name for alias in node.names} <= from_imports.get(node.module, set())
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden, node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "json":
            assert node.attr == "loads", node.attr


def test_summary_check_failing_alone_stops_the_flow():
    d = rerun([job(3, "Python tests / Run tests slice 3/12"), job(5, "Python lints"), job(99, GATE, "failure")])
    assert (d.allowed, d.code, d.matches) == (False, "summary_failed_alone", [])
    d = merge(check_facts=checks(check_run(99, conclusion="failure")))
    assert (d.allowed, d.code) == (False, "check_failure")


def test_summary_check_with_only_listed_flaky_failures_is_a_summary():
    flaky = [slice_job(3, 3, [FLAKY_46]), slice_job(7, 7, [FLAKY_100])]
    d = rerun(flaky + [job(99, GATE, "failure")])
    assert (d.allowed, d.code) == (True, "rerun")
    assert 99 not in {match["job_id"] for match in d.matches}
    # The section 11 rerun rule still applies around the summary.
    for gate, reruns, code in ((job(99, GATE, "failure", run_attempt=2), (), "rerun_used"),
                               (job(99, GATE, "failure"), (H,), "rerun_used"),
                               (job(99, GATE, "failure", head_sha=OLD), (), "head_moved"),
                               (job(99, GATE, None, status="in_progress"), (), "not_completed")):
        d = rerun(flaky + [gate], reruns=reruns)
        assert (d.allowed, d.code) == (False, code)


def test_summary_check_with_any_other_failure_is_a_real_failure():
    for other, code in ((slice_job(3, 3, [UNLISTED]), "not_flaky"),
                        (slice_job(3, 3, [FLAKY_46, UNLISTED]), "not_flaky"),
                        (job(5, "Python lints", "failure"), "not_test_job"),
                        (slice_job(3, 3, []), "no_failed_test")):
        d = rerun([slice_job(7, 7, [FLAKY_100]), other, job(99, GATE, "failure")])
        assert (d.allowed, d.code, d.matches) == (False, code, [])
    # The gate stays failed on a real failure, so the merge is refused.
    d = merge(check_facts=checks(check_run(99, conclusion="failure")))
    assert (d.allowed, d.code) == (False, "check_failure")


def test_a_test_not_in_the_policy_is_never_flaky():
    near_misses = [FLAKY_46.upper(), FLAKY_46 + "_slow", "x" + FLAKY_46, FLAKY_46 + "[1]", FLAKY_46 + " ",
                   FLAKY_46.rsplit("::", 1)[0], "tests/tools/test_zombie_process_cleanup.py"]
    for test in near_misses:
        d = rerun([slice_job(3, 3, [test])])
        assert (d.allowed, d.code, d.matches) == (False, "not_flaky", [])
    # One repository's list never applies to another, and an unlisted repository has none.
    d = rerun([slice_job(3, 3, [FLAKY_46])], repo=WORKSPACE)
    assert (d.allowed, d.code) == (False, "not_flaky")
    d = rerun([slice_job(3, 3, [FLAKY_46])], repo="mtbitcr/unlisted-repo")
    assert (d.allowed, d.code) == (False, "unknown_repository")


def test_rerun_knows_test_slices_by_the_job_names_github_reports():
    # Finding 1: ci.yaml's "Python tests" job calls tests.yml, so the jobs API reports each slice as
    # "Python tests / Run tests slice N/12", while its test step keeps the name "Run tests (slice N/12)".
    live = slice_job(3, 3, [FLAKY_46])
    assert (live["name"], live["steps"][1]["name"]) == ("Python tests / Run tests slice 3/12", "Run tests (slice 3/12)")
    d = rerun([live, job(99, GATE, "failure")])
    assert (d.allowed, d.code) == (True, "rerun")
    assert d.matches == [{"job_id": 3, "test": FLAKY_46, "list_entry": {"test": FLAKY_46, "finding": 46}}]
    for n in range(1, 13):
        assert rerun([slice_job(n, n, [FLAKY_100])]).allowed, n
    # Every other name is an unknown job, the bare called-job name included.
    for name in ("Run tests slice 3/12", "Python tests / Generate slices", "OS-specific tests / Windows-only tests",
                 "Python tests / Run tests slice 13/12", "Python tests / Run tests slice 3/8"):
        d = rerun([{**live, "name": name}, job(99, GATE, "failure")])
        assert (d.allowed, d.code, d.matches) == (False, "not_test_job", []), name
    # Under its full name a slice still has to fail at its own test step and nowhere else.
    for step in ("Install dependencies", "Run tests (slice 4/12)", "Python tests / Run tests (slice 3/12)"):
        d = rerun([slice_job(3, 3, [FLAKY_46], step=step), job(99, GATE, "failure")])
        assert (d.allowed, d.code, d.matches) == (False, "not_test_job", []), step


def test_summary_check_counts_only_when_it_ended_in_failure():
    # Finding 2: only the summary job's ordinary failure summarises the jobs it waits on; any other
    # end refuses the rerun, beside an eligible slice or alone.
    flaky, passed = slice_job(3, 3, [FLAKY_46]), job(3, "Python tests / Run tests slice 3/12")
    for conclusion in ("cancelled", "timed_out", "action_required", "startup_failure", "stale", None, "unknown"):
        decisions = [rerun([other, job(99, GATE, conclusion)]) for other in (flaky, passed)]
        assert [(d.allowed, d.code, d.matches) for d in decisions] == [(False, "summary_not_failure", [])] * 2, conclusion
    d = rerun([flaky, job(99, GATE, "failure")])
    assert (d.allowed, d.code) == (True, "rerun")


def test_merge_follows_the_kernel_record_order():
    # Finding 3: kernel record ids order the facts, as in decide_lens_request; a larger id is later.
    first, second = LENSES
    # The last rerun (10), then the check evidence (20), then both approvals (30 and 31).
    d = merge(check_facts=checks(check_run(), evidence_id=20, last_rerun_id=10))
    assert (d.allowed, d.code) == (True, "merge")
    # The reviewer's two reproductions: approvals before the evidence, and evidence before the rerun.
    early = [{**verdict, "id": 10, "evidence_id": 5} for verdict in LENSES]
    d = merge(check_facts=checks(check_run(), evidence_id=30, last_rerun_id=20), verdicts=early)
    assert (d.allowed, d.code) == (False, "verdict_before_evidence")
    d = merge(check_facts=checks(check_run(), evidence_id=10, last_rerun_id=20),
              verdicts=[{**verdict, "id": 30} for verdict in LENSES])
    assert (d.allowed, d.code) == (False, "evidence_before_rerun")
    d = merge(check_facts=checks(check_run(), evidence_id=20, last_rerun_id=20))
    assert (d.allowed, d.code) == (False, "evidence_before_rerun")
    for verdicts in ([first, {**second, "id": 15}], [{**first, "id": 20}, second]):
        d = merge(verdicts=verdicts)
        assert (d.allowed, d.code) == (False, "verdict_before_evidence"), verdicts
    # Missing ordering facts, or ids that are not ints, prove no order; a bool is not an int.
    facts = checks(check_run())
    unproven = [without(facts, "evidence_id"), without(facts, "last_rerun_id")]
    unproven += [{**facts, "evidence_id": value} for value in (None, "20", 20.0, True)]
    unproven += [{**facts, "last_rerun_id": value} for value in ("10", 10.0, False, True)]
    for check_facts in unproven:
        d = merge(check_facts=check_facts)
        assert (d.allowed, d.code) == (False, "order_unproven"), check_facts
    for verdict in (without(second, "id"), {**second, "id": None}, {**second, "id": "31"}, {**second, "id": True}):
        d = merge(verdicts=[first, verdict])
        assert (d.allowed, d.code) == (False, "order_unproven"), verdict
    # A verdict for another head is not one for H, so it needs no place in H's order.
    assert merge(verdicts=[*LENSES, {**without(second, "id"), "head": OLD}]).allowed


def test_merge_refuses_a_check_on_another_head_whatever_its_name():
    # Finding 4: section 4 binds every check read to H, not only the checks the policy requires.
    lint_status = {"id": 5, "context": "Python lints", "state": "success", "sha": H}
    d = merge(check_facts=checks(check_run(), check_run(2, name="Python lints"), statuses=[lint_status]))
    assert (d.allowed, d.code) == (True, "merge")
    # The reviewer's reproduction first; a head is read as _classify reads it (head_sha, else sha).
    foreign = [checks(check_run(), check_run(2, name="Python lints", head_sha=OLD)),
               checks(check_run(), check_run(2, name="Python lints", head_sha=None)),
               checks(check_run(), without(check_run(2, name="Python lints"), "head_sha")),
               checks(check_run(), statuses=[{**lint_status, "sha": OLD}]),
               checks(check_run(), statuses=[without(lint_status, "sha")]),
               checks(statuses=[{**status(1, "success"), "sha": OLD}, status(2, "success")])]
    for facts in foreign:
        d = merge(check_facts=facts)
        assert (d.allowed, d.code) == (False, "check_stale"), facts
