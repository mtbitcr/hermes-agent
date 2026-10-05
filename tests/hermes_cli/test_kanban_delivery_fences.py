"""Delivery fence core: pure fence decisions over facts the caller supplies.

Tests 1 to 14 are plan section 8's, adapted as the owner directs: load_policy takes the
policy text, and the merge reads the security reviewer's own approval (no mirror). The
four tests after them show owner cases that tests 1 to 14 do not show in full, the next
four are regressions for the first independent review's findings 1 to 4, the next is
the regression for the latest review's finding: an unknown job's unnamed failed step, and the
last six are regressions for security findings 1 to 6 on pull request 140.
"""
import ast
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
# Owner rule 3: the summary check counts only with this one failed step, ci.yaml's "Evaluate job results".
EVALUATE = {"name": "Evaluate job results", "conclusion": "failure"}


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
          security_reviewer=SECURITY, repo=PLATFORM, rerun_attempt=None):
    if reviews is None:
        reviews = [review(10, "APPROVED")]
    return fences.decide_merge(policy(), repo, H, pr_facts or pr(), check_facts or checks(check_run()),
                               list(verdicts), reviews, security_reviewer, rerun_attempt)


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
    for facts, pulls in ((ledger(), [(8, OTHER)]), (recorded, [(7, H), (8, H)])):
        d = fences.decide_publish(approval(), facts, remote(H, *pulls))
        assert (d.allowed, d.code) == (False, "second_pull_request")
    d = fences.decide_publish(approval(), recorded, remote(H))
    assert (d.allowed, d.code) == (False, "pull_request_not_open")
    # Recorded at an unrelated head: returned for changes, but that head is not an ancestor of H.
    unrelated = ledger(7, OTHER, "returned_for_changes")
    d = fences.decide_publish(approval(), unrelated, remote(OTHER, (7, OTHER)))
    assert (d.allowed, d.code) == (False, "not_fast_forward")


def test_no_ledger_and_one_open_pull_at_h_is_adopted():
    # The crash window: the pull request was opened and the ledger was lost. With no ledger, the
    # one open pull request at H on the branch at H is recorded rather than refused.
    d = fences.decide_publish(approval(), ledger(), remote(H, (8, H)))
    assert (d.allowed, d.code) == (True, "adopt_pull_request")
    # Anything else open for the branch is still a second pull request.
    for branch, pulls in ((H, [(8, OTHER)]), (H, [(8, H), (9, H)]), (OTHER, [(8, H)]), (None, [(8, H)])):
        d = fences.decide_publish(approval(), ledger(), remote(branch, *pulls))
        assert (d.allowed, d.code) == (False, "second_pull_request"), (branch, pulls)
    # The approval half still answers first.
    d = fences.decide_publish(approval(reopened_after_approval=True), ledger(), remote(H, (8, H)))
    assert (d.allowed, d.code) == (False, "reopened_after_approval")


def test_publish_refuses_a_head_not_approved_by_the_kernel():
    assert fences.decide_publish(approval(), ledger(), remote()).allowed
    upper = dict.fromkeys(("head_commit", "review_head", "bound_head"), H.upper())
    short = dict.fromkeys(("head_commit", "review_head", "bound_head"), H[:39])
    # Owner rule 1: a malformed head and a None or "" identity are malformed facts, so invalid_fact.
    cases = [({"source_done": False}, "not_approved"),
             ({"approved_by_review_lane": False}, "not_approved"),
             (upper, "invalid_fact"),
             (short, "invalid_fact"),
             ({"reviewer": "worker-profile"}, "reviewer_not_independent"),
             ({"reviewer": None}, "invalid_fact"),
             ({"implementer": ""}, "invalid_fact"),
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
    # Owner rule 1: failed_count is a positive int, so the empty list keeps a count of 1, and a None
    # failed_tests or failed_count is a malformed fact.
    for facts, code in ((slice_job(3, 3, [], failed_count=1), "no_failed_test"),
                        ({**slice_job(3, 3, [], failed_count=1), "failed_tests": None}, "invalid_fact")):
        d = rerun([facts])
        assert (d.allowed, d.code) == (False, code)
    for facts, code in ((slice_job(3, 3, [FLAKY_46], failed_count=2), "count_mismatch"),
                        (slice_job(3, 3, [FLAKY_46, FLAKY_100], failed_count=1), "count_mismatch"),
                        (slice_job(3, 3, [FLAKY_46, FLAKY_46]), "count_mismatch"),
                        ({**slice_job(3, 3, [FLAKY_46]), "failed_count": None}, "invalid_fact")):
        d = rerun([facts])
        assert (d.allowed, d.code) == (False, code)


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
    # Owner rule 3: the gate is a summary only with its failed "Evaluate job results" step.
    d = rerun([slice_job(3, 3, [FLAKY_46]), slice_job(7, 7, [FLAKY_100]), job(99, GATE, "failure", steps=[EVALUATE])])
    assert (d.allowed, d.code) == (True, "rerun")
    assert {match["job_id"] for match in d.matches} == {3, 7}
    d = rerun([job(3, "Python tests / Run tests slice 3/12"), job(99, GATE, "failure", steps=[EVALUATE])])
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
             # Owner rule 1: a None digest is no 64-character hex digest, so a malformed fact.
             ([evidence(1, reread_digest=None)], "invalid_fact"),
             ([evidence(1, digest=None, reread_digest=None)], "invalid_fact")]
    for records, code in cases:
        d = lens_request(records)
        assert (d.allowed, d.code) == (False, code), records
    # Success evidence recorded after the rerun counts; another head's records never do.
    assert lens_request([evidence(1, "failure"), rerun_record(2), evidence(3)]).allowed
    assert lens_request([evidence(1), rerun_record(2, head=OLD)]).allowed
    # The other T4 fences: the pull request head, an independent reviewer, no second set of cards.
    d = lens_request([evidence(1)], pr_head=OLD)
    assert (d.allowed, d.code) == (False, "head_moved")
    # Owner rule 1: the documented None reviewer is missing, but "" is no identity, so a malformed fact.
    for reviewer, code in (("worker-profile", "reviewer_not_independent"), (None, "reviewer_not_independent"),
                           ("", "invalid_fact")):
        d = lens_request([evidence(1)], reviewer=reviewer)
        assert (d.allowed, d.code) == (False, code)
    cards = [{"lens": "first-lens", "task_id": "t1", "head": H},
             {"lens": "second-lens", "task_id": "t2", "head": H}]
    d = lens_request([evidence(1)], cards=cards)
    assert (d.allowed, d.code, d.matches) == (True, "already_requested", cards)
    d = lens_request([evidence(1)], cards=[{**card, "head": OLD} for card in cards])
    assert (d.allowed, d.code) == (True, "request")


def test_merge_refuses_without_both_approvals_on_the_same_head():
    first, second = LENSES
    # Owner rule 1: verdict ids are unique in their list, so each repeated verdict has its own id.
    for verdicts in ([first], [first, {**first, "id": 32}], [first, {**second, "head": OLD}]):
        d = merge(verdicts=verdicts)
        assert (d.allowed, d.code) == (False, "lens_approval_missing")
    d = merge(verdicts=[first, second, {**second, "id": 32, "verdict": "changes"}])
    assert (d.allowed, d.code) == (False, "changes_verdict")
    assert merge(verdicts=[first, second, {**second, "id": 32, "head": OLD, "verdict": "changes"}]).allowed
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
    # Owner rule 1: the documented None is missing, but "" is no identity, so a malformed fact.
    for identity, code in ((None, "no_security_reviewer"), ("", "invalid_fact")):
        d = merge(security_reviewer=identity)
        assert (d.allowed, d.code) == (False, code)


def test_merge_refuses_when_a_required_check_is_not_passing():
    cases = [(checks(check_run(conclusion="failure")), "check_failure"),
             (checks(check_run(status="in_progress", conclusion=None)), "check_pending"),
             # Owner rule 1: total_count is a positive int, so a snapshot with no check run is malformed.
             (checks(), "invalid_fact"),
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
    # As in collect_acceptance, the latest legacy status can satisfy an unpinned check. Owner rule 1:
    # total_count is a positive int, so the snapshot also holds one check run the policy does not require.
    assert merge(check_facts=checks(check_run(name="Python lints"), statuses=[status(1, "failure"), status(2, "success")])).allowed
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
    # Owner rule 3: the summary check failed at its "Evaluate job results" step.
    d = rerun([job(3, "Python tests / Run tests slice 3/12"), job(5, "Python lints"), job(99, GATE, "failure", steps=[EVALUATE])])
    assert (d.allowed, d.code, d.matches) == (False, "summary_failed_alone", [])
    d = merge(check_facts=checks(check_run(99, conclusion="failure")))
    assert (d.allowed, d.code) == (False, "check_failure")


def test_summary_check_with_only_listed_flaky_failures_is_a_summary():
    flaky = [slice_job(3, 3, [FLAKY_46]), slice_job(7, 7, [FLAKY_100])]
    # Owner rule 3: each failed summary check failed at its "Evaluate job results" step.
    d = rerun(flaky + [job(99, GATE, "failure", steps=[EVALUATE])])
    assert (d.allowed, d.code) == (True, "rerun")
    assert 99 not in {match["job_id"] for match in d.matches}
    # The section 11 rerun rule still applies around the summary.
    for gate, reruns, code in ((job(99, GATE, "failure", run_attempt=2, steps=[EVALUATE]), (), "rerun_used"),
                               (job(99, GATE, "failure", steps=[EVALUATE]), (H,), "rerun_used"),
                               (job(99, GATE, "failure", head_sha=OLD, steps=[EVALUATE]), (), "head_moved"),
                               (job(99, GATE, None, status="in_progress"), (), "not_completed")):
        d = rerun(flaky + [gate], reruns=reruns)
        assert (d.allowed, d.code) == (False, code)


def test_summary_check_with_any_other_failure_is_a_real_failure():
    for other, code in ((slice_job(3, 3, [UNLISTED]), "not_flaky"),
                        (slice_job(3, 3, [FLAKY_46, UNLISTED]), "not_flaky"),
                        (job(5, "Python lints", "failure"), "not_test_job"),
                        # Owner rule 1: failed_count is a positive int, so the empty list keeps a count of 1.
                        (slice_job(3, 3, [], failed_count=1), "no_failed_test")):
        # Owner rule 3: the summary check failed at its "Evaluate job results" step.
        d = rerun([slice_job(7, 7, [FLAKY_100]), other, job(99, GATE, "failure", steps=[EVALUATE])])
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
    # Owner rule 3: each summary check below failed at its "Evaluate job results" step.
    d = rerun([live, job(99, GATE, "failure", steps=[EVALUATE])])
    assert (d.allowed, d.code) == (True, "rerun")
    assert d.matches == [{"job_id": 3, "test": FLAKY_46, "list_entry": {"test": FLAKY_46, "finding": 46}}]
    for n in range(1, 13):
        assert rerun([slice_job(n, n, [FLAKY_100])]).allowed, n
    # Every other name is an unknown job, the bare called-job name included.
    for name in ("Run tests slice 3/12", "Python tests / Generate slices", "OS-specific tests / Windows-only tests",
                 "Python tests / Run tests slice 13/12", "Python tests / Run tests slice 3/8"):
        d = rerun([{**live, "name": name}, job(99, GATE, "failure", steps=[EVALUATE])])
        assert (d.allowed, d.code, d.matches) == (False, "not_test_job", []), name
    # Under its full name a slice still has to fail at its own test step and nowhere else.
    for step in ("Install dependencies", "Run tests (slice 4/12)", "Python tests / Run tests (slice 3/12)"):
        d = rerun([slice_job(3, 3, [FLAKY_46], step=step), job(99, GATE, "failure", steps=[EVALUATE])])
        assert (d.allowed, d.code, d.matches) == (False, "not_test_job", []), step


def test_summary_check_counts_only_when_it_ended_in_failure():
    # Finding 2: only the summary job's ordinary failure summarises the jobs it waits on; any other
    # end refuses the rerun, beside an eligible slice or alone.
    flaky, passed = slice_job(3, 3, [FLAKY_46]), job(3, "Python tests / Run tests slice 3/12")
    for conclusion in ("cancelled", "timed_out", "action_required", "startup_failure", "stale", None, "unknown"):
        decisions = [rerun([other, job(99, GATE, conclusion)]) for other in (flaky, passed)]
        assert [(d.allowed, d.code, d.matches) for d in decisions] == [(False, "summary_not_failure", [])] * 2, conclusion
    # Owner rule 3: the summary check failed at its "Evaluate job results" step.
    d = rerun([flaky, job(99, GATE, "failure", steps=[EVALUATE])])
    assert (d.allowed, d.code) == (True, "rerun")


def test_merge_follows_the_kernel_record_order():
    # Finding 3: kernel record ids order the facts, as in decide_lens_request; a larger id is later.
    first, second = LENSES
    # The last rerun (10), then the check evidence (20), then both approvals (30 and 31).
    d = merge(check_facts=checks(check_run(run_attempt=2), evidence_id=20, last_rerun_id=10), rerun_attempt=1)
    assert (d.allowed, d.code) == (True, "merge")
    # The reviewer's two reproductions: approvals before the evidence, and evidence before the rerun.
    # Owner rule 1: verdict ids are unique in their list, so the two approvals keep distinct ids.
    early = [{**verdict, "id": verdict["id"] - 20, "evidence_id": 5} for verdict in LENSES]
    d = merge(check_facts=checks(check_run(run_attempt=2), evidence_id=30, last_rerun_id=20), verdicts=early, rerun_attempt=1)
    assert (d.allowed, d.code) == (False, "verdict_before_evidence")
    d = merge(check_facts=checks(check_run(run_attempt=2), evidence_id=10, last_rerun_id=20), verdicts=LENSES, rerun_attempt=1)
    assert (d.allowed, d.code) == (False, "evidence_before_rerun")
    d = merge(check_facts=checks(check_run(run_attempt=2), evidence_id=20, last_rerun_id=20), rerun_attempt=1)
    assert (d.allowed, d.code) == (False, "evidence_before_rerun")
    for verdicts in ([first, {**second, "id": 15}], [{**first, "id": 20}, second]):
        d = merge(verdicts=verdicts)
        assert (d.allowed, d.code) == (False, "verdict_before_evidence"), verdicts
    # Missing ordering facts, or ids that are not ints, prove no order; a bool is not an int.
    # Owner rule 1: a missing or malformed record id is a malformed fact, so invalid_fact.
    facts = checks(check_run())
    unproven = [without(facts, "evidence_id"), without(facts, "last_rerun_id")]
    unproven += [{**facts, "evidence_id": value} for value in (None, "20", 20.0, True)]
    unproven += [{**facts, "last_rerun_id": value} for value in ("10", 10.0, False, True)]
    for check_facts in unproven:
        d = merge(check_facts=check_facts)
        assert (d.allowed, d.code) == (False, "invalid_fact"), check_facts
    for verdict in (without(second, "id"), {**second, "id": None}, {**second, "id": "31"}, {**second, "id": True}):
        d = merge(verdicts=[first, verdict])
        assert (d.allowed, d.code) == (False, "invalid_fact"), verdict
    # Owner rule 1: a verdict for another head is still a documented fact, so without its id it is malformed.
    d = merge(verdicts=[*LENSES, {**without(second, "id"), "head": OLD}])
    assert (d.allowed, d.code) == (False, "invalid_fact")


def test_merge_refuses_a_check_on_another_head_whatever_its_name():
    # Finding 4: section 4 binds every check read to H, not only the checks the policy requires.
    lint_status = {"id": 5, "context": "Python lints", "state": "success", "sha": H}
    d = merge(check_facts=checks(check_run(), check_run(2, name="Python lints"), statuses=[lint_status]))
    assert (d.allowed, d.code) == (True, "merge")
    # The reviewer's reproduction first; a check run's head is its head_sha, a status's its sha.
    # Owner rule 1: a None or missing head is a malformed fact, and total_count is a positive int, so
    # the status-only snapshot also holds one check run the policy does not require.
    foreign = [(checks(check_run(), check_run(2, name="Python lints", head_sha=OLD)), "check_stale"),
               (checks(check_run(), check_run(2, name="Python lints", head_sha=None)), "invalid_fact"),
               (checks(check_run(), without(check_run(2, name="Python lints"), "head_sha")), "invalid_fact"),
               (checks(check_run(), statuses=[{**lint_status, "sha": OLD}]), "check_stale"),
               (checks(check_run(), statuses=[without(lint_status, "sha")]), "invalid_fact"),
               (checks(check_run(3, name="Python lints"),
                       statuses=[{**status(1, "success"), "sha": OLD}, status(2, "success")]), "check_stale")]
    for facts, code in foreign:
        d = merge(check_facts=facts)
        assert (d.allowed, d.code) == (False, code), facts


@pytest.mark.parametrize("name", ["Python lints", "Run tests slice 3/12"])
def test_rerun_refuses_an_unknown_job_whose_failed_step_has_no_name(name):
    # The latest review's finding: a job off the allowlist is no test job even when its failed step
    # has no name, whether it is unrelated or the bare slice name, which the jobs API never reports
    # (it reports "Python tests / Run tests slice N/12"). The valid full-name control first.
    d = rerun([slice_job(3, 3, [FLAKY_46])])
    assert (d.allowed, d.code) == (True, "rerun")
    # The reviewer's reproduction.
    bad = slice_job(3, 3, [FLAKY_46])
    bad["name"] = name
    bad["steps"][1]["name"] = None
    d = rerun([bad])
    # Owner rule 1: a step name is a non-empty str, so a null one is a malformed fact.
    assert (d.allowed, d.code, d.matches) == (False, "invalid_fact", [])


def test_malformed_facts_refuse_as_invalid_fact():
    # Security finding 1: a fact without its documented type refuses with invalid_fact instead of passing
    # as truthy, falsy, equal or distinct. The reviewer's reproductions, fence by fence.
    for changes in ({"approved_by_review_lane": "false"}, {"reopened_after_approval": None}, {"base": None},
                    {"reviewer": 1, "implementer": 2}):
        d = fences.decide_publish(approval(**changes), ledger(), remote())
        assert (d.allowed, d.code) == (False, "invalid_fact"), changes
    flaky = slice_job(3, 3, [FLAKY_46])
    for head, jobs, reruns in ((H, [{**flaky, "run_attempt": True}], []),
                               (H, [{**flaky, "run_attempt": 1.0}], []),
                               (H, [flaky], {}),
                               (H, [{**flaky, "failed_tests": {FLAKY_46: None}, "failed_count": 1}], []),
                               (None, [{**flaky, "head_sha": None}], [])):
        d = fences.decide_rerun(policy(), PLATFORM, head, jobs, {"reruns": reruns})
        assert (d.allowed, d.code) == (False, "invalid_fact"), (head, jobs, reruns)
    d = fences.decide_lens_request(None, None, [evidence(1, head=None)], [], "reviewer-profile", "worker-profile")
    assert (d.allowed, d.code) == (False, "invalid_fact")
    null_heads = [{**verdict, "head": None} for verdict in LENSES]
    d = fences.decide_merge(policy(), PLATFORM, None, pr(head=None, ledger_head=None),
                            checks(check_run(head_sha=None)), null_heads,
                            [review(10, "APPROVED", commit_id=None)], SECURITY)
    assert (d.allowed, d.code) == (False, "invalid_fact")
    first, second = LENSES
    for d in (merge(verdicts=[{**first, "lens": None}, {**second, "lens": ""}]),
              merge(check_facts=checks(check_run(), total_count=True)),
              merge(reviews=[review(10, "APPROVED", login=7)], security_reviewer=7)):
        assert (d.allowed, d.code) == (False, "invalid_fact")
    # Card review finding 1: every documented fact is checked once, at the start of its fence. The reviewer's
    # reproductions, then the malformed facts the first pass kept under other codes or allowed.
    upper = dict.fromkeys(("head_commit", "review_head", "bound_head"), H.upper())
    reproductions = [
        fences.decide_publish(approval(), ledger(7, OLD, "returned_for_changes", "false"), remote(OLD, (7, OLD))),
        rerun([{**slice_job(3, 3, [FLAKY_46]), "id": None}]),
        merge(pr_facts=pr(merged="false")),
        merge(check_facts=checks(check_run(), evidence_id=0)),
        merge(check_facts=checks(check_run(), last_rerun_id=-1)),
        merge(verdicts=[{**v, "id": 30} for v in LENSES]),
        fences.decide_rerun(policy(), PLATFORM, H, (flaky,), {"reruns": []}),
        fences.decide_merge(policy(), PLATFORM, H, pr(), {**checks(check_run()), "check_runs": (check_run(),)},
                            list(LENSES), [review(10, "APPROVED")], SECURITY),
        fences.decide_lens_request(H, H, (evidence(1),), [], "reviewer-profile", "worker-profile"),
        fences.decide_publish(approval(reviewer=""), ledger(), remote()),
        fences.decide_publish(approval(implementer=""), ledger(), remote()),
        lens_request([evidence(1)], reviewer=""),
        merge(security_reviewer=""),
        fences.decide_publish(approval(**upper), ledger(), remote()),
        rerun([slice_job(3, 3, [FLAKY_46], failed_count=0)]),
        rerun([{**flaky, "failed_count": None}]),
        rerun([slice_job(3, 3, [])]),
        merge(check_facts=checks(statuses=[status(1, "failure"), status(2, "success")])),
        merge(verdicts=[first, first]),
        merge(verdicts=[first, second, {**second, "verdict": "changes"}]),
        merge(check_facts=checks(check_run(), evidence_id=30, last_rerun_id=20),
              verdicts=[{**verdict, "id": 10} for verdict in LENSES])]
    failures = [(index, d.allowed, d.code) for index, d in enumerate(reproductions)
                if (d.allowed, d.code) != (False, "invalid_fact")]
    # One malformed value for every documented fact, one fault per case: each refuses with invalid_fact and its
    # detail names that fact. Every base is allowed, so a fact left unchecked shows.
    filed, branch = ledger(7, OLD, "returned_for_changes", True), remote(OLD, (7, OLD))
    publish = {"approval": approval(), "ledger": filed, "remote": branch}
    malformed = {"source_done": "true", "approved_by_review_lane": 1, "head_commit": H.upper(), "review_head": None,
                 "bound_head": H[:39], "reviewer": "", "implementer": None, "reopened_after_approval": 0,
                 "head_exists": None, "base": BASE.upper(), "base_is_ancestor": "false"}
    publish_faults = [(f"approval.{key}", {"approval": approval(**{key: value})}) for key, value in malformed.items()]
    publish_faults += [("approval", {"approval": list(approval().items())}),
                       ("approval.base", {"approval": without(approval(), "base")}),
                       ("ledger", {"ledger": None}),
                       ("ledger.pull_request", {"ledger": {**filed, "pull_request": "7"}}),
                       ("ledger.head", {"ledger": {**filed, "head": None}}),
                       ("ledger.state", {"ledger": {**filed, "state": ""}}),
                       ("ledger.head_is_ancestor", {"ledger": without(filed, "head_is_ancestor")}),
                       ("remote", {"remote": (OLD, [])}),
                       ("remote.branch_head", {"remote": {**branch, "branch_head": "main"}}),
                       ("remote.open_pulls", {"remote": {**branch, "open_pulls": tuple(branch["open_pulls"])}}),
                       ("remote.open_pulls[0]", {"remote": {**branch, "open_pulls": [(7, OLD)]}}),
                       ("remote.open_pulls[0].number", {"remote": remote(OLD, (True, OLD))}),
                       ("remote.open_pulls[1].number", {"remote": remote(OLD, (7, OLD), (7, OLD))}),
                       ("remote.open_pulls[0].head", {"remote": remote(OLD, (7, OLD[:39]))})]
    summary = job(99, GATE, "failure", steps=[EVALUATE])
    rerun_base = {"policy": policy(), "repo": PLATFORM, "head": H, "jobs": [flaky, summary], "ledger": {"reruns": [OLD]}}
    rerun_faults = [("repo", {"repo": None}),
                    ("head", {"head": H.upper()}),
                    ("jobs", {"jobs": (flaky, summary)}),
                    ("jobs[0]", {"jobs": [list(flaky.items()), summary]}),
                    ("jobs[0].id", {"jobs": [{**flaky, "id": "3"}, summary]}),
                    ("jobs[1].id", {"jobs": [flaky, {**summary, "id": 3}]}),
                    ("jobs[0].name", {"jobs": [{**flaky, "name": None}, summary]}),
                    ("jobs[0].head_sha", {"jobs": [{**flaky, "head_sha": None}, summary]}),
                    ("jobs[0].status", {"jobs": [{**flaky, "status": ""}, summary]}),
                    ("jobs[0].conclusion", {"jobs": [{**flaky, "conclusion": 0}, summary]}),
                    ("jobs[0].run_attempt", {"jobs": [{**flaky, "run_attempt": 1.0}, summary]}),
                    ("jobs[0].steps", {"jobs": [without(flaky, "steps"), summary]}),
                    ("jobs[1].steps", {"jobs": [flaky, {**summary, "steps": (EVALUATE,)}]}),
                    ("jobs[1].steps[0]", {"jobs": [flaky, {**summary, "steps": [("Evaluate job results", "failure")]}]}),
                    ("jobs[1].steps[0].name", {"jobs": [flaky, {**summary, "steps": [{**EVALUATE, "name": ""}]}]}),
                    ("jobs[1].steps[0].conclusion", {"jobs": [flaky, {**summary, "steps": [{**EVALUATE, "conclusion": 1}]}]}),
                    ("jobs[0].failed_tests", {"jobs": [{**flaky, "failed_tests": (FLAKY_46,)}, summary]}),
                    ("jobs[0].failed_tests", {"jobs": [{**flaky, "failed_tests": [None]}, summary]}),
                    ("jobs[0].failed_count", {"jobs": [{**flaky, "failed_count": True}, summary]}),
                    ("jobs[0].failed_count", {"jobs": [without(flaky, "failed_count"), summary]}),
                    ("ledger", {"ledger": [OLD]}),
                    ("ledger.reruns", {"ledger": {}}),
                    ("ledger.reruns", {"ledger": {"reruns": (OLD,)}}),
                    ("ledger.reruns[0]", {"ledger": {"reruns": [OLD.upper()]}})]
    card = {"lens": "first-lens", "task_id": "t1", "head": OLD}
    lens_base = {"head": H, "pr_head": H, "evidence": [rerun_record(1), evidence(2)], "existing_lens_cards": [card],
                 "reviewer": "reviewer-profile", "implementer": "worker-profile"}
    lens_faults = [("head", {"head": None}),
                   ("pr_head", {"pr_head": H[:39]}),
                   ("evidence", {"evidence": (rerun_record(1), evidence(2))}),
                   ("evidence[1]", {"evidence": [rerun_record(1), list(evidence(2).items())]}),
                   ("evidence[0].id", {"evidence": [rerun_record(None), evidence(2)]}),
                   ("evidence[1].id", {"evidence": [rerun_record(2), evidence(2)]}),
                   ("evidence[0].kind", {"evidence": [{**rerun_record(1), "kind": None}, evidence(2)]}),
                   ("evidence[0].head", {"evidence": [rerun_record(1, head=None), evidence(2)]}),
                   ("evidence[1].overall", {"evidence": [rerun_record(1), evidence(2, None)]}),
                   ("evidence[1].digest", {"evidence": [rerun_record(1), without(evidence(2), "digest")]}),
                   ("evidence[1].reread_digest", {"evidence": [rerun_record(1), evidence(2, reread_digest=DIGEST[1:])]}),
                   ("existing_lens_cards", {"existing_lens_cards": (card,)}),
                   ("existing_lens_cards[0]", {"existing_lens_cards": [list(card.values())]}),
                   ("existing_lens_cards[0].lens", {"existing_lens_cards": [{**card, "lens": ""}]}),
                   ("existing_lens_cards[0].task_id", {"existing_lens_cards": [{**card, "task_id": 1}]}),
                   ("existing_lens_cards[0].head", {"existing_lens_cards": [without(card, "head")]}),
                   ("reviewer", {"reviewer": 7}),
                   ("implementer", {"implementer": ""})]
    run, lint, gate_status = check_run(run_attempt=2), check_run(2, name="Python lints"), status(5, "success")
    snapshot = checks(run, lint, statuses=[gate_status], last_rerun_id=10)
    merge_base = {"policy": policy(), "repo": PLATFORM, "head": H, "pr": pr(), "checks": snapshot, "verdicts": list(LENSES),
                  "github_reviews": [review(10, "APPROVED")], "security_reviewer": SECURITY, "rerun_attempt": 1}
    merge_faults = [("repo", {"repo": None}),
                    ("head", {"head": H[:39]}),
                    ("pr", {"pr": list(pr().items())}),
                    ("pr.state", {"pr": pr(state=None)}),
                    ("pr.merged", {"pr": pr(merged=0)}),
                    ("pr.head", {"pr": pr(head=None)}),
                    ("pr.ledger_head", {"pr": without(pr(), "ledger_head")}),
                    ("pr.mergeable", {"pr": pr(mergeable="true")}),
                    ("pr.mergeable_state", {"pr": pr(mergeable_state="")}),
                    ("checks", {"checks": None}),
                    ("checks.total_count", {"checks": {**snapshot, "total_count": "2"}}),
                    ("checks.evidence_id", {"checks": without(snapshot, "evidence_id")}),
                    ("checks.last_rerun_id", {"checks": {**snapshot, "last_rerun_id": 0}}),
                    ("checks.check_runs", {"checks": {**snapshot, "check_runs": (run, lint)}}),
                    ("checks.check_runs[1]", {"checks": {**snapshot, "check_runs": [run, None]}}),
                    ("checks.check_runs[1].id", {"checks": {**snapshot, "check_runs": [run, {**lint, "id": 1}]}}),
                    ("checks.check_runs[1].name", {"checks": {**snapshot, "check_runs": [run, {**lint, "name": ""}]}}),
                    ("checks.check_runs[1].head_sha", {"checks": {**snapshot, "check_runs": [run, {**lint, "head_sha": None}]}}),
                    ("checks.check_runs[1].status", {"checks": {**snapshot, "check_runs": [run, without(lint, "status")]}}),
                    ("checks.check_runs[1].conclusion", {"checks": {**snapshot, "check_runs": [run, {**lint, "conclusion": False}]}}),
                    ("checks.check_runs[0].run_attempt", {"checks": {**snapshot, "check_runs": [check_run(run_attempt="2"), lint]}}),
                    ("checks.statuses", {"checks": {**snapshot, "statuses": None}}),
                    ("checks.statuses[0]", {"checks": {**snapshot, "statuses": [[5, GATE, "success", H]]}}),
                    ("checks.statuses[0].id", {"checks": {**snapshot, "statuses": [status(None, "success")]}}),
                    ("checks.statuses[1].id", {"checks": {**snapshot, "statuses": [gate_status, gate_status]}}),
                    ("checks.statuses[0].context", {"checks": {**snapshot, "statuses": [{**gate_status, "context": None}]}}),
                    ("checks.statuses[0].state", {"checks": {**snapshot, "statuses": [status(5, "")]}}),
                    ("checks.statuses[0].sha", {"checks": {**snapshot, "statuses": [{**gate_status, "sha": H.upper()}]}}),
                    ("verdicts", {"verdicts": LENSES}),
                    ("verdicts[1]", {"verdicts": [first, None]}),
                    ("verdicts[0].id", {"verdicts": [without(first, "id"), second]}),
                    ("verdicts[1].id", {"verdicts": [first, {**second, "id": 30}]}),
                    ("verdicts[0].lens", {"verdicts": [{**first, "lens": None}, second]}),
                    ("verdicts[0].head", {"verdicts": [{**first, "head": None}, second]}),
                    ("verdicts[1].verdict", {"verdicts": [first, {**second, "verdict": ""}]}),
                    ("github_reviews", {"github_reviews": (review(10, "APPROVED"),)}),
                    ("github_reviews[0]", {"github_reviews": [None]}),
                    ("github_reviews[0].id", {"github_reviews": [review(-10, "APPROVED")]}),
                    ("github_reviews[1].id", {"github_reviews": [review(10, "APPROVED"), review(10, "COMMENTED")]}),
                    ("github_reviews[0].user", {"github_reviews": [{**review(10, "APPROVED"), "user": SECURITY}]}),
                    ("github_reviews[0].user.login", {"github_reviews": [review(10, "APPROVED", login=None)]}),
                    ("github_reviews[0].state", {"github_reviews": [review(10, None)]}),
                    ("github_reviews[0].commit_id", {"github_reviews": [review(10, "APPROVED", commit_id=H[:7])]}),
                    ("security_reviewer", {"security_reviewer": [SECURITY]}),
                    ("rerun_attempt", {"rerun_attempt": "1"})]
    for fence, base, faults in ((fences.decide_publish, publish, publish_faults),
                                (fences.decide_rerun, rerun_base, rerun_faults),
                                (fences.decide_lens_request, lens_base, lens_faults),
                                (fences.decide_merge, merge_base, merge_faults)):
        assert fence(**base).allowed, fence.__name__
        for fact, changes in faults:
            try:
                d = fence(**{**base, **changes})
                outcome = (d.allowed, d.code, d.detail.partition(" ")[0])
            except Exception as error:  # a fence that raises on a malformed fact decides nothing
                outcome = type(error).__name__
            if outcome != (False, "invalid_fact", fact):
                failures.append((fence.__name__, fact, outcome))
    # Controls: each documented None decides as before.
    in_progress = job(5, "Python lints", None, status="in_progress", steps=[{"name": "Set up job", "conclusion": None}])
    controls = [(fences.decide_publish(approval(), ledger(), remote()), (True, "push_and_create")),
                (fences.decide_publish(approval(), ledger(), remote(None, (8, H))), (False, "second_pull_request")),
                (rerun([flaky, in_progress]), (False, "not_completed")),
                (lens_request([evidence(1)], reviewer=None), (False, "reviewer_not_independent")),
                (merge(), (True, "merge")),
                (fences.decide_merge(**{**merge_base, "rerun_attempt": None,
                                        "checks": {**snapshot, "last_rerun_id": None}}), (True, "merge")),
                (merge(check_facts=checks(check_run(status="in_progress", conclusion=None))), (False, "check_pending")),
                (merge(pr(mergeable=None)), (False, "not_mergeable")),
                (merge(reviews=[{**review(9, "APPROVED"), "user": None}, review(10, "APPROVED")]), (True, "merge")),
                (merge(security_reviewer=None), (False, "no_security_reviewer"))]
    failures += [("control", index, d.allowed, d.code) for index, (d, expected) in enumerate(controls)
                 if (d.allowed, d.code) != expected]
    assert failures == []


def test_latest_record_selection_needs_unique_int_ids():
    # Security finding 2: the latest evidence, review or status is chosen by id only among unique positive
    # int ids; a str id ("9" sorts after "10") or a tie refuses with invalid_fact.
    for records in ([evidence("9"), rerun_record("10")], [evidence(5), rerun_record(5)]):
        d = lens_request(records)
        assert (d.allowed, d.code) == (False, "invalid_fact"), records
    d = merge(reviews=[review("9", "APPROVED"), review("10", "CHANGES_REQUESTED")])
    assert (d.allowed, d.code) == (False, "invalid_fact")
    # Owner rule 1: total_count is a positive int, so the snapshot holds one check run the policy does not require.
    d = merge(check_facts=checks(check_run(name="Python lints"), statuses=[status("9", "success"), status("10", "failure")]))
    assert (d.allowed, d.code) == (False, "invalid_fact")


def test_lens_request_needs_sha256_digests():
    # Security finding 3: equal digests prove a re-read only as two 64-character lowercase hex SHA-256 strings.
    # Owner rule 1: a digest that is not 64-character lowercase hex is a malformed fact.
    for digest in (1, "not-a-digest", DIGEST.upper()):
        d = lens_request([evidence(1, digest=digest, reread_digest=digest)])
        assert (d.allowed, d.code) == (False, "invalid_fact"), digest


def test_check_kind_comes_from_its_list():
    # Security finding 4: a check run is classified as a run and a legacy status as a status, by the list
    # each came in, whatever other fields it carries.
    # Owner rule 1: a check run without its conclusion key is a malformed fact.
    in_progress = without(check_run(status="in_progress", state="success"), "conclusion")
    d = merge(check_facts=checks(in_progress))
    assert (d.allowed, d.code) == (False, "invalid_fact")
    # Owner rule 1: total_count is a positive int, so the snapshot holds one check run the policy does not require.
    contradictory = {**status(1, "failure"), "conclusion": "success", "status": "completed"}
    d = merge(check_facts=checks(check_run(2, name="Python lints"), statuses=[contradictory]))
    assert (d.allowed, d.code) == (False, "check_failure")


def test_summary_check_is_a_summary_only_when_its_evaluate_step_failed():
    # Security finding 5: failed at "Set up job", the summary check is one more failed job and refuses the
    # rerun; failed at its "Evaluate job results" step it still summarises the eligible slices.
    flaky = slice_job(3, 3, [FLAKY_46])
    setup = job(99, GATE, "failure", steps=[{"name": "Set up job", "conclusion": "failure"}])
    d = rerun([flaky, setup])
    assert (d.allowed, d.code, d.matches) == (False, "not_test_job", [])
    # Card review finding 2: with no failed step, only a successful "Set up job" or a successful "Evaluate job
    # results", it proves no failure at its evaluation either, so it is one more failed job.
    decisions = [rerun([flaky, job(99, GATE, "failure", steps=steps)])
                 for steps in ([], [{"name": "Set up job", "conclusion": "success"}], [{**EVALUATE, "conclusion": "success"}])]
    assert [(d.allowed, d.code, d.matches) for d in decisions] == [(False, "not_test_job", [])] * 3
    evaluate = job(99, GATE, "failure", steps=[{"name": "Set up job", "conclusion": "success"},
                                               {"name": "Evaluate job results", "conclusion": "failure"}])
    d = rerun([flaky, evaluate])
    assert (d.allowed, d.code) == (True, "rerun")


def test_merge_after_a_rerun_needs_required_checks_from_a_later_attempt():
    # Security finding 6: after H's rerun the caller passes the rerun attempt number; a required check run
    # then counts only from a later attempt, so an old success re-read under later evidence is stale.
    for run, outcome in ((check_run(run_attempt=1), (False, "check_stale")),
                         (check_run(), (False, "invalid_fact")),
                         (check_run(run_attempt=True), (False, "invalid_fact")),
                         (check_run(run_attempt=2), (True, "merge"))):
        facts = checks(run, evidence_id=20, last_rerun_id=10)
        d = fences.decide_merge(policy(), PLATFORM, H, pr(), facts, list(LENSES), [review(10, "APPROVED")],
                                SECURITY, rerun_attempt=1)
        assert (d.allowed, d.code) == outcome, run


def test_merge_after_a_rerun_refuses_without_the_rerun_attempt():
    # After H's rerun the attempt number is required: without it an old success could count again.
    facts = checks(check_run(run_attempt=2), evidence_id=20, last_rerun_id=10)
    d = fences.decide_merge(policy(), PLATFORM, H, pr(), facts, list(LENSES), [review(10, "APPROVED")], SECURITY)
    assert (d.allowed, d.code) == (False, "invalid_fact")
    assert "rerun_attempt" in d.detail


def test_merge_after_a_rerun_needs_a_check_run_for_every_required_check():
    # Second security review of PR 140: a legacy status carries no attempt, so after a rerun a required
    # check counts only through a check run from a later attempt.
    lint = check_run(2, name="Python lints", run_attempt=2)
    facts = checks(lint, statuses=[status(5, "success")], evidence_id=20, last_rerun_id=10)
    d = fences.decide_merge(policy(), PLATFORM, H, pr(), facts, list(LENSES), [review(10, "APPROVED")],
                            SECURITY, rerun_attempt=1)
    assert (d.allowed, d.code) == (False, "check_stale")
    # Control: without a rerun the legacy status still decides as before.
    facts = checks(check_run(2, name="Python lints"), statuses=[status(5, "success")], evidence_id=20)
    d = fences.decide_merge(policy(), PLATFORM, H, pr(), facts, list(LENSES), [review(10, "APPROVED")], SECURITY)
    assert (d.allowed, d.code) == (True, "merge")
