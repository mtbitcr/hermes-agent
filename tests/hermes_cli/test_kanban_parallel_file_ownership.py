"""Bounded parallel worktree execution: ownership locks and git receipts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _project_task(conn, *, title: str, owned_paths):
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="worker",
        workspace_kind="worktree",
        owned_paths=owned_paths,
    )
    conn.execute(
        "UPDATE tasks SET project_id = 'project-1' WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    return task_id


def test_owned_path_contract_is_literal_canonical_and_bounded(tmp_path):
    assert kb.normalize_owned_paths(None) is None
    assert kb.normalize_owned_paths([]) == []
    assert kb.normalize_owned_paths(["src/a", "tests/a", "src/a"]) == [
        "src/a",
        "tests/a",
    ]
    assert kb.normalize_owned_paths(["."]) == ["."]

    for invalid in [
        "src/a",
        ["/etc"],
        ["C:/repo"],
        ["../src"],
        ["src/*"],
        [".git/config"],
        [".", "src"],
    ]:
        with pytest.raises(ValueError):
            kb.normalize_owned_paths(invalid)

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError, match="requires a mutating owned_paths"):
            kb.create_task(
                conn,
                title="invalid integration",
                owned_paths=[],
                integrates_parent_heads=True,
            )
        with pytest.raises(ValueError, match="mutating owned_paths require"):
            kb.create_task(
                conn,
                title="unisolated mutator",
                workspace_kind="scratch",
                owned_paths=["src/owned"],
            )
        with pytest.raises(ValueError, match="read-only owned_paths require"):
            kb.create_task(
                conn,
                title="unverifiable reader",
                workspace_kind="dir",
                workspace_path=str(tmp_path),
                owned_paths=[],
            )
    finally:
        conn.close()


def test_claim_allows_only_disjoint_or_read_only_project_work(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        first = _project_task(conn, title="first", owned_paths=["src/a"])
        disjoint = _project_task(conn, title="disjoint", owned_paths=["src/b"])
        overlap = _project_task(conn, title="overlap", owned_paths=["src/a/file.py"])
        case_alias = _project_task(conn, title="case alias", owned_paths=["SRC/A"])
        legacy = _project_task(conn, title="legacy", owned_paths=None)
        reader = _project_task(conn, title="reader", owned_paths=[])
        unlinked = kb.create_task(
            conn,
            title="unlinked mutator",
            assignee="worker",
            workspace_kind="worktree",
            owned_paths=["src/unlinked"],
        )
        local_mutator = kb.create_task(
            conn,
            title="non-isolated mutator",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        conn.execute(
            "UPDATE tasks SET project_id = 'project-1', owned_paths = '[\"src/z\"]' "
            "WHERE id = ?",
            (local_mutator,),
        )
        conn.commit()

        assert kb.claim_task(conn, first, claimer="first") is not None
        assert kb.claim_task(conn, disjoint, claimer="disjoint") is not None
        assert kb.claim_task(conn, overlap, claimer="overlap") is None
        assert kb.file_scope_conflicts(conn, overlap) == [first]
        assert kb.claim_task(conn, case_alias, claimer="case-alias") is None
        assert kb.file_scope_conflicts(conn, case_alias) == [first]
        assert kb.claim_task(conn, legacy, claimer="legacy") is None
        assert set(kb.file_scope_conflicts(conn, legacy)) == {first, disjoint}
        assert kb.claim_task(conn, reader, claimer="reader") is not None
        assert kb.claim_task(conn, local_mutator, claimer="local") is None
        assert kb.claim_task(conn, unlinked, claimer="unlinked") is None
        assert set(kb.file_scope_conflicts(conn, unlinked)) == {
            first,
            disjoint,
        }

        events = kb.list_events(conn, overlap)
        assert [event.kind for event in events].count("claim_deferred_file_scope") == 1
        # Same deferral within the throttle window is still rejected but does
        # not flood the audit log on every dispatcher tick.
        assert kb.claim_task(conn, overlap, claimer="overlap-2") is None
        events = kb.list_events(conn, overlap)
        assert [event.kind for event in events].count("claim_deferred_file_scope") == 1
    finally:
        conn.close()


def test_dispatch_reports_scope_deferral_without_dropping_task(tmp_path, monkeypatch):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        first = _project_task(conn, title="running", owned_paths=["src/a"])
        overlap = _project_task(conn, title="waiting", owned_paths=["src/a/file.py"])
        disjoint = _project_task(conn, title="parallel", owned_paths=["src/ab"])
        assert kb.claim_task(conn, first, claimer="first") is not None

        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
        result = kb.dispatch_once(conn, dry_run=True)

        assert overlap not in {task_id for task_id, _, _ in result.spawned}
        assert (overlap, [first]) in result.skipped_file_scope_conflict
        assert disjoint in {task_id for task_id, _, _ in result.spawned}
        assert kb.get_task(conn, overlap).status == "ready"
        assert kb.claim_task(conn, disjoint, claimer="disjoint") is not None
        assert kb.has_spawnable_ready(conn) is False
    finally:
        conn.close()


def test_duplicate_projects_on_same_repo_still_contend(tmp_path):
    repo = tmp_path / "shared-repo"
    other_repo = tmp_path / "other-repo"
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        first = _project_task(conn, title="first project", owned_paths=["src/shared"])
        duplicate = _project_task(
            conn, title="duplicate project", owned_paths=["src/shared/child"]
        )
        independent = _project_task(
            conn, title="independent project", owned_paths=["src/shared"]
        )
        conn.execute(
            "UPDATE tasks SET workspace_path=? WHERE id=?",
            (str(repo / ".worktrees" / first), first),
        )
        conn.execute(
            "UPDATE tasks SET project_id='project-2', workspace_path=? WHERE id=?",
            (str(repo / ".worktrees" / duplicate), duplicate),
        )
        conn.execute(
            "UPDATE tasks SET project_id='project-3', workspace_path=? WHERE id=?",
            (str(other_repo / ".worktrees" / independent), independent),
        )
        conn.commit()

        assert kb.claim_task(conn, first, claimer="first") is not None
        assert kb.claim_task(conn, duplicate, claimer="duplicate") is None
        assert kb.file_scope_conflicts(conn, duplicate) == [first]
        assert kb.claim_task(conn, independent, claimer="independent") is not None
    finally:
        conn.close()


def _materialize(conn, task_id: str) -> Path:
    claimed = kb.claim_task(conn, task_id, claimer=f"claim-{task_id}")
    assert claimed is not None
    workspace, branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, workspace)
    kb.set_branch_name(conn, task_id, branch)
    kb.record_worktree_base(conn, task_id, workspace)
    return workspace


def _new_file_patch(path: str, content: str) -> bytes:
    lines = content.splitlines(keepends=True)
    body = "".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    ).encode("utf-8")


def _attach_patch(conn, task_id: str, root: Path, path: str, content: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    return kb.store_attachment_bytes(
        conn,
        task_id,
        "sandbox-result.patch",
        _new_file_patch(path, content),
        content_type="text/x-diff",
        uploaded_by="agent",
    )


def test_cross_profile_child_inherits_project_repo_without_sharing_worktree(tmp_path):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(
            conn,
            title="technical coordinator",
            assignee="planner",
            workspace_kind="worktree",
            owned_paths=[],
        )
        parent_path = repo / ".worktrees" / parent_id
        conn.execute(
            "UPDATE tasks SET project_id = ?, workspace_path = ?, branch_name = ? "
            "WHERE id = ?",
            (
                "project-one",
                str(parent_path),
                f"project-one/{parent_id}",
                parent_id,
            ),
        )
        conn.commit()

        child_id = kb.create_task(
            conn,
            title="independent builder",
            assignee="worker",
            workspace_kind="worktree",
            project_id="project-one",
            project_source_task_id=parent_id,
            owned_paths=["src/child"],
        )
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.project_id == "project-one"
        assert child.workspace_kind == "worktree"
        assert child.workspace_path == str(repo / ".worktrees" / child_id)
        assert child.workspace_path != str(parent_path)
        assert child.branch_name.startswith(f"project-one/{child_id}")
    finally:
        conn.close()


def test_kernel_materializes_current_run_patch_inside_owned_scope(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="remote sandbox author",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/remote-patch",
            owned_paths=["src/owned"],
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        attachment_id = _attach_patch(
            conn,
            task_id,
            attachment_root,
            "src/owned/remote.py",
            "remote = True\n",
        )

        # Simulate a process/DB interruption after the kernel commit but
        # before task completion. Retrying the same active-run handoff must
        # recognize its exact attachment/hash trailers instead of applying it
        # twice or leaving the task stuck.
        first_receipt, _ = kb._materialize_remote_worktree_handoff(
            conn,
            task_id,
            patch_attachment_id=attachment_id,
            merge_parent_heads=False,
            expected_run_id=task.current_run_id,
        )
        assert first_receipt["patch_attachment_id"] == attachment_id

        assert kb.complete_task(
            conn,
            task_id,
            summary="Sandbox patch materialized",
            patch_attachment_id=attachment_id,
            expected_run_id=task.current_run_id,
        )
        completed = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert completed is not None and completed.head_commit
        assert _git(
            repo, "show", f"{completed.head_commit}:src/owned/remote.py"
        ) == "remote = True"
        assert run is not None
        assert run.metadata["execution_receipt"]["changed_paths"] == [
            "src/owned/remote.py"
        ]
        receipt = run.metadata["worktree_materialization"]
        assert receipt["patch_attachment_id"] == attachment_id
        assert receipt["materialized_head"] == completed.head_commit
        assert len(receipt["patch_sha256"]) == 64
        assert f"Hermes-Patch-Attachment: {attachment_id}" in _git(
            repo, "log", "-1", "--format=%B", completed.head_commit
        )
    finally:
        conn.close()


def test_kernel_rejects_out_of_scope_patch_and_restores_worktree(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="bounded remote sandbox",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/reject-remote-patch",
            owned_paths=["src/owned"],
        )
        workspace = _materialize(conn, task_id)
        original_head = _git(workspace, "rev-parse", "HEAD")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        attachment_id = _attach_patch(
            conn,
            task_id,
            attachment_root,
            "src/outside.py",
            "outside = True\n",
        )

        with pytest.raises(kb.WorktreeScopeError, match="outside declared ownership"):
            kb.complete_task(
                conn,
                task_id,
                summary="Must fail closed",
                patch_attachment_id=attachment_id,
                expected_run_id=task.current_run_id,
            )
        landed = kb.get_task(conn, task_id)
        assert landed is not None and landed.status == "running"
        assert _git(workspace, "rev-parse", "HEAD") == original_head
        assert _git(workspace, "status", "--porcelain") == ""
        assert not (workspace / "src" / "outside.py").exists()
        assert kb.list_events(conn, task_id)[-1].kind == (
            "completion_blocked_file_scope"
        )
    finally:
        conn.close()


def test_kernel_merges_exact_parent_heads_before_integration_patch(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")

    def create_scoped(title, branch, owned_paths, parents=(), *, integrates=False):
        task_id = kb.create_task(
            conn,
            title=title,
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
            owned_paths=owned_paths,
            integrates_parent_heads=integrates,
            parents=parents,
        )
        conn.execute(
            "UPDATE tasks SET project_id = 'project-remote' WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        return task_id

    try:
        parents = []
        for name in ("left", "right"):
            parent = create_scoped(
                name, f"feature/remote-{name}", [f"src/{name}"]
            )
            workspace = _materialize(conn, parent)
            target = workspace / "src" / name / "feature.py"
            target.parent.mkdir(parents=True)
            target.write_text(f"name = {name!r}\n", encoding="utf-8")
            _git(workspace, "add", f"src/{name}/feature.py")
            _git(workspace, "commit", "-m", f"feat: {name}")
            assert kb.complete_task(conn, parent, summary=f"{name} complete")
            parents.append(parent)

        integration = create_scoped(
            "remote integration",
            "feature/remote-integration",
            ["."],
            parents=parents,
            integrates=True,
        )
        _materialize(conn, integration)
        task = kb.get_task(conn, integration)
        assert task is not None and task.current_run_id is not None
        attachment_id = _attach_patch(
            conn,
            integration,
            attachment_root,
            "integration-receipt.md",
            "# Integrated\n",
        )

        assert kb.complete_task(
            conn,
            integration,
            summary="Exact heads integrated",
            patch_attachment_id=attachment_id,
            merge_parent_heads=True,
            expected_run_id=task.current_run_id,
        )
        run = kb.latest_run(conn, integration)
        assert run is not None
        expected_heads = sorted(
            [
                {
                    "task_id": parent,
                    "head_commit": kb.get_task(conn, parent).head_commit,
                }
                for parent in parents
            ],
            key=lambda item: item["task_id"],
        )
        assert run.metadata["execution_receipt"]["parent_heads"] == expected_heads
        assert run.metadata["worktree_materialization"][
            "merged_parent_heads"
        ] == expected_heads
        completed = kb.get_task(conn, integration)
        assert completed is not None and completed.head_commit
        assert _git(
            repo, "show", f"{completed.head_commit}:src/left/feature.py"
        ) == "name = 'left'"
        assert _git(
            repo, "show", f"{completed.head_commit}:src/right/feature.py"
        ) == "name = 'right'"
        assert _git(
            repo, "show", f"{completed.head_commit}:integration-receipt.md"
        ) == "# Integrated"
    finally:
        conn.close()


def test_cli_claim_records_worktree_base_before_control_returns(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = _repo(tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="manual terminal work",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/manual-claim",
            owned_paths=["src/manual"],
        )

    args = type("Args", (), {"task_id": task_id, "ttl": None})()
    assert kanban_cli._cmd_claim(args) == 0

    with kb.connect() as conn:
        claimed = kb.get_task(conn, task_id)
    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.base_commit == _git(repo, "rev-parse", "HEAD")
    assert claimed.branch_name == "feature/manual-claim"
    assert claimed.workspace_path == str(repo / ".worktrees" / task_id)


def test_completion_derives_exact_clean_in_scope_git_receipt(tmp_path):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="change owned slice",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/owned",
            owned_paths=["src/owned"],
        )
        workspace = _materialize(conn, task_id)
        (workspace / "src" / "owned").mkdir(parents=True)
        (workspace / "src" / "owned" / "feature.py").write_text("ok = True\n", encoding="utf-8")
        _git(workspace, "add", "src/owned/feature.py")
        _git(workspace, "commit", "-m", "feat: add owned slice")

        assert kb.complete_task(
            conn,
            task_id,
            summary="Owned slice complete",
            metadata={"execution_receipt": {"head_commit": "worker-invented"}},
        ) is True
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert task is not None and task.status == "done"
        assert task.base_commit and task.head_commit
        assert task.base_commit != task.head_commit
        assert run is not None
        assert run.metadata["execution_receipt"] == {
            "kind": "scoped_worktree_v1",
            "base_commit": task.base_commit,
            "head_commit": task.head_commit,
            "branch": "feature/owned",
            "owned_paths": ["src/owned"],
            "changed_paths": ["src/owned/feature.py"],
            "parent_heads": [],
        }

        child = kb.create_task(conn, title="integrate", assignee="builder", parents=[task_id])
        context = kb.build_worker_context(conn, child)
        assert f"base={task.base_commit}" in context
        assert f"head={task.head_commit}" in context
        assert "branch=feature/owned" in context
    finally:
        conn.close()


def test_integrator_must_contain_every_exact_same_project_parent_head(tmp_path):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")

    def create_scoped(
        title, branch, owned_paths, parents=(), *, integrates_parent_heads=False
    ):
        task_id = kb.create_task(
            conn,
            title=title,
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
            owned_paths=owned_paths,
            integrates_parent_heads=integrates_parent_heads,
            parents=parents,
        )
        conn.execute(
            "UPDATE tasks SET project_id = 'project-1' WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        return task_id

    try:
        left = create_scoped("left", "feature/left", ["src/left"])
        right = create_scoped("right", "feature/right", ["src/right"])
        for task_id, folder in ((left, "left"), (right, "right")):
            workspace = _materialize(conn, task_id)
            (workspace / "src" / folder).mkdir(parents=True)
            (workspace / "src" / folder / "feature.py").write_text(
                f"name = {folder!r}\n", encoding="utf-8"
            )
            _git(workspace, "add", f"src/{folder}/feature.py")
            _git(workspace, "commit", "-m", f"feat: add {folder}")
            assert kb.complete_task(conn, task_id, summary=f"{folder} complete")

        left_head = kb.get_task(conn, left).head_commit
        right_head = kb.get_task(conn, right).head_commit
        assert left_head and right_head

        missing = create_scoped(
            "bad integration",
            "feature/missing-integration",
            ["."],
            parents=[left, right],
            integrates_parent_heads=True,
        )
        _materialize(conn, missing)
        with pytest.raises(kb.WorktreeScopeError, match="does not contain parent head"):
            kb.complete_task(conn, missing, summary="Not actually integrated")
        assert kb.get_task(conn, missing).status == "running"

        # Release the deliberately rejected run so it does not hold the
        # exclusive scope while the valid integrator is claimed.
        kb.block_task(conn, missing, reason="negative test complete")
        integrated = create_scoped(
            "good integration",
            "feature/good-integration",
            ["."],
            parents=[left, right],
            integrates_parent_heads=True,
        )
        workspace = _materialize(conn, integrated)
        _git(workspace, "merge", "--no-edit", left_head)
        _git(workspace, "merge", "--no-edit", right_head)
        assert kb.complete_task(conn, integrated, summary="Both heads integrated")

        run = kb.latest_run(conn, integrated)
        assert run is not None
        assert run.metadata["execution_receipt"]["parent_heads"] == sorted(
            [
                {"task_id": left, "head_commit": left_head},
                {"task_id": right, "head_commit": right_head},
            ],
            key=lambda item: item["task_id"],
        )
    finally:
        conn.close()


@pytest.mark.parametrize("parent_owned_paths", [None, ["src/parent"]])
def test_integrator_rejects_mutating_parent_without_git_receipt(
    tmp_path, parent_owned_paths
):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent = kb.create_task(
            conn,
            title="corrupt parent",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/corrupt-parent",
            owned_paths=parent_owned_paths,
        )
        conn.execute(
            "UPDATE tasks SET project_id='project-1', status='done' WHERE id=?",
            (parent,),
        )
        conn.commit()
        integrator = kb.create_task(
            conn,
            title="integrate corrupt parent",
            assignee="builder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/integrate-corrupt",
            parents=[parent],
            owned_paths=["."],
            integrates_parent_heads=True,
        )
        conn.execute(
            "UPDATE tasks SET project_id='project-1' WHERE id=?",
            (integrator,),
        )
        conn.commit()
        _materialize(conn, integrator)

        with pytest.raises(kb.WorktreeScopeError, match="missing its git head receipt"):
            kb.complete_task(conn, integrator, summary="Must fail closed")
        assert kb.get_task(conn, integrator).status == "running"
    finally:
        conn.close()


def test_integrator_rechecks_parent_head_at_completion(tmp_path):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent = kb.create_task(
            conn,
            title="parent",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/parent",
            owned_paths=["src/parent"],
        )
        conn.execute(
            "UPDATE tasks SET project_id='project-1' WHERE id=?",
            (parent,),
        )
        conn.commit()
        parent_workspace = _materialize(conn, parent)
        (parent_workspace / "src" / "parent").mkdir(parents=True)
        (parent_workspace / "src" / "parent" / "first.py").write_text(
            "first = True\n", encoding="utf-8"
        )
        _git(parent_workspace, "add", "src/parent/first.py")
        _git(parent_workspace, "commit", "-m", "feat: first parent head")
        assert kb.complete_task(conn, parent, summary="First head complete")
        first_head = kb.get_task(conn, parent).head_commit
        assert first_head

        integrator = kb.create_task(
            conn,
            title="integrator",
            assignee="builder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/integrator",
            parents=[parent],
            owned_paths=["."],
            integrates_parent_heads=True,
        )
        conn.execute(
            "UPDATE tasks SET project_id='project-1' WHERE id=?",
            (integrator,),
        )
        conn.commit()
        integration_workspace = _materialize(conn, integrator)
        _git(integration_workspace, "merge", "--no-edit", first_head)

        # The dependency changes after integration began. Completion must
        # compare against the parent's current exact receipt, not the head the
        # builder happened to merge earlier.
        parent_task = kb.get_task(conn, parent)
        assert parent_task is not None
        parent_workspace, _ = kb._resolve_worktree_workspace(parent_task)
        (parent_workspace / "src" / "parent" / "second.py").write_text(
            "second = True\n", encoding="utf-8"
        )
        _git(parent_workspace, "add", "src/parent/second.py")
        _git(parent_workspace, "commit", "-m", "feat: changed parent head")
        changed_head = _git(parent_workspace, "rev-parse", "HEAD")
        conn.execute(
            "UPDATE tasks SET head_commit=? WHERE id=?",
            (changed_head, parent),
        )
        conn.commit()

        with pytest.raises(kb.WorktreeScopeError, match="does not contain parent head"):
            kb.complete_task(conn, integrator, summary="Stale integration")
        assert kb.get_task(conn, integrator).status == "running"
    finally:
        conn.close()


def test_completion_rejects_dirty_worktree_without_closing_task(tmp_path):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="commit before completion",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/dirty",
            owned_paths=["src/owned"],
        )
        workspace = _materialize(conn, task_id)
        (workspace / "src" / "owned").mkdir(parents=True)
        (workspace / "src" / "owned" / "dirty.py").write_text(
            "dirty = True\n", encoding="utf-8"
        )

        with pytest.raises(kb.WorktreeScopeError, match="worktree is dirty"):
            kb.complete_task(conn, task_id, summary="Should be rejected")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "running" and task.head_commit is None
    finally:
        conn.close()


def test_completion_rejects_committed_out_of_scope_change_without_closing_task(tmp_path):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="stay in scope",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/outside",
            owned_paths=["src/owned"],
        )
        workspace = _materialize(conn, task_id)
        (workspace / "src").mkdir()
        (workspace / "src" / "outside.py").write_text("bad = True\n", encoding="utf-8")
        _git(workspace, "add", "src/outside.py")
        _git(workspace, "commit", "-m", "test: outside scope")

        with pytest.raises(kb.WorktreeScopeError, match="outside declared ownership"):
            kb.complete_task(conn, task_id, summary="Should be rejected")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "running" and task.head_commit is None
        assert kb.list_events(conn, task_id)[-1].kind == "completion_blocked_file_scope"
    finally:
        conn.close()


def test_large_receipt_keeps_bounded_path_evidence(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        monkeypatch.setattr(kb, "_MAX_RECEIPT_CHANGED_PATHS", 1)
        task_id = kb.create_task(
            conn,
            title="bounded receipt",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feature/bounded-receipt",
            owned_paths=["src/owned"],
        )
        workspace = _materialize(conn, task_id)
        (workspace / "src" / "owned").mkdir(parents=True)
        for name in ("a.py", "b.py"):
            (workspace / "src" / "owned" / name).write_text(
                f"name = {name!r}\n", encoding="utf-8"
            )
        _git(workspace, "add", "src/owned")
        _git(workspace, "commit", "-m", "feat: exercise bounded receipt")

        assert kb.complete_task(conn, task_id, summary="Bounded receipt complete")
        run = kb.latest_run(conn, task_id)
        receipt = run.metadata["execution_receipt"]
        assert receipt["changed_paths"] == ["src/owned/a.py"]
        assert receipt["changed_path_count"] == 2
        assert receipt["changed_paths_truncated"] is True
        assert len(receipt["changed_paths_sha256"]) == 64
    finally:
        conn.close()
