"""Build the JSON record text for one kanban task, the string kanban_show returns."""

import json


def _task_dict(t):
    return {
        "id": t.id, "title": t.title, "body": t.body,
        "assignee": t.assignee, "status": t.status,
        "tenant": t.tenant, "priority": t.priority,
        "workspace_kind": t.workspace_kind,
        "workspace_path": t.workspace_path,
        "branch_name": t.branch_name,
        "project_id": t.project_id,
        "owned_paths": t.owned_paths,
        "integrates_parent_heads": t.integrates_parent_heads,
        "base_commit": t.base_commit,
        "head_commit": t.head_commit,
        "created_by": t.created_by, "created_at": t.created_at,
        "started_at": t.started_at,
        "completed_at": t.completed_at,
        "result": t.result,
        "current_run_id": t.current_run_id,
        "model_override": t.model_override,
        "provider_override": t.provider_override,
    }


def _run_dict(r):
    return {
        "id": r.id, "profile": r.profile,
        "status": r.status, "outcome": r.outcome,
        "summary": r.summary, "error": r.error,
        "metadata": r.metadata,
        "started_at": r.started_at, "ended_at": r.ended_at,
    }


def build_record_text(kb, conn, task, tid) -> str:
    """Return ``task``'s record as the JSON text ``kanban_show`` replies with."""
    comments = kb.list_comments(conn, tid)
    events = kb.list_events(conn, tid)
    runs = kb.list_runs(conn, tid)
    parents = kb.parent_ids(conn, tid)
    children = kb.child_ids(conn, tid)
    return json.dumps({
        "task": _task_dict(task),
        "parents": parents,
        "children": children,
        "comments": [
            {"author": c.author, "body": c.body,
             "created_at": c.created_at}
            for c in comments
        ],
        "events": [
            {"kind": e.kind, "payload": e.payload,
             "created_at": e.created_at, "run_id": e.run_id}
            for e in events[-50:]   # cap; full log via CLI
        ],
        "runs": [_run_dict(r) for r in runs],
        # Also surface the worker's own context block so the
        # agent can include it directly if it wants. This is
        # the same string build_worker_context returns to the
        # dispatcher at spawn time.
        "worker_context": kb.build_worker_context(conn, tid),
    })
