"""Serve receipt-backed Projects and optional portfolio advice."""

import contextlib
from datetime import date
import logging
import sqlite3

from aiohttp import web

from hermes_cli import kanban_db, projects_db
from hermes_cli.owner_text import check_owner_text

logger = logging.getLogger(__name__)


def _portfolio_view(projects):
    from hermes_cli.owner_project_review import read_settings
    from hermes_cli.owner_workspace import OwnerWorkspaceError, _open_read_only_sqlite

    # The owner's current settings decide: advice of a review or portfolio switched off, or of a Project no longer
    # allowlisted (by id or slug), is not shown.
    settings = read_settings()
    if not (settings.enabled and settings.portfolio):
        return None
    path = kanban_db.kanban_home() / "projects.db"
    if not path.is_file():
        return None
    try:
        with contextlib.closing(_open_read_only_sqlite(path, label="projects.db")) as registry:
            stored = projects_db.get_portfolio_review_brief(registry)
        if stored is None:
            return None
        scope = stored["project_scope"]
        visible = {item["project_id"] for item in projects}
        slugs = {item["project_id"]: item.get("slug") for item in projects}
        if (not isinstance(scope, list) or not scope or any(not isinstance(item, str) for item in scope)
                or len(scope) != len(set(scope)) or not set(scope).issubset(visible)
                or any(item not in settings.allowlist and slugs[item] not in settings.allowlist for item in scope)):
            return None
        names = [item["name"] for item in projects if item["project_id"] in scope]
        result = {key: stored[key] for key in ("review_date", "conflicts", "priorities", "shared_blockers", "written_at", "revision")}
        if (date.fromisoformat(result["review_date"]).isoformat() != result["review_date"]
                or type(result["written_at"]) is not int or result["written_at"] < 0
                or type(result["revision"]) is not int or result["revision"] < 1):
            return None
        for key in ("conflicts", "priorities", "shared_blockers"):
            items = result[key]
            if not isinstance(items, list) or len(items) > 5:
                return None
            for text in items:
                if (not isinstance(text, str) or not 1 <= len(text) <= 200
                        or check_owner_text(text, "planner_prose") != text
                        or not any(text.startswith(name + ": ") for name in names)):
                    return None
        return result
    except (OwnerWorkspaceError, sqlite3.Error, OSError, ValueError, TypeError, KeyError):
        return None


async def list_owner_projects(adapter, request):
    from gateway.platforms.api_server import (
        _openai_error, _owner_workspace_capability_requested, _owner_workspace_toolset_enabled,
    )

    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    try:
        from gateway.run import _load_gateway_config

        if not _owner_workspace_toolset_enabled(_load_gateway_config()):
            return web.json_response(_openai_error(
                "Owner workspace is not enabled for this profile", code="owner_workspace_not_enabled",
            ), status=404)
        from hermes_cli.owner_workspace import (
            OWNER_PROJECT_BOARD_COUNTS_CAPABILITY, OWNER_PROJECT_LAST_CHANGE_CAPABILITY,
            OWNER_PROJECT_LIFECYCLE_REVISION_CAPABILITY, OWNER_PROJECT_REMOVAL_STATE_CAPABILITY,
            list_committed_projects, resolve_owner_context,
        )

        projects = list_committed_projects(
            resolve_owner_context(),
            lifecycle_revision=_owner_workspace_capability_requested(request, OWNER_PROJECT_LIFECYCLE_REVISION_CAPABILITY),
            removal_state=_owner_workspace_capability_requested(request, OWNER_PROJECT_REMOVAL_STATE_CAPABILITY),
            board_counts=_owner_workspace_capability_requested(request, OWNER_PROJECT_BOARD_COUNTS_CAPABILITY),
            project_last_change=_owner_workspace_capability_requested(request, OWNER_PROJECT_LAST_CHANGE_CAPABILITY),
        )
        response = {"object": "hermes.owner_workspace.project_list", "data": projects}
        if _owner_workspace_capability_requested(request, "portfolio_review"):
            response["portfolio_review"] = _portfolio_view(projects)
        return web.json_response(response)
    except Exception:
        logger.exception("[api_server] owner-workspace Project projection failed")
        return web.json_response(_openai_error(
            "Owner workspace Project projection is unavailable", code="owner_workspace_unavailable",
        ), status=500)
