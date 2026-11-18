"""Model update routes of Settings: undo the last update, start a practice update.

``POST /api/model-updates/undo`` rolls the last applied model update back: its
record entry and every role it moved, in one route batch whose audit names the
model each role runs again. ``POST /api/model-updates/practice`` files one
practice card a month for a role on a Project's board; the card's worker runs
the newer-version check for that role, and its suggestion names the same model
twice, so Update on it changes nothing. Both routes are the Models machine's
only; any other caller is told they do not exist. ``profile`` is the machine's
own and may only be absent or ``default``. An error is logged by its kind
only, because its text can name a sign-in.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from hermes_cli.web_deps import late

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Late-bound web_server helper (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_require_token = late("_require_token")

# The Project refusals of the cron routes, word for word.
_NO_PROJECT = "That Project does not exist or has been archived."
_PROJECTS_UNAVAILABLE = "Projects are unavailable right now. Try again shortly."


def _models_machine_profile(request: Request, profile: Optional[str]) -> Optional[str]:
    """The checked ``profile`` of a Models machine call; 404 for any other caller."""
    from plugins.dashboard_auth.raphael_workspace.model_policy import (
        model_machine_request,
        require_models_machine_profile,
    )

    if not model_machine_request(request):
        raise HTTPException(status_code=404, detail="Not found")
    return require_models_machine_profile(request, profile, allowed=(None, "default"))


def _unavailable(route: str, exc: BaseException) -> HTTPException:
    _log.warning("model-updates/%s: failed (%s)", route, type(exc).__name__)
    return HTTPException(status_code=503, detail="Service Unavailable")


@router.post("/api/model-updates/undo")
async def undo_model_update(request: Request, profile: Optional[str] = None):
    """Undo the last applied model update, or say why nothing changed."""
    _require_token(request)
    scoped_profile = _models_machine_profile(request, profile)
    from hermes_cli import owner_model_updates as omu
    from plugins.dashboard_auth.raphael_workspace import model_policy

    def journal(revisions: dict, provider: str, model: str) -> None:
        try:
            model_policy.journal_models_machine_batch_success(
                request,
                action="model_update_undo",
                roles=[(role, provider) for role in sorted(revisions)],
                models={role: model for role in revisions},
            ).commit()
        except Exception as exc:
            raise omu.Unavailable() from exc

    try:
        result = await run_in_threadpool(omu.undo_last_model_update, journal=journal)
    except Exception as exc:
        raise _unavailable("undo", exc) from None
    if not getattr(request.state, "token_route_audited", False):
        model_policy.audit_models_machine_success(
            request, action="model_update_undo", profile=scoped_profile
        )
    return result


@router.post("/api/model-updates/practice")
async def start_practice_update(request: Request, profile: Optional[str] = None):
    """File this month's practice card for ``role`` on the Project ``project``."""
    _require_token(request)
    scoped_profile = _models_machine_profile(request, profile)
    from hermes_cli import owner_model_updates as omu
    from hermes_cli import owner_workspace as ow
    from hermes_cli import projects_db
    from plugins.dashboard_auth.raphael_workspace import model_policy

    try:
        body = json.loads(await request.body())
    except (ValueError, RecursionError):
        body = None
    body = body if isinstance(body, dict) else {}
    role, slug = body.get("role"), body.get("project")
    if not isinstance(role, str) or role not in model_policy.admitted_profile_ids():
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        slug = projects_db.normalize_slug(slug) if isinstance(slug, str) else None
    except ValueError:
        slug = None
    if slug is None:
        raise HTTPException(status_code=400, detail=_NO_PROJECT)
    try:
        projects = await run_in_threadpool(
            lambda: ow.list_committed_projects(ow.resolve_owner_context())
        )
    except Exception as exc:
        _log.warning("model-updates/practice: projects unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail=_PROJECTS_UNAVAILABLE) from None
    project = next((p for p in projects if p["slug"] == slug and not p["archived"]), None)
    if project is None:
        raise HTTPException(status_code=400, detail=_NO_PROJECT)
    try:
        state = await run_in_threadpool(omu.start_practice, project, role)
    except Exception as exc:
        raise _unavailable("practice", exc) from None
    model_policy.audit_models_machine_success(
        request, action="model_update_practice", profile=scoped_profile
    )
    return {"state": state, "roles": [role]}
