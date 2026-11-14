"""Send one optional daily portfolio digest through the owner's existing adapter.

Persist the sending state before transport admission. An uncertain send stays told after a restart.
Keep daily keys separate from heads-up keys because heads-up pruning follows individual waits.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import unicodedata
from collections import Counter
from datetime import date

from gateway import owner_channel_pause, owner_heads_up
from gateway.config import Platform
from gateway.platforms.api_server_owner_projects import _portfolio_view
from gateway.status import _release_file_lock, _try_acquire_file_lock
from hermes_cli import owner_project_review as review, owner_project_summary as summary, owner_workspace, projects_db
from hermes_cli.owner_text import check_owner_text
from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)

MAX_MESSAGE = 1500  # The approved digest has a stricter limit than ordinary Telegram messages.
RECORD_FILE_NAME = "owner_review_digest.json"
LOCK_FILE_NAME = "owner_review_digest.lock"
# Seconds after the first and the second retryable refusal. A short network loss ("Not connected") at
# delivery time must not spend every attempt of the day on consecutive ticks.
RETRY_WAITS = (5 * 60, 15 * 60)
# Model prose reaches the owner as notes, never as a request: every action stays in the app.
NOTES_LINE = "These are Raphael's notes from today's review. To act on any of them, open the app."
NO_ADVICE_LINE = "Raphael has no advice across your projects today."
QUOTE_WORDS = 4  # A model line sharing this many consecutive words with a live question (all of a shorter one) quotes it.
# Cyrillic and Greek letters drawn like Latin ones, so a look-alike spelling still matches the question.
_LOOK_ALIKES = str.maketrans(
    "\u0430\u0435\u0456\u0458\u043e\u0440\u0441\u0443\u0445\u0455\u0501\u04bb\u04cf\u051b\u051d"
    "\u03bf\u03b1\u03b9\u03ba\u03bd\u03c1\u03c4\u03c5\u03c7",
    "aeijopcyxsdhlqwoaikvptux",
)

_warned_on = None  # The day of the last failure warning: a lasting fault warns once a day, not every tick.


def _read_record(path):
    told = owner_heads_up._read_record(path)
    if told is None:
        return None
    try:
        for day, entry in told.items():
            if (date.fromisoformat(day).isoformat() != day
                    or not 0 <= entry["attempts"] <= owner_heads_up.MAX_ATTEMPTS
                    or type(entry["at"]) is not int or entry["at"] < 0):
                return None
    except (ValueError, TypeError):
        return None
    return told


def _due(entry, now):
    """Never tried, or refused as retryable fewer than MAX_ATTEMPTS times and past its retry wait."""
    if not owner_heads_up._due(entry):
        return False
    return entry is None or now - entry["at"] >= RETRY_WAITS[min(entry["attempts"], len(RETRY_WAITS)) - 1]


def _compared(text):
    # The one form in which model lines meet live questions. A quote may keep or drop the punctuation inside a word
    # ("follow-up", "followup"), so the first list splits at it and the second removes it.
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_LOOK_ALIKES)
    pieces = ("".join(char for char in piece if char.isalnum()) for piece in folded.split())
    return "".join(char if char.isalnum() else " " for char in folded).split(), [piece for piece in pieces if piece]


def _quotes(line, question):
    # Either text may be the one that kept the punctuation, so each word list of the line meets each of the question.
    for asked in question:
        if not asked:
            continue  # An empty run would be found in every line.
        size = min(QUOTE_WORDS, len(asked))
        runs = {tuple(asked[start:start + size]) for start in range(len(asked) - size + 1)}
        if any(tuple(words[start:start + size]) in runs for words in line for start in range(len(words) - size + 1)):
            return True
    return False


def _no_portfolio_result(registry, settings, day):
    """No portfolio brief of the day, its attempts used up by the cap and no portfolio task of the day open.

    A task that is done waits for its harvest, and a blocked one is archived by the next portfolio tick: both are open.
    A brief is stored before its task is archived, so it is read last.
    """
    if review.budget(registry, day)[1] < settings.portfolio_cap:
        return False
    board = summary._summary_board_path()
    if board.exists():
        with contextlib.closing(owner_workspace._open_read_only_sqlite(board, label="kanban.db")) as tasks:
            if tasks.execute(
                "SELECT 1 FROM tasks WHERE created_by=? AND idempotency_key=? AND status!='archived'",
                (review.CREATED_BY, f"project-review:portfolio:{day}:0"),
            ).fetchone() is not None:
                return False
    return projects_db.get_portfolio_review_brief(registry, day) is None


def _ready(settings, day, now):
    ctx = owner_workspace.OwnerContext(actor="default", profile="default", session="")
    projects = [project for project in owner_workspace.list_committed_projects(ctx)
                if not project["archived"] and (project["project_id"] in settings.allowlist
                                                or project["slug"] in settings.allowlist)]
    portfolio = _portfolio_view(projects)
    # Only a day without a portfolio brief can go out without advice; a brief of the day, proven or not, keeps the rules.
    advised = portfolio is not None and portfolio["review_date"] == day
    if advised and portfolio["written_at"] > now:
        return None
    heads = owner_heads_up._read_record(get_default_hermes_root() / owner_heads_up.RECORD_FILE_NAME)
    if heads is None:
        return None
    notes = []
    with contextlib.closing(owner_workspace._open_read_only_sqlite(
        get_default_hermes_root() / "projects.db", label="projects.db",
    )) as registry:
        if advised:
            stored = projects_db.get_portfolio_review_brief(registry, day)
            # The optional API view validates receipt scope and prose. Refuse a result changed between these reads.
            if stored is None or any(stored[key] != value for key, value in portfolio.items()):
                return None
            visible = [project for project in projects if project["project_id"] in stored["project_scope"]]
        elif _no_portfolio_result(registry, settings, day):
            visible = projects
        else:
            return None
        for project in visible:
            brief = projects_db.get_project_review_brief(registry, project["project_id"])
            if brief is None or brief["review_date"] != day or brief["quiet"]:
                continue
            headline = brief["headline"]
            if (isinstance(headline, str) and 1 <= len(headline) <= review._BRIEF_LIMITS["headline"]
                    and check_owner_text(headline, "planner_prose") == headline):
                notes.append(f'{project["name"]}: {headline}')
    waits, read = owner_heads_up._waiting()
    scope = {project["project_id"] for project in visible}
    # An unread board gives neither its wait count nor its questions. Leave the day due for a later tick.
    if not scope <= read:
        return None
    counts = Counter(wait.project_id for wait in waits if wait.project_id in scope
                     and heads.get(wait.key, {}).get("state") not in ("sent", "paused", "unknown", "sending"))
    # Counts lead, so the cut at MAX_MESSAGE removes advice before an answer waiting on the owner.
    lines = []
    for project in visible:
        count = counts[project["project_id"]]
        if count:
            noun = "answer" if count == 1 else "answers"
            lines.append(f'{project["name"]}: {count} {noun} waiting on Decisions')
    if advised:
        notes.extend(text for key in ("conflicts", "priorities", "shared_blockers") for text in portfolio[key])
    # Valid prose can still quote a live question of any Project, also in part or in another spelling.
    # Keep that quote out even when its heads-up was already told.
    questions = [_compared(wait.question) for wait in waits]
    lines.extend(line for line in notes if not any(_quotes(_compared(line), question) for question in questions))
    if not lines:
        return None
    text = "\n".join([f"Your daily project review ({day})", NOTES_LINE if advised else NO_ADVICE_LINE, *lines])
    if len(text) > MAX_MESSAGE:
        text = text[:MAX_MESSAGE - 1] + "…"
    return (portfolio["written_at"] if advised else now), text


async def tick(runner) -> int:
    """Send at most one due digest. Refuse uncertain records and never raise into the watcher."""
    global _warned_on
    # Context-local pins survive to_thread without changing another gateway turn's process environment.
    token = set_hermes_home_override(get_default_hermes_root())
    try:
        settings = review.read_settings()
        if not (settings.enabled and settings.portfolio and settings.digest):
            return 0
        gates = owner_heads_up._gates(runner)
        if gates is None:
            return 0
        chat, adapter = gates
        root = get_default_hermes_root()
        record = root / RECORD_FILE_NAME
        with open(root / LOCK_FILE_NAME, "a+", encoding="utf-8") as handle:
            if not _try_acquire_file_lock(handle):
                return 0
            try:
                told = _read_record(record)
                if told is None:
                    logger.warning("Daily digest record is unreadable; nothing is sent")
                    return 0
                now = int(time.time())
                day = review.review_date(now, settings.review_hour)
                # Decide from the record first: a told day reads no registry and no board until the next day.
                entry = told.get(day)
                if day is None or not _due(entry, now):
                    return 0
                ready = await asyncio.to_thread(_ready, settings, day, now)
                if ready is None:
                    return 0
                created_at, text = ready
                attempts = entry["attempts"] if entry else 0
                if owner_channel_pause.drops_notice(Platform.TELEGRAM, chat, created_at):
                    told[day] = owner_heads_up._entry("paused", attempts)
                    owner_heads_up._write_record(record, told)
                    return 0
                told[day] = owner_heads_up._entry("sending", attempts)
                owner_heads_up._write_record(record, told)
                sent = 0
                try:
                    result = await adapter.send(chat, text)
                except Exception:
                    # Transport exceptions cannot prove refusal. The message can already have arrived.
                    told[day] = owner_heads_up._entry("unknown", attempts)
                else:
                    sent = int(getattr(result, "success", None) is True)
                    if sent:
                        told[day] = owner_heads_up._entry("sent", attempts)
                    elif getattr(result, "success", None) is False and getattr(result, "retryable", None) is True:
                        told[day] = owner_heads_up._entry("failed", attempts + 1)
                    else:
                        told[day] = owner_heads_up._entry("unknown", attempts)
                owner_heads_up._write_record(record, told)
                return sent
            finally:
                _release_file_lock(handle)
    except Exception as exc:
        # The type only: the text of a transport or store error can carry details that stay out of the log.
        today = date.fromtimestamp(time.time())
        level = logging.DEBUG if _warned_on == today else logging.WARNING
        _warned_on = today
        logger.log(level, "Daily digest tick could not finish: %s", type(exc).__name__)
    finally:
        reset_hermes_home_override(token)
    return 0
