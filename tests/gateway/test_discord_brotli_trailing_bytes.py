"""A Brotli attachment body with trailing bytes must end in a recorded failure.

Documents that ``att.read()`` cannot fetch fall back to an aiohttp download in
``DiscordAdapter._cache_discord_document``, and aiohttp decodes
``Content-Encoding: br`` through brotlicffi. A valid Brotli stream followed by
extra bytes is malformed: the download must fail cleanly and be logged against
the attachment, never hang and never be cached or injected as if it were whole.

The server is an in-process aiohttp app on 127.0.0.1; the deadline is enforced
with a worker thread so it holds whether or not pytest-timeout is installed.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from aiohttp import web

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

# brotli.compress(b"hello from a discord attachment\n"): one complete Brotli stream.
_BROTLI_STREAM = bytes.fromhex("1b1f00f88d54b5bf0a934b0510b21e449145b0919a0cf4123b2b99ed59")
_TRAILING_BYTES = b"TRAILING-GARBAGE"
_DEADLINE_SECONDS = 20  # well under the adapter's own 30 s aiohttp timeout


async def _collect_from_local_server(body: bytes) -> tuple:
    async def serve_attachment(request):
        return web.Response(body=body, headers={"Content-Type": "text/plain", "Content-Encoding": "br"})

    app = web.Application()
    app.router.add_get("/attachments/notes.txt", serve_attachment)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        port = runner.addresses[0][1]
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        attachment = SimpleNamespace(  # no .read(): forces the aiohttp document fallback
            url=f"http://127.0.0.1:{port}/attachments/notes.txt",
            filename="notes.txt",
            content_type="text/plain",
            size=len(body),
        )
        return await adapter._collect_attachment_media([attachment])
    finally:
        await runner.cleanup()


def _run_with_deadline(body: bytes) -> tuple:
    outcome = {}

    def worker():
        try:
            outcome["result"] = asyncio.run(_collect_from_local_server(body))
        except BaseException as exc:  # re-raised in the test thread
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="discord-brotli-download", daemon=True)
    with patch("plugins.platforms.discord.adapter.is_safe_url", return_value=True), patch(
        "gateway.platforms.base.resolve_proxy_url", return_value=None
    ):
        thread.start()
        thread.join(_DEADLINE_SECONDS)
    assert not thread.is_alive(), f"attachment download still running after {_DEADLINE_SECONDS}s (hang)"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def test_brotli_body_with_trailing_bytes_is_a_recorded_download_failure(caplog):
    caplog.set_level(logging.WARNING, logger="plugins.platforms.discord.adapter")

    media_urls, media_types, injected_text = _run_with_deadline(_BROTLI_STREAM + _TRAILING_BYTES)

    assert media_urls == [] and media_types == [], "malformed br body was cached as a document"
    assert injected_text is None, "malformed br body was injected as text"
    failures = [
        record for record in caplog.records
        if record.name == "plugins.platforms.discord.adapter"
        and record.getMessage().startswith("[Discord] Failed to cache document notes.txt")
    ]
    assert len(failures) == 1, [record.getMessage() for record in caplog.records]
    failure = failures[0]
    assert failure.levelno == logging.WARNING
    assert "content-encoding" in failure.getMessage().lower()
    assert failure.exc_info and isinstance(failure.exc_info[1], aiohttp.ClientError)
