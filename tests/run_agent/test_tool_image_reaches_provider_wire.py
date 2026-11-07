"""A picture a tool returns reaches the provider request as a picture, not as its text summary.

The image reader ``kanban_view_image`` runs as a dispatched kanban worker runs it: a real board under
the test's own HERMES_HOME, a claimed task with a small PNG attachment made here with Pillow, and the
dispatcher's environment set with monkeypatch. The model reaches the reader through the ``tool_call``
bridge (the reader is a deferred tool), the agent's own tool execution commits the result, and the
agent's own request builder builds the request. Nothing is sent: every outbound connection and name
lookup is refused and recorded, and none may happen.

The two ``anthropic`` routes have no ``model.supports_vision`` key and the catalog has no entry for
their models, so only the main route's own vision verdict keeps their picture.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from hermes_cli import kanban_db as kb
from run_agent import AIAgent
from tools.design_access import DESIGNER
from tools.registry import invalidate_check_fn_cache

ENV = ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
       "HERMES_KANBAN_CLAIM_LOCK")
LEFT, RIGHT = (220, 30, 40), (20, 60, 220)  # the two halves of the picture


@pytest.fixture
def network_attempts(monkeypatch):
    """Every outbound connection and name lookup, refused and recorded."""
    attempts = []
    real_connect = socket.socket.connect

    def connect(sock, address):
        if sock.family == socket.AF_UNIX:
            return real_connect(sock, address)
        attempts.append(address)
        raise OSError(f"no network in this test: {address}")

    def getaddrinfo(host, *args, **kwargs):
        attempts.append(host)
        raise OSError(f"no network in this test: {host}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return attempts


@pytest.fixture
def attachment(tmp_path, monkeypatch) -> int:
    """A claimed design task with one PNG attachment, and the environment the dispatcher injects into
    that task's worker, all under the test's own HERMES_HOME (a profile home under the root)."""
    home = tmp_path / "root" / "profiles" / "designer"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_DELEGATED_CHILD_CONTEXT", *ENV):
        monkeypatch.delenv(var, raising=False)
    kb.init_db(board=kb.DEFAULT_BOARD)
    kb.create_board("design")
    kb.connect(board="design").close()
    db = kb.kanban_db_path(board="design")
    picture = Image.new("RGB", (64, 32), LEFT)
    ImageDraw.Draw(picture).rectangle((32, 0, 64, 32), fill=RIGHT)
    png = io.BytesIO()
    picture.save(png, format="PNG")
    with contextlib.closing(kb.connect(db)) as conn:
        task_id = kb.create_task(conn, title="own", assignee=DESIGNER)
        assert kb.claim_task(conn, task_id) is not None
        task = kb.get_task(conn, task_id)
        attachment_id = kb.store_attachment_bytes(conn, task_id, "picture.png", png.getvalue(),
                                                  content_type="image/png", board="design")
    for key, value in {
        "HERMES_KANBAN_DB": str(db), "HERMES_KANBAN_BOARD": "design", "HERMES_KANBAN_TASK": task_id,
        "HERMES_KANBAN_RUN_ID": str(task.current_run_id), "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock or "",
    }.items():
        monkeypatch.setenv(key, value)
    return attachment_id


def decoded(data: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
    image.load()
    return image.convert("RGB")


def close(pixel, colour) -> bool:
    return all(abs(a - b) <= 40 for a, b in zip(pixel, colour))


def picture_in_tool_result(request: dict) -> Image.Image:
    """anthropic_messages: the image block inside the tool_result answering call_1."""
    (result,) = [block for message in request["messages"] if isinstance(message["content"], list)
                 for block in message["content"] if block.get("type") == "tool_result"]
    assert result["tool_use_id"] == "call_1"
    assert isinstance(result["content"], list), result["content"]  # not the picture's text summary
    (image,) = [part for part in result["content"] if part.get("type") == "image"]
    assert image["source"]["type"] == "base64"
    return decoded(image["source"]["data"])


def picture_in_function_call_output(request: dict) -> Image.Image:
    """codex_responses: the input_image inside the function_call_output answering call_1."""
    (output,) = [item["output"] for item in request["input"]
                 if item.get("type") == "function_call_output" and item.get("call_id") == "call_1"]
    assert isinstance(output, list), output  # not the picture's text summary
    (image,) = [part for part in output if part.get("type") == "input_image"]
    header, _, data = image["image_url"].partition(",")
    assert header.startswith("data:image/") and header.endswith(";base64")
    return decoded(data)


@pytest.mark.parametrize(("provider", "api_mode", "base_url", "model", "vision_key"), [
    pytest.param("anthropic", "anthropic_messages", "https://api.anthropic.com", "claude-sonnet-5-5", {},
                 id="anthropic_messages-claude-sonnet-5-5"),
    pytest.param("anthropic", "anthropic_messages", "https://api.anthropic.com", "claude-opus-5-5", {},
                 id="anthropic_messages-claude-opus-5-5"),
    pytest.param("openai-codex", "codex_responses", "https://chatgpt.com/backend-api/codex", "gpt-6.1-sol",
                 {"supports_vision": True}, id="codex_responses-gpt-6.1-sol"),
])
def test_the_readers_picture_reaches_the_provider_request(network_attempts, attachment, provider, api_mode,
                                                          base_url, model, vision_key):
    # context_length is set so the agent does not ask the provider for it while it starts.
    route = {"provider": provider, "default": model, "base_url": base_url, "context_length": 200000, **vision_key}
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        json.dumps({"design_browser": {"boards": ["design"]}, "model": route}), encoding="utf-8")
    invalidate_check_fn_cache()  # the reader's availability is decided under this worker's environment
    arguments = json.dumps({"calls": [{"name": "kanban_view_image", "arguments": {"attachment_id": attachment}}]})
    call = SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name="tool_call", arguments=arguments))
    messages = [
        {"role": "user", "content": "Look at the picture attached to your task."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "tool_call", "arguments": arguments}}]},
    ]

    with patch("agent.models_dev.get_model_capabilities", return_value=None):  # the catalog has no entry
        agent = AIAgent(api_key="test-key", base_url=base_url, provider=provider, api_mode=api_mode, model=model,
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        enabled_toolsets=["design_vision"])
        agent._execute_tool_calls(SimpleNamespace(content="", tool_calls=[call]), messages, "task-1")
        request = agent._build_api_kwargs(messages)

    if api_mode == "anthropic_messages":
        picture = picture_in_tool_result(request)
    else:
        picture = picture_in_function_call_output(request)
    width, height = picture.size
    assert close(picture.getpixel((width // 4, height // 2)), LEFT), picture.getpixel((width // 4, height // 2))
    assert close(picture.getpixel((3 * width // 4, height // 2)), RIGHT), picture.getpixel((3 * width // 4, height // 2))
    assert network_attempts == []
