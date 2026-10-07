from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.core.agent import Agent
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.types import BaseEvent


async def _wait_for(
    pilot, condition: Callable[[], object | None], timeout: float = 3.0
) -> object | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        await pilot.pause(0.05)
    return None


class CapturingAgent(Agent):
    def __init__(self) -> None:
        self.messages: list = []
        self.stats = SimpleNamespace(context_tokens=0)
        self.approval_callback = None
        self.prompts: list[str] = []

    async def act(self, msg: str) -> AsyncGenerator[BaseEvent]:
        self.prompts.append(msg)
        if False:
            yield


def _patch_agent_init(monkeypatch: pytest.MonkeyPatch, agent: Agent) -> None:
    async def _fake_initialize(self: VibeApp) -> None:
        if self.agent or self._agent_initializing:
            return

        self._agent_initializing = True
        try:
            self.agent = agent
        finally:
            self._agent_initializing = False
            self._agent_init_task = None

    monkeypatch.setattr(VibeApp, "_initialize_agent", _fake_initialize, raising=True)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("QQCODE_HOME", str(tmp_path / ".qqcode-home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def templates_dir(workdir: Path) -> Path:
    path = workdir / "templates"
    path.mkdir()
    (path / "pr.md").write_text(
        """---
description: Create a pull request
argument-hint: <branch>
---

Open a pull request for $1 into ${2:-main}.
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def vibe_config(workdir: Path) -> VibeConfig:
    return VibeConfig(
        workdir=workdir,
        session_logging=SessionLoggingConfig(enabled=False),
        enable_update_checks=False,
    )


@pytest.mark.asyncio
async def test_prompt_template_runs_end_to_end(
    templates_dir: Path, vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = VibeApp(config=vibe_config, prompt_template_dirs=[templates_dir])
    agent = CapturingAgent()
    _patch_agent_init(monkeypatch, agent)

    async with app.run_test() as pilot:
        await pilot.press(*'/pr "feature branch"')
        await pilot.press("enter")

        await _wait_for(pilot, lambda: agent.prompts)

    assert agent.prompts == ["Open a pull request for feature branch into main."]


def test_prompt_template_registered_in_help(
    templates_dir: Path, vibe_config: VibeConfig
) -> None:
    app = VibeApp(config=vibe_config, prompt_template_dirs=[templates_dir])

    command = app.commands.find_command("/pr")

    assert command is not None
    assert command.template == "pr"
    assert "`/pr`" in app.commands.get_help_text()


def test_template_name_colliding_with_builtin_loses(
    workdir: Path, vibe_config: VibeConfig
) -> None:
    collision_dir = workdir / "collision"
    collision_dir.mkdir()
    (collision_dir / "help.md").write_text("custom help", encoding="utf-8")

    app = VibeApp(config=vibe_config, prompt_template_dirs=[collision_dir])

    command = app.commands.find_command("/help")
    assert command is not None
    assert command.handler == "_show_help"
    assert "template:help" not in app.commands.commands
