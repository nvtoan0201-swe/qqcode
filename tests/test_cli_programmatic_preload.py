from __future__ import annotations

from pathlib import Path

import pytest

from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core import run_programmatic
from vibe.core.config import Backend, SessionLoggingConfig, VibeConfig
from vibe.core.sessions.entries import UsageEntry
from vibe.core.sessions.store import JsonlSessionStore
from vibe.core.types import LLMMessage, OutputFormat, Role


class SpyStreamingFormatter:
    def __init__(self) -> None:
        self.emitted: list[tuple[Role, str | None]] = []

    def on_message_added(self, message: LLMMessage) -> None:
        self.emitted.append((message.role, message.content))

    def on_event(self, _event) -> None:  # No-op for this test
        pass

    def finalize(self) -> str | None:
        return None


def test_run_programmatic_preload_streaming_is_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = SpyStreamingFormatter()
    monkeypatch.setattr(
        "vibe.core.programmatic.create_formatter", lambda *_args, **_kwargs: spy
    )

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([
            mock_llm_chunk(
                content="Decorators are wrappers that modify function behavior.",
                finish_reason="stop",
            )
        ]),
    ):
        cfg = VibeConfig(
            session_logging=SessionLoggingConfig(enabled=False),
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
        )

        previous = [
            LLMMessage(
                role=Role.system, content="This system message should be ignored."
            ),
            LLMMessage(
                role=Role.user, content="Previously, you told me about decorators."
            ),
            LLMMessage(
                role=Role.assistant,
                content="Sure, decorators allow you to wrap functions.",
            ),
        ]

        run_programmatic(
            config=cfg,
            prompt="Can you summarize what decorators are?",
            output_format=OutputFormat.STREAMING,
            previous_messages=previous,
            mode="auto-approve",
        )

        roles = [r for r, _ in spy.emitted]
        assert roles == [
            Role.system,
            Role.user,
            Role.assistant,
            Role.user,
            Role.assistant,
        ]
        assert (
            spy.emitted[0][1] == "You are Vibe, a super useful programming assistant."
        )
        assert spy.emitted[1][1] == "Previously, you told me about decorators."
        assert spy.emitted[2][1] == "Sure, decorators allow you to wrap functions."
        assert spy.emitted[3][1] == "Can you summarize what decorators are?"
        assert (
            spy.emitted[4][1]
            == "Decorators are wrappers that modify function behavior."
        )


def test_run_programmatic_ignores_system_messages_in_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = SpyStreamingFormatter()
    monkeypatch.setattr(
        "vibe.core.programmatic.create_formatter", lambda *_args, **_kwargs: spy
    )

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([mock_llm_chunk(content="Understood.")]),
    ):
        cfg = VibeConfig(
            session_logging=SessionLoggingConfig(enabled=False),
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
        )

        run_programmatic(
            config=cfg,
            prompt="Let's move on to practical examples.",
            output_format=OutputFormat.STREAMING,
            previous_messages=[
                LLMMessage(
                    role=Role.system,
                    content="First system message that should be ignored.",
                ),
                LLMMessage(role=Role.user, content="Continue our previous discussion."),
                LLMMessage(
                    role=Role.system,
                    content="Second system message that should be ignored.",
                ),
            ],
            mode="auto-approve",
        )

        roles = [r for r, _ in spy.emitted]
        assert roles == [Role.system, Role.user, Role.user, Role.assistant]
        assert (
            spy.emitted[0][1] == "You are Vibe, a super useful programming assistant."
        )
        assert spy.emitted[1][1] == "Continue our previous discussion."
        assert spy.emitted[2][1] == "Let's move on to practical examples."
        assert spy.emitted[3][1] == "Understood."


def test_run_programmatic_persists_jsonl_session(tmp_path: Path) -> None:
    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([
            mock_llm_chunk(content="persisted answer", finish_reason="stop")
        ]),
    ):
        cfg = VibeConfig(
            session_logging=SessionLoggingConfig(
                enabled=True,
                fsync=False,
                sessions_dir=str(tmp_path),
                save_dir=str(tmp_path / "legacy"),
            ),
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
        )

        run_programmatic(
            config=cfg,
            prompt="persist this",
            output_format=OutputFormat.TEXT,
            mode="auto-approve",
        )

    store = JsonlSessionStore(cfg.session_logging)
    summaries = store.list(workdir=None, limit=10)
    assert len(summaries) == 1
    view = store.read(summaries[0].path)
    assert [message.content for message in view.messages()] == [
        "persist this",
        "persisted answer",
    ]
    assert any(isinstance(entry, UsageEntry) for entry in view.entries)
