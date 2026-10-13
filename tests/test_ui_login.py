from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.cli.textual_ui.widgets.login_screen import LoginScreen, SecretScreen
from vibe.core.config import Backend, ProviderConfig, SessionLoggingConfig, VibeConfig


@pytest.fixture
def vibe_config(tmp_path: Path) -> VibeConfig:
    return VibeConfig(
        workdir=tmp_path,
        session_logging=SessionLoggingConfig(enabled=False),
        enable_update_checks=False,
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            ),
            ProviderConfig(
                name="anthropic",
                api_base="https://api.anthropic.com",
                api_key_env_var="ANTHROPIC_API_KEY",
                api_style="anthropic",
            ),
        ],
    )


async def type_and_submit(pilot: Any, text: str) -> None:
    await pilot.press(*text)
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_login_flow_stores_an_api_key(
    vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "vibe.cli.textual_ui.app.save_api_key",
        lambda provider, key: saved.append((provider.name, key)) or Path("/tmp/.env"),
    )
    app = VibeApp(config=vibe_config)

    async with app.run_test() as pilot:
        await type_and_submit(pilot, "/login")

        assert isinstance(app.screen, LoginScreen)
        await type_and_submit(pilot, "api key")

        assert isinstance(app.screen, LoginScreen)
        await type_and_submit(pilot, "mistral")

        assert isinstance(app.screen, SecretScreen)
        await type_and_submit(pilot, "sk-test-key")

    assert saved == [("mistral", "sk-test-key")]


@pytest.mark.asyncio
async def test_login_with_provider_argument_skips_selection(
    vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[str] = []
    monkeypatch.setattr(
        "vibe.cli.textual_ui.app.save_api_key",
        lambda provider, key: saved.append(key) or Path("/tmp/.env"),
    )
    app = VibeApp(config=vibe_config)

    async with app.run_test() as pilot:
        await type_and_submit(pilot, "/login mistral")

        assert isinstance(app.screen, SecretScreen)
        await type_and_submit(pilot, "sk-direct")

    assert saved == ["sk-direct"]


@pytest.mark.asyncio
async def test_login_offers_subscription_for_oauth_provider(
    vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "vibe.cli.login.login_provider", lambda config, name: called.append(name)
    )
    app = VibeApp(config=vibe_config)
    monkeypatch.setattr(app, "_reload_config", _noop)

    async with app.run_test() as pilot:
        await type_and_submit(pilot, "/login anthropic")

        assert isinstance(app.screen, LoginScreen)
        await type_and_submit(pilot, "subscription")

        assert not isinstance(app.screen, LoginScreen)

    assert called == ["anthropic"]


@pytest.mark.asyncio
async def test_login_reports_unknown_provider(vibe_config: VibeConfig) -> None:
    app = VibeApp(config=vibe_config)

    async with app.run_test() as pilot:
        await type_and_submit(pilot, "/login nope")

        assert not isinstance(app.screen, LoginScreen)


@pytest.mark.asyncio
async def test_logout_removes_a_configured_credential(
    vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test-key")
    removed: list[str] = []
    monkeypatch.setattr(
        "vibe.cli.textual_ui.app.logout",
        lambda config, name: removed.append(name) or [f"removed {name}"],
    )
    app = VibeApp(config=vibe_config)

    async with app.run_test() as pilot:
        await type_and_submit(pilot, "/logout")

        assert isinstance(app.screen, LoginScreen)
        await type_and_submit(pilot, "mistral")

    assert removed == ["mistral"]


async def _noop() -> None:
    return None
