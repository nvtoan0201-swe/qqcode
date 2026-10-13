from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import dotenv_values
import pytest

from vibe.cli import login as login_cli
from vibe.core.auth import (
    all_provider_auths,
    logout,
    provider_auths,
    remove_api_key,
    save_api_key,
)
from vibe.core.config import (
    Backend,
    ProviderConfig,
    VibeConfig,
    clear_oauth_token,
    config_file_override,
)
from vibe.core.oauth import AnthropicOAuthToken


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr("vibe.core.auth.GLOBAL_ENV_FILE", path)
    return path


@pytest.fixture
def config(tmp_path: Path) -> VibeConfig:
    return VibeConfig(
        workdir=tmp_path,
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


def test_provider_auths_lists_oauth_and_api_key(config: VibeConfig) -> None:
    kinds = {auth.kind for auth in provider_auths(config, "anthropic")}
    assert kinds == {"oauth", "api_key"}

    mistral = provider_auths(config, "mistral")
    assert [auth.kind for auth in mistral] == ["api_key"]
    assert mistral[0].env_var == "MISTRAL_API_KEY"


def test_unknown_provider_has_no_auths(config: VibeConfig) -> None:
    assert provider_auths(config, "nope") == []


def test_api_key_source_reports_env_and_dotenv(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert provider_auths(config, "mistral")[0].source is None

    env_file.write_text("MISTRAL_API_KEY=from-file\n", encoding="utf-8")
    assert provider_auths(config, "mistral")[0].source == ".env: MISTRAL_API_KEY"

    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    assert provider_auths(config, "mistral")[0].source == "env: MISTRAL_API_KEY"


def test_oauth_token_reports_configured(config: VibeConfig) -> None:
    provider = config.providers[1]
    provider.oauth_token = AnthropicOAuthToken.from_token_response(
        "access", "refresh", 3600
    )

    oauth = next(
        auth for auth in provider_auths(config, "anthropic") if auth.kind == "oauth"
    )

    assert oauth.configured
    assert oauth.source == "OAuth token"
    assert oauth.subscription


def test_save_api_key_writes_dotenv_and_environment(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    provider = config.providers[0]

    path = save_api_key(provider, "secret-key")

    assert path == env_file
    assert dotenv_values(env_file)["MISTRAL_API_KEY"] == "secret-key"
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert os.environ["MISTRAL_API_KEY"] == "secret-key"


def test_remove_api_key_clears_dotenv_and_environment(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    provider = config.providers[0]
    save_api_key(provider, "secret-key")

    assert remove_api_key(provider) is True

    assert "MISTRAL_API_KEY" not in dotenv_values(env_file)
    assert "MISTRAL_API_KEY" not in os.environ
    assert remove_api_key(provider) is False


def test_clear_oauth_token_removes_it_from_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[[providers]]\n"
        'name = "anthropic"\n'
        'api_base = "https://api.anthropic.com"\n'
        'api_key_env_var = "ANTHROPIC_API_KEY"\n'
        "[providers.oauth_token]\n"
        'access_token = "access"\n'
        'refresh_token = "refresh"\n'
        "expires_in = 3600\n"
        "expires_at = 9999999999\n",
        encoding="utf-8",
    )

    with config_file_override(config_file):
        assert clear_oauth_token("anthropic") is True
        assert clear_oauth_token("anthropic") is False

    assert "oauth_token" not in config_file.read_text(encoding="utf-8")


def test_logout_removes_every_credential(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    provider = config.providers[0]
    save_api_key(provider, "secret-key")

    notes = logout(config, "mistral")

    assert len(notes) == 1
    assert "MISTRAL_API_KEY" in notes[0]
    assert logout(config, "mistral") == []
    assert all_provider_auths(config, "api_key")[0].source is None


def test_login_command_saves_key_for_named_provider(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(VibeConfig, "load", classmethod(lambda cls, **_: config))
    monkeypatch.setattr(login_cli.Prompt, "ask", lambda *args, **kwargs: "typed-key")

    exit_code = login_cli.run_login_command(["mistral"])

    assert exit_code == 0
    assert dotenv_values(env_file)["MISTRAL_API_KEY"] == "typed-key"


def test_login_command_dispatches_oauth_flows(
    config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(VibeConfig, "load", classmethod(lambda cls, **_: config))
    called: list[str] = []
    monkeypatch.setattr(
        login_cli, "login_anthropic", lambda: called.append("anthropic")
    )
    monkeypatch.setattr(login_cli.Prompt, "ask", lambda *args, **kwargs: "1")

    assert login_cli.run_login_command(["anthropic"]) == 0

    assert called == ["anthropic"]


def test_login_command_interactive_flow(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(VibeConfig, "load", classmethod(lambda cls, **_: config))
    answers = iter(["2", "1", "interactive-key"])
    monkeypatch.setattr(login_cli.Prompt, "ask", lambda *args, **kwargs: next(answers))

    assert login_cli.run_login_command([]) == 0

    assert dotenv_values(env_file)["MISTRAL_API_KEY"] == "interactive-key"


def test_logout_command_removes_named_credential(
    config: VibeConfig, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    save_api_key(config.providers[0], "secret-key")
    monkeypatch.setattr(VibeConfig, "load", classmethod(lambda cls, **_: config))

    assert login_cli.run_logout_command(["mistral"]) == 0

    assert "MISTRAL_API_KEY" not in dotenv_values(env_file)


def test_entrypoint_routes_login_and_logout(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe.cli import entrypoint

    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        entrypoint, "run_login_command", lambda argv: calls.append(("login", argv)) or 0
    )
    monkeypatch.setattr(
        entrypoint,
        "run_logout_command",
        lambda argv: calls.append(("logout", argv)) or 0,
    )

    monkeypatch.setattr(sys, "argv", ["qqcode", "login", "mistral"])
    with pytest.raises(SystemExit):
        entrypoint._handle_subcommands()

    monkeypatch.setattr(sys, "argv", ["qqcode", "logout"])
    with pytest.raises(SystemExit):
        entrypoint._handle_subcommands()

    assert calls == [("login", ["mistral"]), ("logout", [])]
