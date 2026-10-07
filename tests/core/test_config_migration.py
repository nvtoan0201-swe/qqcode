from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tomllib

import pytest
import tomli_w

from vibe.core import config
from vibe.core.config import CompactionConfig, ModelConfig, VibeConfig


def _restore_dump_config(config_file: Path):
    original_dump_config = VibeConfig.dump_config

    def real_dump_config(cls, config_dict: dict) -> None:
        try:
            with config_file.open("wb") as f:
                tomli_w.dump(config_dict, f)
        except OSError:
            config_file.write_text(
                "\n".join(
                    f"{k} = {v!r}" for k, v in config_dict.items() if v is not None
                ),
                encoding="utf-8",
            )

    VibeConfig.dump_config = classmethod(real_dump_config)  # type: ignore[assignment]
    return original_dump_config


@contextmanager
def _migrate_config_file(tmp_path: Path, content: str):
    config_file = tmp_path / "config.toml"
    config_file.write_text(content, encoding="utf-8")

    original_config_file = config.CONFIG_FILE
    original_dump_config = _restore_dump_config(config_file)

    try:
        config.CONFIG_FILE = config_file
        VibeConfig._migrate()
        yield config_file
    finally:
        config.CONFIG_FILE = original_config_file
        VibeConfig.dump_config = original_dump_config


def _load_migrated_config(config_file: Path) -> dict:
    with config_file.open("rb") as f:
        return tomllib.load(f)


def test_default_compaction_config() -> None:
    config = VibeConfig()

    assert config.compaction.enabled is True
    assert config.compaction.reserve_tokens == 16_384
    assert config.compaction.keep_recent_tokens == 20_000
    assert config.compaction.threshold_ratio == pytest.approx(0.90)
    assert config.compaction.prompt == "summary"
    assert config.context_warnings is True


def test_old_config_file_without_compaction_still_loads(tmp_path: Path) -> None:
    content = 'active_model = "devstral-2"\nauto_compact_threshold = 100000\n'

    with _migrate_config_file(tmp_path, content) as config_file:
        data = _load_migrated_config(config_file)
        assert data["auto_compact_threshold"] == 200_000
        assert "compaction" not in data
        config = VibeConfig(**data)

    assert config.compaction == CompactionConfig()
    assert config.context_warnings is True


def test_migrate_keeps_existing_compaction_section(tmp_path: Path) -> None:
    content = "auto_compact_threshold = 200000\n\n[compaction]\nenabled = false\n"

    with _migrate_config_file(tmp_path, content) as config_file:
        data = _load_migrated_config(config_file)

    assert data["compaction"]["enabled"] is False


def test_compaction_section_is_parsed() -> None:
    config = VibeConfig(
        compaction=CompactionConfig(enabled=False, keep_recent_tokens=5)
    )

    assert config.compaction.enabled is False
    assert config.compaction.keep_recent_tokens == 5
    assert config.compaction.reserve_tokens == 16_384


def test_get_compaction_threshold_reserves_tokens() -> None:
    model = ModelConfig(
        name="test-model", provider="mistral", alias="test-model", context_limit=100_000
    )
    config = VibeConfig(active_model="test-model", models=[model])

    assert (
        config.get_compaction_threshold() == 100_000 - config.compaction.reserve_tokens
    )
