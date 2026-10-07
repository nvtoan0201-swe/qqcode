from __future__ import annotations

from typing import TYPE_CHECKING

from vibe.core.context.compactor import CompactionResult
from vibe.core.types import LLMMessage, LLMUsage

if TYPE_CHECKING:
    from vibe.core.agent import Agent
    from vibe.core.interaction_logger import InteractionLogger


class SessionRecorder:
    """Thin bridge between the agent lifecycle and the session store."""

    def __init__(self, logger: InteractionLogger, agent: Agent | None = None) -> None:
        self._logger = logger
        self._agent = agent

    @property
    def logger(self) -> InteractionLogger:
        return self._logger

    def on_message_added(
        self, message: LLMMessage, parent_id: str | None = None
    ) -> str | None:
        return self._logger.append_message(message, parent_id)

    def on_usage(self, usage: LLMUsage, cost: float = 0.0) -> None:
        self._logger.append_usage(usage, cost)

    def on_model_change(self, alias: str, provider: str) -> None:
        self._logger.append_model_change(alias, provider)

    def on_compaction(
        self, result: CompactionResult, first_kept_id: str | None = None
    ) -> None:
        self._logger.append_compaction(result, first_kept_id)

    def entry_id_for(self, message: LLMMessage) -> str | None:
        return self._logger.entry_id_for(message)

    async def flush(self) -> None:
        agent = self._agent
        if agent is None:
            return
        await self._logger.save_interaction(
            agent.messages,
            agent.stats,
            agent.config,
            agent.tool_manager,
            agent._additional_directory_contexts,
        )
