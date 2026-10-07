from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
import json
import sys
from typing import Any, TextIO
import uuid

from vibe.core.types import (
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    LLMMessage,
    OutputFormat,
    StreamEventType,
    ToolCallEvent,
    ToolResultEvent,
)

EVENT_SCHEMA_VERSION = 1


class OutputFormatter(ABC):
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self._messages: list[LLMMessage] = []
        self._final_response: str | None = None

    @abstractmethod
    def on_message_added(self, message: LLMMessage) -> None:
        pass

    @abstractmethod
    def on_event(self, event: BaseEvent) -> None:
        pass

    @abstractmethod
    def finalize(self) -> str | None:
        """Finalize output and return any final text to be printed.

        Returns:
            String to print, or None if formatter handles its own output
        """
        pass


class TextOutputFormatter(OutputFormatter):
    def on_message_added(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def on_event(self, event: BaseEvent) -> None:
        if isinstance(event, AssistantEvent):
            self._final_response = event.content

    def finalize(self) -> str | None:
        return self._final_response


class JsonOutputFormatter(OutputFormatter):
    def on_message_added(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def on_event(self, event: BaseEvent) -> None:
        pass

    def finalize(self) -> str | None:
        messages_data = [msg.model_dump(mode="json") for msg in self._messages]
        json.dump(messages_data, self.stream, indent=2)
        self.stream.write("\n")
        self.stream.flush()
        return None


class StreamingJsonOutputFormatter(OutputFormatter):
    def on_message_added(self, message: LLMMessage) -> None:
        json.dump(message.model_dump(mode="json"), self.stream)
        self.stream.write("\n")
        self.stream.flush()

    def on_event(self, event: BaseEvent) -> None:
        pass

    def finalize(self) -> str | None:
        return None


class VScodeJsonFormatter(OutputFormatter):
    """JSON formatter for VSCode extension integration.

    Emits structured streaming events that the VSCode extension can parse:
    - thread.started: Conversation begins (includes session_id)
    - item.updated: Incremental content chunks
    - tool.call: Tool execution requests
    - tool.result: Tool execution results
    - turn.completed: Agent turn finishes
    - error: Error events
    """

    def __init__(
        self, stream: TextIO = sys.stdout, session_id: str | None = None
    ) -> None:
        super().__init__(stream)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.thread_id = self.session_id  # Keep thread_id for backward compatibility
        self.current_item_id: str | None = None
        self.accumulated_content = ""
        self.accumulated_thinking = ""
        self.thread_started = False

    def _emit_event(self, event_type: StreamEventType, data: dict[str, Any]) -> None:
        """Emit a single JSON event to stdout."""
        event = {"type": event_type, "timestamp": datetime.now(UTC).isoformat(), **data}
        json.dump(event, self.stream)
        self.stream.write("\n")
        self.stream.flush()

    def _ensure_thread_started(self) -> None:
        """Emit thread.started event if not already emitted."""
        if not self.thread_started:
            self._emit_event(
                StreamEventType.THREAD_STARTED,
                {"thread_id": self.thread_id, "session_id": self.session_id},
            )
            self.thread_started = True

    def _generate_item_id(self) -> str:
        """Generate a unique item ID."""
        return f"msg_{uuid.uuid4().hex[:8]}"

    def on_message_added(self, message: LLMMessage) -> None:
        """Called when a complete message is added to the conversation."""
        self._messages.append(message)

    def on_event(self, event: BaseEvent) -> None:
        """Handle agent events and convert to streaming JSON events."""
        self._ensure_thread_started()

        if isinstance(event, AssistantEvent):
            # Handle assistant response (both content and reasoning)
            if not self.current_item_id:
                self.current_item_id = self._generate_item_id()
                self._emit_event(StreamEventType.TURN_STARTED, {})

            # Emit thinking/reasoning content if present
            if event.reasoning_content:
                reasoning_delta = event.reasoning_content[
                    len(self.accumulated_thinking) :
                ]
                if reasoning_delta:
                    self.accumulated_thinking = event.reasoning_content
                    self._emit_event(
                        StreamEventType.THINKING_UPDATED,
                        {
                            "item_id": self.current_item_id,
                            "delta": reasoning_delta,
                            "content": self.accumulated_thinking,
                        },
                    )

            # Emit regular content
            if event.content:
                content_delta = event.content[len(self.accumulated_content) :]
                if content_delta:
                    self.accumulated_content = event.content
                    self._emit_event(
                        StreamEventType.ITEM_UPDATED,
                        {
                            "item_id": self.current_item_id,
                            "role": "assistant",
                            "delta": content_delta,
                            "content": self.accumulated_content,
                        },
                    )

            # If this is the final event (turn completed)
            if event.stopped_by_middleware or event.completion_tokens > 0:
                self._emit_event(
                    StreamEventType.ITEM_COMPLETED,
                    {
                        "item_id": self.current_item_id,
                        "content": self.accumulated_content,
                    },
                )
                self._emit_event(
                    StreamEventType.TURN_COMPLETED,
                    {
                        "finish_reason": "end_turn",
                        "usage": {
                            "input_tokens": event.prompt_tokens,
                            "output_tokens": event.completion_tokens,
                        },
                    },
                )
                # Reset for next turn
                self.current_item_id = None
                self.accumulated_content = ""
                self.accumulated_thinking = ""

        elif isinstance(event, ToolCallEvent):
            # Emit tool call event
            self._emit_event(
                StreamEventType.TOOL_CALL,
                {
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "args": event.args.model_dump(mode="json"),
                },
            )

        elif isinstance(event, ToolResultEvent):
            # Emit tool result event
            result_data = event.result.model_dump(mode="json") if event.result else None
            self._emit_event(
                StreamEventType.TOOL_RESULT,
                {
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "result": str(result_data) if result_data else None,
                    "error": event.error,
                    "is_error": event.error is not None,
                    "skipped": event.skipped,
                    "skip_reason": event.skip_reason,
                    "duration": event.duration,
                },
            )

    def finalize(self) -> str | None:
        """Finalize output. Returns None as we handle our own output."""
        return None


class EventJsonlFormatter(OutputFormatter):
    """Versioned JSONL event stream documented in ``docs/cli-events.md``.

    Every line carries ``type``, ``schema`` and ``timestamp`` fields so clients
    can detect and adapt to schema changes. Unlike the VSCode formatter the
    payloads are stable, documented and free of tool-class specific data.
    """

    def __init__(
        self, stream: TextIO = sys.stdout, session_id: str | None = None
    ) -> None:
        super().__init__(stream)
        self.session_id = session_id or str(uuid.uuid4())
        self._turn = 0
        self._item_id: str | None = None
        self._content = ""
        self._reasoning = ""
        self._usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._cost_usd = 0.0
        self._emit(StreamEventType.SESSION_STARTED, {"session_id": self.session_id})

    def _emit(self, event_type: StreamEventType, data: dict[str, Any]) -> None:
        event = {
            "type": str(event_type),
            "schema": EVENT_SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }
        json.dump(event, self.stream)
        self.stream.write("\n")
        self.stream.flush()

    def _emit_event(self, event_type: StreamEventType, data: dict[str, Any]) -> None:
        """Alias used by the shared stdin approval callbacks."""
        self._emit(event_type, data)

    def emit_error(self, message: str) -> None:
        """Emit a top-level ``error`` event."""
        self._emit(StreamEventType.ERROR, {"message": message})

    def on_message_added(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def on_event(self, event: BaseEvent) -> None:
        match event:
            case AssistantEvent():
                self._handle_assistant_event(event)
            case ToolCallEvent():
                self._emit(
                    StreamEventType.TOOL_CALL,
                    {
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name,
                        "args": event.args.model_dump(mode="json"),
                    },
                )
            case ToolResultEvent():
                self._emit_tool_result(event)
            case CompactEndEvent():
                self._emit(
                    StreamEventType.SESSION_COMPACTED,
                    {
                        "tokens_before": event.old_context_tokens,
                        "tokens_after": event.new_context_tokens,
                    },
                )

    def _emit_tool_result(self, event: ToolResultEvent) -> None:
        match event:
            case ToolResultEvent(error=error) if error is not None:
                status = "error"
                text = error
            case ToolResultEvent(skipped=True, skip_reason=reason):
                status = "skipped"
                text = reason or ""
            case ToolResultEvent(result=result) if result is not None:
                status = "ok"
                text = result.model_dump_json()
            case _:
                status = "ok"
                text = ""

        self._emit(
            StreamEventType.TOOL_RESULT,
            {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "status": status,
                "text": text,
                "truncated": False,
            },
        )

    def _handle_assistant_event(self, event: AssistantEvent) -> None:
        if self._item_id is None:
            self._turn += 1
            self._item_id = f"msg_{uuid.uuid4().hex[:8]}"
            self._content = ""
            self._emit(StreamEventType.TURN_STARTED, {"turn": self._turn})

        if event.reasoning_content:
            reasoning_delta = event.reasoning_content[len(self._reasoning) :]
            if reasoning_delta:
                self._reasoning = event.reasoning_content
                self._emit(
                    StreamEventType.THINKING_UPDATED,
                    {
                        "item_id": self._item_id,
                        "delta": reasoning_delta,
                        "content": self._reasoning,
                    },
                )

        content_delta = event.content[len(self._content) :] if event.content else ""
        if content_delta:
            self._content = event.content
            self._emit(
                StreamEventType.MESSAGE_DELTA,
                {
                    "item_id": self._item_id,
                    "delta": content_delta,
                    "content": self._content,
                },
            )

        if event.cache_read_tokens:
            self._usage["cache_read"] = event.cache_read_tokens
        if event.cache_write_tokens:
            self._usage["cache_write"] = event.cache_write_tokens
        if event.prompt_tokens:
            self._usage["input"] = event.prompt_tokens
        if event.completion_tokens:
            self._usage["output"] = event.completion_tokens
        if event.cost_usd:
            self._cost_usd = event.cost_usd

        if self._is_final_turn_event(event):
            self._emit(
                StreamEventType.TURN_COMPLETED,
                {"usage": dict(self._usage), "cost_usd": round(self._cost_usd, 6)},
            )
            self._item_id = None
            self._content = ""
            self._reasoning = ""
            self._usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
            self._cost_usd = 0.0

    @staticmethod
    def _is_final_turn_event(event: AssistantEvent) -> bool:
        return event.stopped_by_middleware or event.completion_tokens > 0

    def finalize(self) -> str | None:
        return None


def create_formatter(
    format_type: OutputFormat,
    stream: TextIO = sys.stdout,
    session_id: str | None = None,
) -> OutputFormatter:
    formatters = {
        OutputFormat.TEXT: TextOutputFormatter,
        OutputFormat.JSON: JsonOutputFormatter,  # Original: all messages as JSON at end
        OutputFormat.STREAMING: StreamingJsonOutputFormatter,  # Original: each message as JSON
        OutputFormat.VSCODE: VScodeJsonFormatter,  # New: structured events for VSCode extension
        OutputFormat.EVENTS: EventJsonlFormatter,  # Versioned JSONL event stream
    }

    formatter_class = formatters.get(format_type, TextOutputFormatter)
    # VScodeJsonFormatter and EventJsonlFormatter support session_id parameter
    if format_type in {OutputFormat.VSCODE, OutputFormat.EVENTS} and session_id:
        return formatter_class(stream, session_id=session_id)
    return formatter_class(stream)
