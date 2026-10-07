from __future__ import annotations

from vibe.core.oauth.base import AnyOAuthToken, OAuthFlow
from vibe.core.oauth.claude import (
    ClaudeOAuthFlow,
    exchange_token,
    get_authorize_url,
    get_pkce_challenge,
    refresh_token,
)
from vibe.core.oauth.qwen import (
    DeviceCodeResponse,
    QwenOAuthFlow,
    get_pkce_challenge as qwen_get_pkce_challenge,
    poll_for_token as qwen_poll_for_token,
    refresh_token as qwen_refresh_token,
    request_device_code as qwen_request_device_code,
)
from vibe.core.oauth.registry import UnknownOAuthFlowError, get_flow, register_flow
from vibe.core.oauth.token import AnthropicOAuthToken, OAuthToken, QwenOAuthToken

__all__ = [
    "AnthropicOAuthToken",
    "AnyOAuthToken",
    "ClaudeOAuthFlow",
    "DeviceCodeResponse",
    "OAuthFlow",
    "OAuthToken",
    "QwenOAuthFlow",
    "QwenOAuthToken",
    "UnknownOAuthFlowError",
    "exchange_token",
    "get_authorize_url",
    "get_flow",
    "get_pkce_challenge",
    "qwen_get_pkce_challenge",
    "qwen_poll_for_token",
    "qwen_refresh_token",
    "qwen_request_device_code",
    "refresh_token",
    "register_flow",
]
