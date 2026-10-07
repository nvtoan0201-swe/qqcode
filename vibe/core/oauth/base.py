from __future__ import annotations

from typing import ClassVar, Protocol

from vibe.core.oauth.token import AnthropicOAuthToken, QwenOAuthToken

AnyOAuthToken = AnthropicOAuthToken | QwenOAuthToken


class OAuthFlow(Protocol):
    """Provider login/refresh flow registered under a provider name."""

    name: ClassVar[str]
    is_subscription: ClassVar[bool]

    async def refresh(self, refresh_token: str) -> AnyOAuthToken: ...
