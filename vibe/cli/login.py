from __future__ import annotations

import asyncio
import webbrowser

from rich import print as rprint
from rich.prompt import Prompt

from vibe.core.auth import (
    AuthKind,
    ProviderAuth,
    all_provider_auths,
    find_provider,
    logout,
    provider_auths,
    save_api_key,
)
from vibe.core.config import (
    ProviderConfig,
    VibeConfig,
    save_oauth_token,
    skip_api_key_check,
)
from vibe.core.oauth import (
    exchange_token,
    get_authorize_url,
    get_pkce_challenge,
    qwen_get_pkce_challenge,
    qwen_poll_for_token,
    qwen_request_device_code,
)


async def _login_anthropic_async() -> None:
    """Interactive OAuth login flow for Anthropic."""
    rprint("\n[bold blue]Anthropic Claude Max Login[/bold blue]\n")

    # Generate PKCE challenge
    verifier, challenge = get_pkce_challenge()
    url = get_authorize_url(challenge, state=verifier)

    rprint("[dim]Opening browser for authentication...[/dim]\n")
    rprint(f"If the browser doesn't open, visit this URL:\n[link={url}]{url}[/link]\n")

    # Try to open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Get authorization code from user
    code = Prompt.ask("\n[bold]Paste the authorization code from the browser[/bold]")
    code = code.strip()

    if not code:
        rprint("[red]No authorization code provided. Login cancelled.[/red]")
        return

    rprint("\n[dim]Exchanging code for tokens...[/dim]")

    try:
        token = await exchange_token(code, verifier)
    except Exception as e:
        rprint(f"[red]Failed to exchange token: {e}[/red]")
        return

    # Save to config
    try:
        save_oauth_token("anthropic", token)
        rprint("\n[bold green]Successfully logged in to Anthropic![/bold green]")
        rprint(
            "[dim]You can now use Claude models. Run 'qqcode' and use /models to select a model.[/dim]\n"
        )
    except Exception as e:
        rprint(f"[red]Failed to save token: {e}[/red]")


def login_anthropic() -> None:
    """Run the Anthropic OAuth login flow."""
    asyncio.run(_login_anthropic_async())


async def _login_qwen_async() -> None:
    """Interactive OAuth login flow for Qwen (device authorization)."""
    rprint("\n[bold blue]Qwen OAuth Login[/bold blue]\n")

    # Generate PKCE challenge
    verifier, challenge = qwen_get_pkce_challenge()

    rprint("[dim]Requesting device authorization...[/dim]\n")

    try:
        device_response = await qwen_request_device_code(challenge)
    except Exception as e:
        rprint(f"[red]Failed to get device code: {e}[/red]")
        return

    # Show verification URL
    url = device_response.verification_uri_complete
    rprint(f"Please visit this URL to authorize:\n[link={url}]{url}[/link]\n")

    # Try to open browser
    try:
        webbrowser.open(url)
        rprint("[dim]Browser opened. Please complete authorization...[/dim]\n")
    except Exception:
        rprint("[dim]Could not open browser. Please visit the URL manually.[/dim]\n")

    # Define status callback for progress updates
    def status_callback(status: str) -> None:
        rprint(f"[dim]{status}[/dim]")

    rprint("[dim]Waiting for authorization (press Ctrl+C to cancel)...[/dim]\n")

    try:
        token = await qwen_poll_for_token(
            device_code=device_response.device_code,
            verifier=verifier,
            expires_in=device_response.expires_in,
            status_callback=status_callback,
        )
    except KeyboardInterrupt:
        rprint("\n[yellow]Login cancelled.[/yellow]")
        return
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            rprint("\n[red]Authorization timed out. Please try again.[/red]")
        else:
            rprint(f"\n[red]Failed to get token: {e}[/red]")
        return

    # Save to config
    try:
        save_oauth_token("qwen", token)
        rprint("\n[bold green]Successfully logged in to Qwen![/bold green]")
        rprint(
            "[dim]You can now use Qwen models. Run 'qqcode' and use /model to select a model.[/dim]\n"
        )
    except Exception as e:
        rprint(f"[red]Failed to save token: {e}[/red]")


def login_qwen() -> None:
    """Run the Qwen OAuth login flow."""
    asyncio.run(_login_qwen_async())


def _select(prompt: str, options: list[str]) -> int:
    for index, option in enumerate(options, start=1):
        rprint(f"  [bold]{index}.[/] {option}")
    choices = [str(index) for index in range(1, len(options) + 1)]
    answer = Prompt.ask(f"\n{prompt}", choices=choices, default="1")
    return int(answer) - 1


def _status_suffix(auth: ProviderAuth) -> str:
    if auth.source:
        return f" [green]✓ {auth.source}[/]"
    return " [dim]not configured[/]"


def _login_oauth(provider_name: str) -> None:
    match provider_name:
        case "anthropic":
            login_anthropic()
        case "qwen":
            login_qwen()
        case _:
            rprint(
                f"[red]No interactive login flow is available for "
                f"'{provider_name}'.[/red]"
            )


def _login_api_key(provider: ProviderConfig) -> None:
    env_var = provider.api_key_env_var
    key = Prompt.ask(
        f"[bold]Enter {env_var} for {provider.name}[/bold]", password=True
    ).strip()
    if not key:
        rprint("[red]No API key provided. Login cancelled.[/red]")
        return

    try:
        path = save_api_key(provider, key)
    except (OSError, ValueError) as exc:
        rprint(f"[red]Failed to save API key: {exc}[/red]")
        return

    rprint(f"\n[bold green]Saved {env_var} to {path}[/bold green]")
    rprint("[dim]New QQcode sessions will use this key.[/dim]\n")


def _run_auth(config: VibeConfig, auth: ProviderAuth) -> None:
    provider = find_provider(config, auth.provider)
    if provider is None:
        rprint(f"[red]Unknown provider '{auth.provider}'.[/red]")
        return
    match auth.kind:
        case "oauth":
            _login_oauth(provider.name)
        case "api_key":
            _login_api_key(provider)


def login_provider(config: VibeConfig, name: str) -> None:
    """Log in to one provider, asking how when it offers several methods."""
    auths = provider_auths(config, name)
    match auths:
        case []:
            rprint(
                f"[red]Unknown provider '{name}'. Run 'qqcode login' to list "
                "the configured providers.[/red]"
            )
        case [auth]:
            _run_auth(config, auth)
        case _:
            index = _select(
                f"Select authentication method for {name}:",
                [f"{auth.label}{_status_suffix(auth)}" for auth in auths],
            )
            _run_auth(config, auths[index])


def _select_and_login(config: VibeConfig, kind: AuthKind) -> None:
    auths = all_provider_auths(config, kind)
    if not auths:
        rprint(f"[red]No {kind} providers are configured.[/red]")
        return
    index = _select(
        "Select provider:",
        [
            f"{auth.provider} [dim]({auth.label})[/]{_status_suffix(auth)}"
            for auth in auths
        ],
    )
    _run_auth(config, auths[index])


def _load_login_config() -> VibeConfig:
    """Load the configuration without requiring a usable API key."""
    with skip_api_key_check():
        return VibeConfig.load()


def run_login_command(argv: list[str]) -> int:
    """Run ``qqcode login [provider]``."""
    config = _load_login_config()

    match argv:
        case [name]:
            login_provider(config, name)
            return 0
        case []:
            auths = all_provider_auths(config)
            if not auths:
                rprint("[red]No providers are configured.[/red]")
                return 1
            kinds: set[AuthKind] = {auth.kind for auth in auths}
            kind: AuthKind
            if len(kinds) > 1:
                index = _select(
                    "Select authentication method:",
                    [
                        "Sign in with a subscription or account",
                        "Sign in with an API key",
                    ],
                )
                kind = "oauth" if index == 0 else "api_key"
            else:
                kind = next(iter(kinds))
            _select_and_login(config, kind)
            return 0
        case _:
            rprint("[red]Usage: qqcode login [provider][/red]")
            return 1


def _logout(config: VibeConfig, name: str) -> None:
    notes = logout(config, name)
    if not notes:
        rprint(f"[dim]No stored credentials found for '{name}'.[/dim]")
        return
    for note in notes:
        rprint(f"[green]✓[/] {note}")
    rprint("[dim]Environment variables set outside .env are not affected.[/dim]")


def run_logout_command(argv: list[str]) -> int:
    """Run ``qqcode logout [provider]``."""
    config = _load_login_config()

    match argv:
        case [name]:
            _logout(config, name)
            return 0
        case []:
            auths = [auth for auth in all_provider_auths(config) if auth.configured]
            if not auths:
                rprint("[dim]No stored credentials to remove.[/dim]")
                return 0
            index = _select(
                "Select credential to remove:",
                [f"{auth.provider} ({auth.source})" for auth in auths],
            )
            _logout(config, auths[index].provider)
            return 0
        case _:
            rprint("[red]Usage: qqcode logout [provider][/red]")
            return 1
