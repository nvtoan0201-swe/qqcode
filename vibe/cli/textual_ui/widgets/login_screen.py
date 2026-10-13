from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from vibe.core.autocompletion.fuzzy import fuzzy_match

SELECTED_CLASS = "login-option-selected"
DIALOG_CLASS = "login-dialog"
TITLE_CLASS = "login-title"
SEARCH_CLASS = "login-search"
LIST_CLASS = "login-list"
EMPTY_CLASS = "login-empty"
HELP_CLASS = "login-help"
SECRET_CLASS = "login-secret"


@dataclass(frozen=True)
class SelectorEntry:
    """One selectable row: a provider, an auth method or a credential."""

    label: str
    value: Any = None
    detail: str = ""
    configured: bool = False

    def searchable_text(self) -> str:
        return f"{self.label} {self.detail}"


class LoginScreen(ModalScreen[Any]):
    """Modal selector with fuzzy search, used for login and logout flows."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, title: str, entries: list[SelectorEntry]) -> None:
        super().__init__()
        self.title_text = title
        self.entries = entries
        self.filtered = list(entries)
        self.selected_index = 0
        self.rows: list[tuple[SelectorEntry, Static]] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes=DIALOG_CLASS):
            yield Static(self.title_text, classes=TITLE_CLASS)
            yield Input(placeholder="Search…", classes=SEARCH_CLASS)
            with VerticalScroll(classes=LIST_CLASS):
                for entry in self.entries:
                    widget = Static("", classes="login-option")
                    self.rows.append((entry, widget))
                    yield widget
                yield Static("  No matches.", classes=EMPTY_CLASS)
            yield Static("↑↓ navigate  Enter select  Esc cancel", classes=HELP_CLASS)

    def on_mount(self) -> None:
        self._update_display()
        self.query_one(f".{SEARCH_CLASS}", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_select()

    def _apply_filter(self, pattern: str) -> None:
        pattern = pattern.strip()
        scored = [
            (match.score, entry)
            for entry in self.entries
            if (match := fuzzy_match(pattern, entry.searchable_text())).matched
        ]
        self.filtered = [
            entry for _, entry in sorted(scored, key=lambda item: item[0], reverse=True)
        ]
        self.selected_index = 0
        self._update_display()

    def _update_display(self) -> None:
        visible = {id(entry) for entry in self.filtered}
        selected = self.filtered[self.selected_index] if self.filtered else None
        for entry, widget in self.rows:
            is_selected = entry is selected
            widget.display = id(entry) in visible
            widget.update(self._format_entry(entry, selected=is_selected))
            widget.set_class(is_selected, SELECTED_CLASS)
        self.query_one(f".{EMPTY_CLASS}", Static).display = not self.filtered

    @staticmethod
    def _format_entry(entry: SelectorEntry, *, selected: bool) -> str:
        cursor = "›" if selected else " "
        detail = f"  [dim]{entry.detail}[/dim]" if entry.detail else ""
        return f"{cursor} {entry.label}{detail}"

    def action_move_up(self) -> None:
        if not self.filtered:
            return
        self.selected_index = (self.selected_index - 1) % len(self.filtered)
        self._update_display()

    def action_move_down(self) -> None:
        if not self.filtered:
            return
        self.selected_index = (self.selected_index + 1) % len(self.filtered)
        self._update_display()

    def action_select(self) -> None:
        if not self.filtered:
            return
        self.dismiss(self.filtered[self.selected_index].value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SecretScreen(ModalScreen[str | None]):
    """Modal asking for a secret (an API key), masked while typing."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    def __init__(self, title: str, placeholder: str) -> None:
        super().__init__()
        self.title_text = title
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(classes=DIALOG_CLASS):
            yield Static(self.title_text, classes=TITLE_CLASS)
            yield Input(
                password=True, placeholder=self.placeholder, classes=SECRET_CLASS
            )
            yield Static("Enter save  Esc cancel", classes=HELP_CLASS)

    def on_mount(self) -> None:
        self.query_one(f".{SECRET_CLASS}", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
