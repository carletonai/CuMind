"""Main menu screen for CuMind TUI."""

from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static


class MainMenuScreen(Screen):
    """Main menu screen with navigation options."""

    BINDINGS = [
        ("t", "start_training", "Start Training"),
        ("i", "load_inference", "Load & Inference"),
        ("c", "edit_config", "Edit Config"),
        ("h", "show_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        """Create the main menu layout."""
        with Center():
            with Vertical(classes="main-menu"):
                yield Static("Welcome to CuMind TUI", classes="title")
                yield Static("JAX-based Reinforcement Learning Framework", classes="subtitle")
                
                with Vertical(classes="menu-options"):
                    yield Button("Start New Training", id="start_training", variant="primary")
                    yield Button("Load Checkpoint & Inference", id="load_inference", variant="default")
                    yield Button("⚙️  Edit Configuration", id="edit_config", variant="default")
                    yield Button("View Training History", id="view_history", variant="default")
                    yield Button("❓ Help & Documentation", id="show_help", variant="default")
                
                with Horizontal(classes="info-panel"):
                    with Vertical():
                        yield Static("Recent Activity", classes="panel-title")
                        yield Static("• No recent training runs", classes="info-item")
                        yield Static("• No checkpoints found", classes="info-item")
                    
                    with Vertical():
                        yield Static("Quick Stats", classes="panel-title")
                        yield Static("• Environment: CartPole-v1", classes="info-item")
                        yield Static("• Episodes: 0/500", classes="info-item")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "start_training":
            self.app.push_screen("training")
        elif event.button.id == "load_inference":
            self.app.notify("Inference screen not implemented yet", severity="info")
        elif event.button.id == "edit_config":
            self.app.notify("Configuration editor not implemented yet", severity="info")
        elif event.button.id == "view_history":
            self.app.notify("Training history viewer not implemented yet", severity="info")
        elif event.button.id == "show_help":
            self.app.notify("Help documentation not implemented yet", severity="info")

    def action_start_training(self) -> None:
        """Start training action."""
        self.app.push_screen("training")

    def action_load_inference(self) -> None:
        """Load inference action."""
        self.app.notify("Inference screen not implemented yet", severity="info")

    def action_edit_config(self) -> None:
        """Edit config action."""
        self.app.notify("Configuration editor not implemented yet", severity="info")

    def action_show_help(self) -> None:
        """Show help action."""
        self.app.notify("Help documentation not implemented yet", severity="info")