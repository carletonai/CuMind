"""Training dashboard screen for CuMind TUI."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, ProgressBar, Static


class TrainingScreen(Screen):
    """Training dashboard with real-time metrics."""

    BINDINGS = [
        ("p", "toggle_pause", "Pause/Resume"),
        ("s", "stop_training", "Stop"),
        ("m", "back_to_menu", "Main Menu"),
    ]

    def compose(self) -> ComposeResult:
        """Create the training dashboard layout."""
        with Container():
            yield Static("🧠 CuMind Training Dashboard", classes="screen-title")
            
            # Progress section
            with Horizontal(classes="progress-section"):
                with Vertical():
                    yield Static("Episode Progress", classes="section-title")
                    yield ProgressBar(total=500, progress=234, classes="episode-progress")
                    yield Static("Episode: 234/500 (46.8%)", classes="progress-text")
                
                with Vertical():
                    yield Static("Current Episode", classes="section-title")
                    yield Static("Steps: 142", classes="metric")
                    yield Static("Reward: 196", classes="metric")
                    yield Static("Action: LEFT", classes="metric")
            
            # Metrics section
            with Horizontal(classes="metrics-section"):
                with Vertical():
                    yield Static("📊 Training Metrics", classes="section-title")
                    yield Static("Avg Reward: 187.3", classes="metric")
                    yield Static("Loss: 0.0234", classes="metric")
                    yield Static("Learning Rate: 0.001", classes="metric")
                    yield Static("Memory Size: 847/1000", classes="metric")
                
                with Vertical():
                    yield Static("🎯 Performance", classes="section-title")
                    yield Static("Best Reward: 250", classes="metric")
                    yield Static("Success Rate: 78%", classes="metric")
                    yield Static("Steps/sec: 45.2", classes="metric")
                    yield Static("Training Time: 12m 34s", classes="metric")
            
            # Chart placeholder
            with Container(classes="chart-section"):
                yield Static("📈 Reward History", classes="section-title")
                yield Static(self._create_chart_placeholder(), classes="chart")
            
            # Controls
            with Horizontal(classes="controls"):
                yield Button("⏸️  Pause", id="pause", variant="warning")
                yield Button("⏹️  Stop", id="stop", variant="error")
                yield Button("📁 Save Checkpoint", id="save", variant="success")
                yield Button("🏠 Main Menu", id="menu", variant="default")

    def _create_chart_placeholder(self) -> str:
        """Create a simple ASCII chart placeholder."""
        return """
    250 ┤                           ╭─        
    200 ┤                      ╭────╯         
    150 ┤                 ╭────╯             
    100 ┤            ╭────╯                  
     50 ┤       ╭────╯                       
      0 └────────────────────────────────────
        0    50   100  150  200  234
        """

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "pause":
            self.action_toggle_pause()
        elif event.button.id == "stop":
            self.action_stop_training()
        elif event.button.id == "save":
            self.app.notify("Checkpoint saved!", severity="success")
        elif event.button.id == "menu":
            self.action_back_to_menu()

    def action_toggle_pause(self) -> None:
        """Toggle training pause/resume."""
        self.app.notify("Training paused/resumed", severity="info")

    def action_stop_training(self) -> None:
        """Stop training."""
        self.app.notify("Training stopped", severity="warning")
        self.action_back_to_menu()

    def action_back_to_menu(self) -> None:
        """Return to main menu."""
        self.app.pop_screen()