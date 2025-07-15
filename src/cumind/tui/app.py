import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from .screens.main_menu import MainMenuScreen
from .screens.training import TrainingScreen


class CuMindTUI(App):
    
    CSS_PATH = Path(__file__).parent / "cumind.tcss"
    TITLE = "CuMind - JAX-based Reinforcement Learning"
    SUB_TITLE = "Monte Carlo Tree Search + Neural Networks"
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("m", "show_main", "Main Menu"),
        Binding("t", "show_training", "Training"),
        Binding("i", "show_inference", "Inference"),
        Binding("c", "show_config", "Config"),
    ]

    SCREENS = {
        "main_menu": MainMenuScreen,
        "training": TrainingScreen,
    }

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("main_menu")

    def action_show_main(self) -> None:
        self.push_screen("main_menu")

    def action_show_training(self) -> None:
        self.push_screen("training")

    def action_show_inference(self) -> None:
        self.notify("Inference viewer not implemented yet", severity="info")

    def action_show_config(self) -> None:
        self.notify("Configuration editor not implemented yet", severity="info")


def run_tui() -> None:
    app = CuMindTUI()
    app.run()


if __name__ == "__main__":
    run_tui()