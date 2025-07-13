"""Interactive Rich-based TUI for CuMind with keyboard controls."""

import asyncio
import sys
import termios
import tty
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box

from .rich_app import Job, JobStatus, TrainingMetrics, CuMindRichTUI
from .job_manager import JobManager, JobConfig


class InteractiveCuMindTUI(CuMindRichTUI):
    """Interactive Rich TUI with keyboard controls and job management."""
    
    def __init__(self):
        super().__init__()
        self.job_manager = JobManager(max_concurrent_jobs=4)
        self.selected_job_index = 0
        self.show_help = False
        self.input_mode = None  # None, 'new_job', 'config_path'
        self.input_buffer = ""
        self.status_message = ""
        self.status_message_time = 0
        
    def handle_key(self, key: str):
        """Handle keyboard input."""
        if self.input_mode:
            self._handle_input_mode(key)
        else:
            self._handle_navigation_mode(key)
            
    def _handle_navigation_mode(self, key: str):
        """Handle keys in navigation mode."""
        if key == 'q':
            self.running = False
        elif key == 'h':
            self.show_help = not self.show_help
        elif key == 'n':
            self.input_mode = 'new_job'
            self.input_buffer = ""
            self.status_message = "Enter job name: "
        elif key == 'p':
            self._toggle_pause_selected_job()
        elif key == 's':
            self._stop_selected_job()
        elif key == '\x1b[A':  # Up arrow
            self._move_selection_up()
        elif key == '\x1b[B':  # Down arrow
            self._move_selection_down()
        elif key == '\r':  # Enter
            self._view_selected_job()
        elif key == 'r':
            self._refresh_jobs()
            
    def _handle_input_mode(self, key: str):
        """Handle keys in input mode."""
        if key == '\x1b':  # Escape
            self.input_mode = None
            self.input_buffer = ""
            self.status_message = "Input cancelled"
            self.status_message_time = time.time()
        elif key == '\r':  # Enter
            if self.input_mode == 'new_job':
                job_name = self.input_buffer.strip()
                if job_name:
                    self.input_mode = 'config_path'
                    self.input_buffer = ""
                    self.status_message = f"Enter config path for '{job_name}': "
                else:
                    self.status_message = "Job name cannot be empty"
                    self.status_message_time = time.time()
            elif self.input_mode == 'config_path':
                config_path = self.input_buffer.strip()
                self._create_new_job(config_path)
                self.input_mode = None
                self.input_buffer = ""
        elif key == '\x7f':  # Backspace
            self.input_buffer = self.input_buffer[:-1]
        elif len(key) == 1 and 32 <= ord(key) <= 126:  # Printable characters
            self.input_buffer += key
            
    def _create_new_job(self, config_path: str):
        """Create a new training job."""
        try:
            # Extract job name from previous input
            job_name = self.status_message.split("'")[1].split("'")[0]
            
            config = JobConfig(
                config_path=config_path,
                job_name=job_name,
            )
            
            job_id = self.job_manager.create_job(config)
            
            # Create Job object for display
            job = Job(
                id=job_id,
                name=job_name,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
            )
            self.jobs[job_id] = job
            
            self.status_message = f"Job '{job_name}' created successfully"
            self.status_message_time = time.time()
            
            # Start the job
            self.job_manager.start_job(job_id)
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
        except Exception as e:
            self.status_message = f"Error creating job: {str(e)}"
            self.status_message_time = time.time()
            
    def _toggle_pause_selected_job(self):
        """Toggle pause/resume for selected job."""
        job = self._get_selected_job()
        if job and job.status in [JobStatus.RUNNING, JobStatus.PAUSED]:
            try:
                if job.status == JobStatus.RUNNING:
                    self.job_manager.pause_job(job.id)
                    job.status = JobStatus.PAUSED
                    self.status_message = f"Job '{job.name}' paused"
                else:
                    self.job_manager.resume_job(job.id)
                    job.status = JobStatus.RUNNING
                    self.status_message = f"Job '{job.name}' resumed"
                self.status_message_time = time.time()
            except Exception as e:
                self.status_message = f"Error: {str(e)}"
                self.status_message_time = time.time()
                
    def _stop_selected_job(self):
        """Stop the selected job."""
        job = self._get_selected_job()
        if job and job.status in [JobStatus.RUNNING, JobStatus.PAUSED]:
            try:
                self.job_manager.stop_job(job.id)
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                self.status_message = f"Job '{job.name}' stopped"
                self.status_message_time = time.time()
            except Exception as e:
                self.status_message = f"Error: {str(e)}"
                self.status_message_time = time.time()
                
    def _move_selection_up(self):
        """Move selection up in job list."""
        if self.jobs:
            self.selected_job_index = max(0, self.selected_job_index - 1)
            
    def _move_selection_down(self):
        """Move selection down in job list."""
        if self.jobs:
            self.selected_job_index = min(len(self.jobs) - 1, self.selected_job_index + 1)
            
    def _get_selected_job(self) -> Optional[Job]:
        """Get currently selected job."""
        if self.jobs:
            job_ids = list(self.jobs.keys())
            if 0 <= self.selected_job_index < len(job_ids):
                return self.jobs[job_ids[self.selected_job_index]]
        return None
        
    def _view_selected_job(self):
        """View details of selected job."""
        job = self._get_selected_job()
        if job:
            self.active_job_id = job.id
            
    def _refresh_jobs(self):
        """Refresh job list from job manager."""
        all_jobs = self.job_manager.get_all_jobs()
        for job_id, job_info in all_jobs.items():
            if job_id not in self.jobs:
                # Create Job object from job_info
                self.jobs[job_id] = Job(
                    id=job_id,
                    name=job_info['config']['job_name'],
                    status=JobStatus(job_info['status']),
                    created_at=datetime.fromisoformat(job_info['created_at']),
                )
                
    def create_job_table(self) -> Table:
        """Create job status table with selection highlight."""
        table = Table(
            title="📋 Training Jobs",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
        )
        
        table.add_column("", width=2)  # Selection indicator
        table.add_column("ID", style="dim", width=12)
        table.add_column("Name", style="cyan")
        table.add_column("Status", justify="center", width=12)
        table.add_column("Progress", justify="center", width=20)
        table.add_column("Duration", justify="right", width=12)
        
        for idx, (job_id, job) in enumerate(self.jobs.items()):
            # Selection indicator
            selected = "▶" if idx == self.selected_job_index else " "
            
            # Status with color
            status_style = {
                JobStatus.PENDING: "yellow",
                JobStatus.RUNNING: "green",
                JobStatus.COMPLETED: "blue",
                JobStatus.FAILED: "red",
                JobStatus.PAUSED: "yellow dim",
            }[job.status]
            
            status_text = Text(job.status.value.upper(), style=f"bold {status_style}")
            
            # Progress bar
            if job.status == JobStatus.RUNNING:
                progress_bar = f"[green]{'█' * int(job.progress / 5)}[/green][dim]{'░' * (20 - int(job.progress / 5))}[/dim] {job.progress:.1f}%"
            else:
                progress_bar = f"[dim]{job.progress:.1f}%[/dim]"
            
            # Duration
            if job.started_at:
                if job.completed_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                else:
                    duration = (datetime.now() - job.started_at).total_seconds()
                duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
            else:
                duration_str = "-"
            
            # Highlight selected row
            row_style = "bold" if idx == self.selected_job_index else None
            
            table.add_row(
                selected,
                job.id,
                job.name,
                status_text,
                progress_bar,
                duration_str,
                style=row_style
            )
            
        return table
        
    def create_help_panel(self) -> Panel:
        """Create help panel."""
        help_text = """
[bold cyan]Keyboard Controls:[/bold cyan]

[yellow]Navigation:[/yellow]
  ↑/↓     - Select job
  Enter   - View job details
  h       - Toggle this help
  r       - Refresh job list
  q       - Quit

[yellow]Job Control:[/yellow]
  n       - New job
  p       - Pause/Resume selected job
  s       - Stop selected job

[yellow]Input Mode:[/yellow]
  Escape  - Cancel input
  Enter   - Confirm input

[dim]Press 'h' to hide this help[/dim]
"""
        return Panel(
            help_text.strip(),
            title="❓ Help",
            box=box.ROUNDED,
            style="cyan",
        )
        
    def create_status_bar(self) -> Text:
        """Create status bar with messages."""
        if self.input_mode:
            return Text(self.status_message + self.input_buffer + "█", style="bold yellow")
        elif self.status_message and (time.time() - self.status_message_time) < 3:
            return Text(self.status_message, style="bold green")
        else:
            return Text(
                f"Jobs: {len(self.jobs)} | Running: {len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING])} | "
                f"Press 'h' for help | Last updated: {datetime.now().strftime('%H:%M:%S')}",
                style="dim"
            )
            
    def create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout(name="root")
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        
        if self.show_help:
            layout["body"].split_row(
                Layout(name="left", ratio=3),
                Layout(name="right", ratio=2),
            )
            
            layout["left"].split_column(
                Layout(name="jobs", ratio=2),
                Layout(name="metrics", ratio=1),
            )
            
            layout["right"].update(self.create_help_panel())
        else:
            layout["body"].split_row(
                Layout(name="left", ratio=3),
                Layout(name="right", ratio=2),
            )
            
            layout["left"].split_column(
                Layout(name="jobs", ratio=2),
                Layout(name="metrics", ratio=1),
            )
            
            layout["right"].split_column(
                Layout(name="chart"),
                Layout(name="controls", size=10),
            )
            
            active_job = self.jobs.get(self.active_job_id) if self.active_job_id else None
            layout["chart"].update(self.create_reward_chart(active_job))
            layout["controls"].update(self.create_controls_panel())
        
        # Common elements
        layout["header"].update(self.create_header())
        layout["jobs"].update(self.create_job_table())
        
        active_job = self.jobs.get(self.active_job_id) if self.active_job_id else None
        layout["metrics"].update(self.create_metrics_panel(active_job))
        
        # Status bar
        layout["footer"].update(Align.center(self.create_status_bar()))
        
        return layout
        
    async def update_jobs(self):
        """Update job metrics from job manager."""
        while self.running:
            for job_id, job in self.jobs.items():
                if job.status == JobStatus.RUNNING:
                    # Get messages from job manager
                    messages = self.job_manager.get_job_messages(job_id)
                    
                    for msg in messages:
                        if msg.type == "metrics":
                            # Update job metrics
                            data = msg.data
                            job.metrics.episode = data.get("episode", 0)
                            job.metrics.total_episodes = data.get("total_episodes", 500)
                            job.metrics.current_reward = data.get("reward", 0)
                            job.metrics.avg_reward = data.get("avg_reward", 0)
                            job.metrics.best_reward = data.get("best_reward", 0)
                            job.metrics.loss = data.get("loss", 0)
                            job.progress = (job.metrics.episode / job.metrics.total_episodes) * 100
                            
                        elif msg.type == "status":
                            # Update job status
                            status_map = {
                                "running": JobStatus.RUNNING,
                                "completed": JobStatus.COMPLETED,
                                "failed": JobStatus.FAILED,
                                "paused": JobStatus.PAUSED,
                            }
                            new_status = status_map.get(msg.data.get("status"))
                            if new_status:
                                job.status = new_status
                                if new_status == JobStatus.COMPLETED:
                                    job.completed_at = datetime.now()
                                    job.progress = 100.0
                                    
                        elif msg.type == "error":
                            job.status = JobStatus.FAILED
                            job.error = msg.data.get("error", "Unknown error")
                            
            # Cleanup completed jobs periodically
            self.job_manager.cleanup_completed_jobs()
            
            await asyncio.sleep(0.25)


@contextmanager
def raw_mode():
    """Context manager for raw terminal mode."""
    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)


def run_interactive_tui():
    """Run the interactive Rich TUI."""
    app = InteractiveCuMindTUI()
    
    # Keyboard input thread
    def keyboard_thread():
        with raw_mode():
            while app.running:
                key = sys.stdin.read(1)
                if key == '\x1b':  # Escape sequence
                    seq = key + sys.stdin.read(2)
                    app.handle_key(seq)
                else:
                    app.handle_key(key)
                    
    kb_thread = threading.Thread(target=keyboard_thread, daemon=True)
    kb_thread.start()
    
    # Run the main TUI
    try:
        with Live(
            app.create_layout(),
            refresh_per_second=app.refresh_rate,
            screen=True,
        ) as live:
            # Start async job updates
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            update_task = loop.create_task(app.update_jobs())
            
            # Main loop
            while app.running:
                try:
                    # Update display
                    live.update(app.create_layout())
                    time.sleep(1 / app.refresh_rate)
                    
                except KeyboardInterrupt:
                    app.running = False
                    
            update_task.cancel()
            
    except Exception as e:
        app.console.print(f"[red]Error: {e}[/red]")
        

if __name__ == "__main__":
    run_interactive_tui()