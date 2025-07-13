"""Rich-based TUI for CuMind job monitoring and management."""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskID
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class TrainingMetrics:
    """Training metrics data."""
    episode: int = 0
    total_episodes: int = 500
    current_reward: float = 0.0
    avg_reward: float = 0.0
    best_reward: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.001
    steps_per_sec: float = 0.0
    training_time: float = 0.0
    memory_size: int = 0
    memory_capacity: int = 1000


@dataclass
class Job:
    """Represents a training job."""
    id: str
    name: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    progress: float = 0.0
    error: Optional[str] = None


class CuMindRichTUI:
    """Rich-based TUI for CuMind."""
    
    def __init__(self):
        self.console = Console()
        self.jobs: Dict[str, Job] = {}
        self.active_job_id: Optional[str] = None
        self.running = True
        self.refresh_rate = 4  # Updates per second
        
        # Sample jobs for demonstration
        self._create_sample_jobs()
        
    def _create_sample_jobs(self):
        """Create sample jobs for demonstration."""
        # Active training job
        job1 = Job(
            id="job_001",
            name="CartPole-v1 Training",
            status=JobStatus.RUNNING,
            created_at=datetime.now(),
            started_at=datetime.now(),
            progress=46.8
        )
        job1.metrics = TrainingMetrics(
            episode=234,
            total_episodes=500,
            current_reward=196,
            avg_reward=187.3,
            best_reward=250,
            loss=0.0234,
            learning_rate=0.001,
            steps_per_sec=45.2,
            training_time=754,  # seconds
            memory_size=847,
            memory_capacity=1000
        )
        
        # Completed job
        job2 = Job(
            id="job_002",
            name="MountainCar-v0 Training",
            status=JobStatus.COMPLETED,
            created_at=datetime.now(),
            started_at=datetime.now(),
            completed_at=datetime.now(),
            progress=100.0
        )
        job2.metrics = TrainingMetrics(
            episode=1000,
            total_episodes=1000,
            avg_reward=-125.4,
            best_reward=-89,
        )
        
        # Failed job
        job3 = Job(
            id="job_003",
            name="Atari Pong Training",
            status=JobStatus.FAILED,
            created_at=datetime.now(),
            started_at=datetime.now(),
            progress=12.0,
            error="CUDA out of memory"
        )
        
        self.jobs = {
            job1.id: job1,
            job2.id: job2,
            job3.id: job3,
        }
        self.active_job_id = "job_001"
        
    def create_header(self) -> Panel:
        """Create header panel."""
        header_text = Text()
        header_text.append("CuMind ", style="bold cyan")
        header_text.append("- JAX-based Reinforcement Learning\n", style="bright_white")
        header_text.append("Monte Carlo Tree Search + Neural Networks", style="dim")
        
        return Panel(
            Align.center(header_text),
            style="bold white on blue",
            box=box.DOUBLE_EDGE,
        )
        
    def create_job_table(self) -> Table:
        """Create job status table."""
        table = Table(
            title="Training Jobs",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
        )
        
        table.add_column("ID", style="dim", width=12)
        table.add_column("Name", style="cyan")
        table.add_column("Status", justify="center", width=12)
        table.add_column("Progress", justify="center", width=20)
        table.add_column("Duration", justify="right", width=12)
        
        for job in self.jobs.values():
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
            
            table.add_row(
                job.id,
                job.name,
                status_text,
                progress_bar,
                duration_str,
            )
            
        return table
        
    def create_metrics_panel(self, job: Job) -> Panel:
        """Create metrics panel for active job."""
        if not job or job.status != JobStatus.RUNNING:
            return Panel(
                "[dim italic]No active job[/dim italic]",
                title="Training Metrics",
                box=box.ROUNDED,
            )
            
        metrics = job.metrics
        
        # Create two columns of metrics
        left_column = Table(show_header=False, box=None, padding=0)
        left_column.add_column("Metric", style="cyan")
        left_column.add_column("Value", style="white")
        
        left_column.add_row("Episode", f"{metrics.episode}/{metrics.total_episodes}")
        left_column.add_row("Avg Reward", f"{metrics.avg_reward:.2f}")
        left_column.add_row("Best Reward", f"{metrics.best_reward:.0f}")
        left_column.add_row("Loss", f"{metrics.loss:.4f}")
        
        right_column = Table(show_header=False, box=None, padding=0)
        right_column.add_column("Metric", style="cyan")
        right_column.add_column("Value", style="white")
        
        right_column.add_row("Current Reward", f"{metrics.current_reward:.0f}")
        right_column.add_row("Learning Rate", f"{metrics.learning_rate:.4f}")
        right_column.add_row("Steps/sec", f"{metrics.steps_per_sec:.1f}")
        right_column.add_row("Memory", f"{metrics.memory_size}/{metrics.memory_capacity}")
        
        columns = Columns([left_column, right_column], padding=2)
        
        return Panel(
            columns,
            title=f"Training Metrics - {job.name}",
            box=box.ROUNDED,
        )
        
    def create_reward_chart(self, job: Job) -> Panel:
        """Create a simple ASCII reward chart."""
        if not job or job.status == JobStatus.PENDING:
            return Panel(
                "[dim italic]No data available[/dim italic]",
                title="Reward History",
                box=box.ROUNDED,
            )
            
        # Simple ASCII chart
        chart_lines = []
        chart_lines.append("    250 ┤                           ╭─")
        chart_lines.append("    200 ┤                      ╭────╯")
        chart_lines.append("    150 ┤                 ╭────╯")
        chart_lines.append("    100 ┤            ╭────╯")
        chart_lines.append("     50 ┤       ╭────╯")
        chart_lines.append("      0 └────────────────────────────────────")
        chart_lines.append(f"        0    50   100  150  200  {job.metrics.episode}")
        
        chart_text = "\n".join(chart_lines)
        
        return Panel(
            chart_text,
            title="Reward History",
            box=box.ROUNDED,
            style="green",
        )
        
    def create_controls_panel(self) -> Panel:
        """Create controls panel."""
        controls = Table(show_header=False, box=None)
        controls.add_column("Key", style="bold yellow", width=12)
        controls.add_column("Action", style="white")
        
        controls.add_row("[p]", "Pause/Resume")
        controls.add_row("[s]", "Stop Job")
        controls.add_row("[n]", "New Job")
        controls.add_row("[↑/↓]", "Select Job")
        controls.add_row("[Enter]", "View Details")
        controls.add_row("[q]", "Quit")
        
        return Panel(
            controls,
            title="⌨️  Controls",
            box=box.ROUNDED,
        )
        
    def create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout(name="root")
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        
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
        
        # Populate layout
        layout["header"].update(self.create_header())
        layout["jobs"].update(self.create_job_table())
        
        active_job = self.jobs.get(self.active_job_id) if self.active_job_id else None
        layout["metrics"].update(self.create_metrics_panel(active_job))
        layout["chart"].update(self.create_reward_chart(active_job))
        layout["controls"].update(self.create_controls_panel())
        
        # Footer
        footer_text = Text(
            f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Press 'q' to quit",
            style="dim",
        )
        layout["footer"].update(Align.center(footer_text))
        
        return layout
        
    async def update_metrics(self):
        """Simulate metrics updates for running jobs."""
        while self.running:
            for job in self.jobs.values():
                if job.status == JobStatus.RUNNING:
                    # Update progress
                    job.progress = min(100.0, job.progress + 0.1)
                    job.metrics.episode = int(job.progress * 5)
                    
                    # Update metrics with some randomness
                    job.metrics.current_reward = 150 + (50 * (job.progress / 100))
                    job.metrics.avg_reward = 180 + (20 * (job.progress / 100))
                    job.metrics.steps_per_sec = 40 + (10 * (job.progress / 100))
                    job.metrics.training_time += 0.25
                    
                    # Complete job
                    if job.progress >= 100.0:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        
            await asyncio.sleep(0.25)
            
    def run(self):
        """Run the TUI."""
        try:
            with Live(
                self.create_layout(),
                refresh_per_second=self.refresh_rate,
                screen=True,
            ) as live:
                # Start async metrics update
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                update_task = loop.create_task(self.update_metrics())
                
                # Main loop
                while self.running:
                    try:
                        # Update display
                        live.update(self.create_layout())
                        time.sleep(1 / self.refresh_rate)
                        
                    except KeyboardInterrupt:
                        self.running = False
                        
                update_task.cancel()
                
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            

def run_rich_tui():
    """Entry point for Rich TUI."""
    app = CuMindRichTUI()
    app.run()


if __name__ == "__main__":
    run_rich_tui()