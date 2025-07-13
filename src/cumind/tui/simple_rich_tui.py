"""Simple Rich-based TUI for CuMind without terminal mode manipulation."""

import time
from datetime import datetime
from typing import Dict, Optional
import threading

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from .rich_app import Job, JobStatus, TrainingMetrics


class SimpleRichTUI:
    """Simple Rich TUI for job monitoring without keyboard interaction."""
    
    def __init__(self):
        self.console = Console()
        self.jobs: Dict[str, Job] = {}
        self.running = True
        
        # Create demo jobs
        self._create_demo_jobs()
        
    def _create_demo_jobs(self):
        """Create demonstration jobs."""
        # Running job
        job1 = Job(
            id="job_001",
            name="CartPole-v1 Training",
            status=JobStatus.RUNNING,
            created_at=datetime.now(),
            started_at=datetime.now(),
            progress=0.0
        )
        job1.metrics = TrainingMetrics(
            episode=0,
            total_episodes=500,
            current_reward=0,
            avg_reward=0,
            best_reward=0,
            steps_per_sec=0,
            training_time=0,
            memory_size=0
        )
        
        self.jobs[job1.id] = job1
        
    def create_header(self) -> Panel:
        """Create header panel."""
        header = Table(show_header=False, box=None)
        header.add_column(justify="center")
        header.add_row("[bold cyan]🧠 CuMind - Training Monitor[/bold cyan]")
        header.add_row("[dim]Real-time training job monitoring dashboard[/dim]")
        
        return Panel(header, style="bold white on blue", box=box.DOUBLE_EDGE)
        
    def create_job_list(self) -> Panel:
        """Create job list panel."""
        table = Table(
            title="Active Jobs",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Job ID", style="dim", width=12)
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Progress", justify="center", width=25)
        
        for job in self.jobs.values():
            # Status
            status_colors = {
                JobStatus.PENDING: "yellow",
                JobStatus.RUNNING: "green",
                JobStatus.COMPLETED: "blue",
                JobStatus.FAILED: "red",
                JobStatus.PAUSED: "yellow dim"
            }
            status = f"[{status_colors[job.status]}]{job.status.value.upper()}[/{status_colors[job.status]}]"
            
            # Progress bar
            progress_text = f"{job.progress:.1f}%"
            if job.status == JobStatus.RUNNING:
                filled = int(job.progress / 5)
                empty = 20 - filled
                progress_bar = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim] {progress_text}"
            else:
                progress_bar = f"[dim]{progress_text}[/dim]"
            
            table.add_row(job.id, job.name, status, progress_bar)
            
        return Panel(table, title="📋 Job Queue", box=box.ROUNDED)
        
    def create_metrics_display(self, job: Optional[Job]) -> Panel:
        """Create metrics display for selected job."""
        if not job or job.status != JobStatus.RUNNING:
            return Panel(
                "[dim italic]No active job selected[/dim italic]",
                title="📊 Metrics",
                box=box.ROUNDED
            )
            
        # Create progress indicators
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True
        )
        
        episode_task = progress.add_task(
            "Episodes",
            total=job.metrics.total_episodes,
            completed=job.metrics.episode
        )
        
        memory_task = progress.add_task(
            "Memory",
            total=job.metrics.memory_capacity,
            completed=job.metrics.memory_size
        )
        
        # Metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white", justify="right")
        
        metrics_table.add_row("Current Reward", f"{job.metrics.current_reward:.1f}")
        metrics_table.add_row("Average Reward", f"{job.metrics.avg_reward:.2f}")
        metrics_table.add_row("Best Reward", f"{job.metrics.best_reward:.0f}")
        metrics_table.add_row("Loss", f"{job.metrics.loss:.4f}")
        metrics_table.add_row("Learning Rate", f"{job.metrics.learning_rate:.4f}")
        metrics_table.add_row("Steps/sec", f"{job.metrics.steps_per_sec:.1f}")
        
        # Combine progress and metrics
        content = Columns([progress, metrics_table], padding=1)
        
        return Panel(
            content,
            title=f"📊 Training Metrics - {job.name}",
            box=box.ROUNDED
        )
        
    def create_chart(self, job: Optional[Job]) -> Panel:
        """Create reward history chart."""
        if not job or job.status == JobStatus.PENDING:
            return Panel(
                "[dim italic]No data available[/dim italic]",
                title="📈 Reward History",
                box=box.ROUNDED
            )
            
        # Simple sparkline-style chart
        if job.metrics.episode > 0:
            # Generate fake historical data for demo
            history_length = min(50, job.metrics.episode)
            max_height = 8
            
            # Create chart
            chart_lines = []
            
            # Y-axis labels
            max_reward = 300
            for i in range(max_height, -1, -1):
                value = int(max_reward * (i / max_height))
                chart_lines.append(f"{value:>4} │")
                
            # X-axis
            chart_lines.append("     └" + "─" * (history_length + 2))
            chart_lines.append(f"      0{' ' * (history_length - 5)}{job.metrics.episode}")
            
            # Add data points (simplified)
            for i in range(max_height + 1):
                line_idx = max_height - i
                progress_ratio = job.metrics.episode / job.metrics.total_episodes
                threshold = i / max_height
                
                # Simple visualization based on progress
                if progress_ratio > threshold * 0.8:
                    chart_lines[line_idx] += " ▄"
                    
            chart_text = "\n".join(chart_lines)
        else:
            chart_text = "[dim]Waiting for data...[/dim]"
            
        return Panel(
            chart_text,
            title="📈 Reward History",
            box=box.ROUNDED,
            style="green"
        )
        
    def create_info_panel(self) -> Panel:
        """Create information panel."""
        info_text = """
[yellow]Training Information:[/yellow]

• Environment: CartPole-v1
• Algorithm: MuZero (MCTS + Neural Network)
• Device: CPU/GPU (auto-detected)

[yellow]Features:[/yellow]
• Real-time metrics visualization
• Progress tracking
• Reward history chart
• Multi-job support

[dim]Press Ctrl+C to exit[/dim]
"""
        return Panel(
            info_text.strip(),
            title="ℹ️  Information",
            box=box.ROUNDED
        )
        
    def create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout(name="root")
        
        # Split into header and body
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=1)
        )
        
        # Split body into left and right
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )
        
        # Split left into jobs and metrics
        layout["left"].split_column(
            Layout(name="jobs", ratio=1),
            Layout(name="metrics", ratio=2)
        )
        
        # Split right into chart and info
        layout["right"].split_column(
            Layout(name="chart", ratio=2),
            Layout(name="info", ratio=1)
        )
        
        # Update content
        layout["header"].update(self.create_header())
        layout["jobs"].update(self.create_job_list())
        
        # Get first running job for display
        active_job = None
        for job in self.jobs.values():
            if job.status == JobStatus.RUNNING:
                active_job = job
                break
                
        layout["metrics"].update(self.create_metrics_display(active_job))
        layout["chart"].update(self.create_chart(active_job))
        layout["info"].update(self.create_info_panel())
        
        # Footer
        footer_text = Text(
            f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Ctrl+C to exit",
            style="dim",
            justify="center"
        )
        layout["footer"].update(Align.center(footer_text))
        
        return layout
        
    def update_simulation(self):
        """Simulate training progress updates."""
        while self.running:
            for job in self.jobs.values():
                if job.status == JobStatus.RUNNING and job.progress < 100:
                    # Update progress
                    job.progress += 0.5
                    job.metrics.episode = int(job.progress * 5)
                    
                    # Update metrics with realistic values
                    job.metrics.current_reward = 100 + (job.progress * 1.5)
                    job.metrics.avg_reward = 90 + (job.progress * 1.2)
                    job.metrics.best_reward = 150 + (job.progress * 1.0)
                    job.metrics.loss = 0.1 * (1 - job.progress / 100)
                    job.metrics.steps_per_sec = 30 + (job.progress * 0.2)
                    job.metrics.training_time += 0.25
                    job.metrics.memory_size = min(1000, int(job.metrics.episode * 4))
                    
                    # Complete job
                    if job.progress >= 100:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        
            time.sleep(0.25)
            
    def run(self):
        """Run the TUI."""
        # Start simulation thread
        sim_thread = threading.Thread(target=self.update_simulation, daemon=True)
        sim_thread.start()
        
        try:
            with Live(
                self.create_layout(),
                refresh_per_second=4,
                screen=True
            ) as live:
                while self.running:
                    live.update(self.create_layout())
                    time.sleep(0.25)
                    
        except KeyboardInterrupt:
            self.running = False
            self.console.print("\n[yellow]Training monitor stopped.[/yellow]")
            

def run_simple_rich_tui():
    """Entry point for simple Rich TUI."""
    tui = SimpleRichTUI()
    tui.run()
    

if __name__ == "__main__":
    run_simple_rich_tui()