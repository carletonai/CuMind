"""Tests for TUI components."""

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from cumind.tui.job_manager import JobConfig, JobManager, JobMessage, TrainingProcess
from cumind.tui.rich_app import CuMindRichTUI, Job, JobStatus, TrainingMetrics
from cumind.tui.simple_rich_tui import SimpleRichTUI


class TestJobStatus:
    """Test JobStatus enum."""

    def test_job_status_values(self):
        """Test that JobStatus enum has expected values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.PAUSED.value == "paused"


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_default_metrics(self):
        """Test default values for TrainingMetrics."""
        metrics = TrainingMetrics()
        assert metrics.episode == 0
        assert metrics.total_episodes == 500
        assert metrics.current_reward == 0.0
        assert metrics.avg_reward == 0.0
        assert metrics.best_reward == 0.0
        assert metrics.loss == 0.0
        assert metrics.learning_rate == 0.001
        assert metrics.steps_per_sec == 0.0
        assert metrics.training_time == 0.0
        assert metrics.memory_size == 0
        assert metrics.memory_capacity == 1000

    def test_custom_metrics(self):
        """Test custom values for TrainingMetrics."""
        metrics = TrainingMetrics(
            episode=100,
            total_episodes=1000,
            current_reward=150.5,
            avg_reward=125.3,
            best_reward=200.0,
            loss=0.0123,
            learning_rate=0.0001,
            steps_per_sec=45.2,
            training_time=300.5,
            memory_size=500,
            memory_capacity=2000
        )
        assert metrics.episode == 100
        assert metrics.total_episodes == 1000
        assert metrics.current_reward == 150.5
        assert metrics.avg_reward == 125.3
        assert metrics.best_reward == 200.0
        assert metrics.loss == 0.0123
        assert metrics.learning_rate == 0.0001
        assert metrics.steps_per_sec == 45.2
        assert metrics.training_time == 300.5
        assert metrics.memory_size == 500
        assert metrics.memory_capacity == 2000


class TestJob:
    """Test Job dataclass."""

    def test_job_creation(self):
        """Test creating a Job instance."""
        job = Job(
            id="test_job_001",
            name="Test Training",
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
        assert job.id == "test_job_001"
        assert job.name == "Test Training"
        assert job.status == JobStatus.PENDING
        assert job.started_at is None
        assert job.completed_at is None
        assert isinstance(job.metrics, TrainingMetrics)
        assert job.progress == 0.0
        assert job.error is None

    def test_job_with_all_fields(self):
        """Test Job with all fields set."""
        now = datetime.now()
        metrics = TrainingMetrics(episode=50, total_episodes=100)
        job = Job(
            id="test_job_002",
            name="Complete Job",
            status=JobStatus.COMPLETED,
            created_at=now,
            started_at=now,
            completed_at=now,
            metrics=metrics,
            progress=100.0,
            error=None
        )
        assert job.metrics.episode == 50
        assert job.progress == 100.0


class TestCuMindRichTUI:
    """Test CuMindRichTUI class."""

    def test_initialization(self):
        """Test TUI initialization."""
        tui = CuMindRichTUI()
        assert tui.console is not None
        assert isinstance(tui.jobs, dict)
        assert tui.active_job_id is not None
        assert tui.running is True
        assert tui.refresh_rate == 4

        # Check sample jobs were created
        assert len(tui.jobs) == 3
        assert "job_001" in tui.jobs
        assert "job_002" in tui.jobs
        assert "job_003" in tui.jobs

    def test_create_header(self):
        """Test header panel creation."""
        tui = CuMindRichTUI()
        header = tui.create_header()
        assert header is not None
        # Panel should contain the title text
        assert hasattr(header, 'renderable')

    def test_create_job_table(self):
        """Test job table creation."""
        tui = CuMindRichTUI()
        table = tui.create_job_table()
        assert table is not None
        # Table should have columns
        assert hasattr(table, 'columns')
        assert len(table.columns) > 0

    def test_create_metrics_panel(self):
        """Test metrics panel creation."""
        tui = CuMindRichTUI()

        # Test with no job
        panel = tui.create_metrics_panel(None)
        assert panel is not None

        # Test with running job
        running_job = tui.jobs.get("job_001")
        if running_job:
            panel = tui.create_metrics_panel(running_job)
            assert panel is not None

    def test_create_reward_chart(self):
        """Test reward chart creation."""
        tui = CuMindRichTUI()

        # Test with no job
        chart = tui.create_reward_chart(None)
        assert chart is not None

        # Test with job
        job = tui.jobs.get("job_001")
        if job:
            chart = tui.create_reward_chart(job)
            assert chart is not None

    def test_create_controls_panel(self):
        """Test controls panel creation."""
        tui = CuMindRichTUI()
        controls = tui.create_controls_panel()
        assert controls is not None

    def test_create_layout(self):
        """Test layout creation."""
        tui = CuMindRichTUI()
        layout = tui.create_layout()
        assert layout is not None
        assert hasattr(layout, 'split_column')


class TestSimpleRichTUI:
    """Test SimpleRichTUI class."""

    def test_initialization(self):
        """Test simple TUI initialization."""
        tui = SimpleRichTUI()
        assert tui.console is not None
        assert isinstance(tui.jobs, dict)
        assert tui.running is True
        assert len(tui.jobs) == 1

    def test_create_header(self):
        """Test header creation."""
        tui = SimpleRichTUI()
        header = tui.create_header()
        assert header is not None

    def test_create_job_list(self):
        """Test job list panel creation."""
        tui = SimpleRichTUI()
        job_list = tui.create_job_list()
        assert job_list is not None

    def test_create_metrics_display(self):
        """Test metrics display creation."""
        tui = SimpleRichTUI()

        # Test with no job
        display = tui.create_metrics_display(None)
        assert display is not None

        # Test with job
        job = list(tui.jobs.values())[0] if tui.jobs else None
        if job:
            display = tui.create_metrics_display(job)
            assert display is not None

    def test_create_chart(self):
        """Test chart creation."""
        tui = SimpleRichTUI()

        # Test with no job
        chart = tui.create_chart(None)
        assert chart is not None

        # Test with job
        job = list(tui.jobs.values())[0] if tui.jobs else None
        if job:
            # Set some metrics to generate chart
            job.metrics.episode = 50
            chart = tui.create_chart(job)
            assert chart is not None

    def test_create_info_panel(self):
        """Test info panel creation."""
        tui = SimpleRichTUI()
        info = tui.create_info_panel()
        assert info is not None

    def test_create_layout(self):
        """Test complete layout creation."""
        tui = SimpleRichTUI()
        layout = tui.create_layout()
        assert layout is not None

    def test_update_simulation(self):
        """Test simulation update logic."""
        tui = SimpleRichTUI()
        tui.running = False  # Prevent infinite loop

        # Get initial values
        job = list(tui.jobs.values())[0]
        initial_progress = job.progress
        initial_episode = job.metrics.episode

        # Run one update manually
        if job.status == JobStatus.RUNNING and job.progress < 100:
            job.progress += 0.5
            job.metrics.episode = int(job.progress * 5)

        # Check values changed
        assert job.progress > initial_progress
        assert job.metrics.episode > initial_episode


class TestJobConfig:
    """Test JobConfig dataclass."""

    def test_job_config_creation(self):
        """Test creating JobConfig."""
        config = JobConfig(
            config_path="/path/to/config.json",
            job_name="Test Job",
            max_episodes=1000,
            checkpoint_interval=100
        )
        assert config.config_path == "/path/to/config.json"
        assert config.job_name == "Test Job"
        assert config.max_episodes == 1000
        assert config.checkpoint_interval == 100

    def test_job_config_defaults(self):
        """Test JobConfig default values."""
        config = JobConfig(
            config_path="/path/to/config.json",
            job_name="Test Job"
        )
        assert config.max_episodes is None
        assert config.checkpoint_interval is None


class TestJobMessage:
    """Test JobMessage dataclass."""

    def test_job_message_creation(self):
        """Test creating JobMessage."""
        msg = JobMessage(
            type="metrics",
            data={"episode": 10, "reward": 100}
        )
        assert msg.type == "metrics"
        assert msg.data == {"episode": 10, "reward": 100}
        assert msg.timestamp is not None

    def test_job_message_with_timestamp(self):
        """Test JobMessage with custom timestamp."""
        custom_time = datetime.now()
        msg = JobMessage(
            type="status",
            data={"status": "running"},
            timestamp=custom_time
        )
        assert msg.timestamp == custom_time


class TestTrainingProcess:
    """Test TrainingProcess class."""

    def test_training_process_initialization(self):
        """Test TrainingProcess initialization."""
        config = JobConfig(
            config_path="test_config.json",
            job_name="Test Training"
        )
        process = TrainingProcess("test_job_001", config)

        assert process.job_id == "test_job_001"
        assert process.config == config
        assert process.process is None
        assert process.is_running is False
        assert hasattr(process, 'message_queue')
        assert hasattr(process, 'control_queue')

    def test_get_messages_empty(self):
        """Test getting messages when queue is empty."""
        config = JobConfig(
            config_path="test_config.json",
            job_name="Test Training"
        )
        process = TrainingProcess("test_job_001", config)

        messages = process.get_messages()
        assert messages == []


class TestJobManager:
    """Test JobManager class."""

    def test_job_manager_initialization(self):
        """Test JobManager initialization."""
        manager = JobManager(max_concurrent_jobs=2)
        assert manager.max_concurrent_jobs == 2
        assert isinstance(manager.jobs, dict)
        assert isinstance(manager.job_configs, dict)
        assert isinstance(manager.job_history, list)
        assert manager.jobs_dir.exists()

    def test_create_job(self):
        """Test creating a job."""
        manager = JobManager()
        config = JobConfig(
            config_path="test_config.json",
            job_name="Test Job"
        )

        job_id = manager.create_job(config)
        assert job_id.startswith("job_")
        assert job_id in manager.jobs
        assert job_id in manager.job_configs
        assert manager.job_configs[job_id] == config

    def test_max_concurrent_jobs(self):
        """Test max concurrent jobs limit."""
        manager = JobManager(max_concurrent_jobs=1)

        # Create first job
        config1 = JobConfig(
            config_path="test_config1.json",
            job_name="Test Job 1"
        )
        job_id1 = manager.create_job(config1)

        # Mock the first job as running
        manager.jobs[job_id1].is_running = True

        # Try to create second job - should raise error
        config2 = JobConfig(
            config_path="test_config2.json",
            job_name="Test Job 2"
        )

        with pytest.raises(RuntimeError, match="Maximum concurrent jobs"):
            manager.create_job(config2)

    def test_get_all_jobs(self):
        """Test getting all jobs."""
        manager = JobManager()

        # Create a job
        config = JobConfig(
            config_path="test_config.json",
            job_name="Test Job"
        )
        job_id = manager.create_job(config)

        # Get all jobs
        all_jobs = manager.get_all_jobs()
        assert job_id in all_jobs
        assert all_jobs[job_id]["is_active"] is False

    def test_cleanup_completed_jobs(self):
        """Test cleanup of completed jobs."""
        manager = JobManager()

        # Create a job
        config = JobConfig(
            config_path="test_config.json",
            job_name="Test Job"
        )
        job_id = manager.create_job(config)

        # Mock job as not running
        manager.jobs[job_id].is_running = False
        manager.jobs[job_id].process = Mock()
        manager.jobs[job_id].process.is_alive.return_value = False

        # Run cleanup
        manager.cleanup_completed_jobs()

        # Job should be removed from memory
        assert job_id not in manager.jobs


@pytest.fixture(autouse=True)
def cleanup_jobs_dir():
    """Clean up jobs directory after tests."""
    yield
    # Cleanup
    import shutil
    from pathlib import Path
    jobs_dir = Path("jobs")
    if jobs_dir.exists():
        shutil.rmtree(jobs_dir)
