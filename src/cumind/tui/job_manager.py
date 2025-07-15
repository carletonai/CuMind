"""Job management system for CuMind training jobs."""

import asyncio
import json
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4

from ..config import Config
from ..runner import train


@dataclass
class JobConfig:
    """Configuration for a training job."""
    config_path: str
    job_name: str
    max_episodes: Optional[int] = None
    checkpoint_interval: Optional[int] = None
    

@dataclass
class JobMessage:
    """Message from training process."""
    type: str  # 'metrics', 'status', 'error', 'log'
    data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TrainingProcess:
    """Wrapper for training process with communication."""
    
    def __init__(self, job_id: str, config: JobConfig):
        self.job_id = job_id
        self.config = config
        self.process: Optional[mp.Process] = None
        self.message_queue: mp.Queue = mp.Queue()
        self.control_queue: mp.Queue = mp.Queue()
        self.is_running = False
        
    def start(self):
        """Start the training process."""
        if self.process and self.process.is_alive():
            return
            
        self.process = mp.Process(
            target=self._run_training,
            args=(self.job_id, self.config, self.message_queue, self.control_queue)
        )
        self.process.start()
        self.is_running = True
        
    def stop(self):
        """Stop the training process."""
        if self.process and self.process.is_alive():
            self.control_queue.put({"command": "stop"})
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
        self.is_running = False
        
    def pause(self):
        """Pause the training process."""
        if self.process and self.process.is_alive():
            self.control_queue.put({"command": "pause"})
            
    def resume(self):
        """Resume the training process."""
        if self.process and self.process.is_alive():
            self.control_queue.put({"command": "resume"})
            
    def get_messages(self) -> List[JobMessage]:
        """Get all pending messages from the process."""
        messages = []
        while not self.message_queue.empty():
            try:
                msg = self.message_queue.get_nowait()
                messages.append(JobMessage(**msg))
            except queue.Empty:
                break
        return messages
        
    @staticmethod
    def _run_training(job_id: str, config: JobConfig, msg_queue: mp.Queue, ctrl_queue: mp.Queue):
        """Run training in separate process."""
        try:
            # Send initial status
            msg_queue.put({
                "type": "status",
                "data": {"status": "starting", "job_id": job_id}
            })
            
            # Load configuration
            cfg = Config.from_json(config.config_path)
            if config.max_episodes:
                cfg.num_episodes = config.max_episodes
            
            # Training state
            is_paused = False
            should_stop = False
            
            # Callback to send metrics
            def on_episode_end(episode: int, metrics: Dict[str, Any]):
                """Called at the end of each episode."""
                # Check control messages
                nonlocal is_paused, should_stop
                
                while not ctrl_queue.empty():
                    try:
                        cmd = ctrl_queue.get_nowait()
                        if cmd["command"] == "stop":
                            should_stop = True
                        elif cmd["command"] == "pause":
                            is_paused = True
                        elif cmd["command"] == "resume":
                            is_paused = False
                    except queue.Empty:
                        break
                        
                # Wait if paused
                while is_paused and not should_stop:
                    time.sleep(0.1)
                    # Check for resume/stop
                    while not ctrl_queue.empty():
                        try:
                            cmd = ctrl_queue.get_nowait()
                            if cmd["command"] == "stop":
                                should_stop = True
                            elif cmd["command"] == "resume":
                                is_paused = False
                        except queue.Empty:
                            break
                            
                if should_stop:
                    return False  # Stop training
                    
                # Send metrics
                msg_queue.put({
                    "type": "metrics",
                    "data": {
                        "episode": episode,
                        "total_episodes": cfg.num_episodes,
                        **metrics
                    }
                })
                
                return True  # Continue training
                
            # Run training with callback
            msg_queue.put({
                "type": "status",
                "data": {"status": "running", "job_id": job_id}
            })
            
            # Note: This is a placeholder - actual integration would require
            # modifying the train function to support callbacks
            train(cfg)  # This would need to be modified to support callbacks
            
            msg_queue.put({
                "type": "status",
                "data": {"status": "completed", "job_id": job_id}
            })
            
        except Exception as e:
            msg_queue.put({
                "type": "error",
                "data": {"error": str(e), "job_id": job_id}
            })
            msg_queue.put({
                "type": "status",
                "data": {"status": "failed", "job_id": job_id}
            })


class JobManager:
    """Manages multiple training jobs."""
    
    def __init__(self, max_concurrent_jobs: int = 4):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, TrainingProcess] = {}
        self.job_configs: Dict[str, JobConfig] = {}
        self.job_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Create jobs directory
        self.jobs_dir = Path("jobs")
        self.jobs_dir.mkdir(exist_ok=True)
        
    def create_job(self, config: JobConfig) -> str:
        """Create a new training job."""
        job_id = f"job_{uuid4().hex[:8]}"
        
        with self._lock:
            if len([j for j in self.jobs.values() if j.is_running]) >= self.max_concurrent_jobs:
                raise RuntimeError(f"Maximum concurrent jobs ({self.max_concurrent_jobs}) reached")
                
            process = TrainingProcess(job_id, config)
            self.jobs[job_id] = process
            self.job_configs[job_id] = config
            
            # Save job info
            job_info = {
                "id": job_id,
                "config": asdict(config),
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
            self._save_job_info(job_id, job_info)
            
        return job_id
        
    def start_job(self, job_id: str):
        """Start a training job."""
        with self._lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
                
            self.jobs[job_id].start()
            self._update_job_status(job_id, "running")
            
    def stop_job(self, job_id: str):
        """Stop a training job."""
        with self._lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
                
            self.jobs[job_id].stop()
            self._update_job_status(job_id, "stopped")
            
    def pause_job(self, job_id: str):
        """Pause a training job."""
        with self._lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
                
            self.jobs[job_id].pause()
            self._update_job_status(job_id, "paused")
            
    def resume_job(self, job_id: str):
        """Resume a training job."""
        with self._lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
                
            self.jobs[job_id].resume()
            self._update_job_status(job_id, "running")
            
    def get_job_messages(self, job_id: str) -> List[JobMessage]:
        """Get messages from a specific job."""
        if job_id not in self.jobs:
            return []
            
        return self.jobs[job_id].get_messages()
        
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all jobs."""
        with self._lock:
            all_jobs = {}
            
            # Active jobs
            for job_id, process in self.jobs.items():
                job_info = self._load_job_info(job_id)
                job_info["is_active"] = process.is_running
                all_jobs[job_id] = job_info
                
            # Historical jobs
            for job_file in self.jobs_dir.glob("job_*.json"):
                job_id = job_file.stem
                if job_id not in all_jobs:
                    job_info = self._load_job_info(job_id)
                    job_info["is_active"] = False
                    all_jobs[job_id] = job_info
                    
        return all_jobs
        
    def cleanup_completed_jobs(self):
        """Remove completed jobs from memory."""
        with self._lock:
            completed_jobs = []
            for job_id, process in self.jobs.items():
                if not process.is_running and process.process and not process.process.is_alive():
                    completed_jobs.append(job_id)
                    
            for job_id in completed_jobs:
                del self.jobs[job_id]
                
    def _save_job_info(self, job_id: str, info: Dict[str, Any]):
        """Save job information to disk."""
        job_file = self.jobs_dir / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(info, f, indent=2)
            
    def _load_job_info(self, job_id: str) -> Dict[str, Any]:
        """Load job information from disk."""
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            with open(job_file, "r") as f:
                return json.load(f)
        return {}
        
    def _update_job_status(self, job_id: str, status: str):
        """Update job status on disk."""
        job_info = self._load_job_info(job_id)
        job_info["status"] = status
        job_info["updated_at"] = datetime.now().isoformat()
        self._save_job_info(job_id, job_info)