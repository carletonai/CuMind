import sys
from pathlib import Path

import jax.profiler

from cumind.utils.config import cfg


def start_trace(workspace_path: Path) -> None:
    if not cfg.tracing.enabled:
        return
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = cfg.tracing.host_level
    options.python_tracer_level = cfg.tracing.python_level
    if cfg.device == "tpu":
        options.advanced_configuration = {
            "tpu_trace_mode": cfg.tracing.tpu_trace_mode,
            "tpu_num_sparse_cores_to_trace": cfg.tracing.tpu_num_sparse_cores_to_trace,
            "tpu_num_sparse_core_tiles_to_trace": cfg.tracing.tpu_num_sparse_core_tiles_to_trace,
            "tpu_num_chips_to_profile_per_task": cfg.tracing.tpu_num_chips_to_profile_per_task,
        }
    path = workspace_path / "trace"
    jax.profiler.start_trace(log_dir=str(path), create_perfetto_trace=True, profiler_options=options)


def stop_trace(workspace_path: Path) -> None:
    if cfg.tracing.enabled:
        jax.profiler.stop_trace()  #  type: ignore
        if cfg.tracing.save_memory:
            path = workspace_path / "memory.pprof"
            jax.profiler.save_device_memory_profile(str(path), cfg.device)
    sys.exit(0)
