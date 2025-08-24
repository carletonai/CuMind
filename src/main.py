"""Entry for CuMind."""

import os

import jax.profiler

from cumind.agent.runner import inference, train
from cumind.utils.checkpoint import find_latest_checkpoint_for_env
from cumind.utils.config import cfg
from cumind.utils.logger import log


def main() -> None:
    """Main function for running training example."""

    train()

    ckpt = find_latest_checkpoint_for_env(cfg.env.name, cfg.workspace)
    if ckpt is None:
        raise RuntimeError("No checkpoint found for environment.")
    inference(ckpt, 500)


def flags() -> None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    if cfg.device == "cpu":
        os.environ["XLA_FLAGS"] = "--xla_cpu_use_xla_runtime=true --xla_cpu_enable_fast_math=true"
        return
    if cfg.device == "gpu":
        os.environ["XLA_FLAGS"] = "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_triton_gemm_any=true --xla_gpu_cuda_graph_enable=false --xla_gpu_memory_limit_slop_factor=90 --xla_gpu_per_fusion_autotune_cache_dir=/tmp/xla_autotune_cache"
        return

    if cfg.device == "tpu":
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8 --xla_tpu_enable_fast_relay_computation=true --xla_tpu_enable_async_collectives=true --xla_tpu_enable_dynamic_shape_support=true --xla_tpu_saturate_infeed=true"
        return


if __name__ == "__main__":
    workspace, trace_options = cfg.load("test.json")
    print(f"Workspace directory: {workspace}")

    flags()
    jax.profiler.start_trace(log_dir=f"{workspace}/trace", create_perfetto_trace=True, profiler_options=trace_options)
    try:
        main()
    except Exception as e:
        log.exception(f"Exception occurred: {e}")
    finally:
        jax.profiler.stop_trace()  # type: ignore
        if cfg.tracing.save_memory:
            jax.profiler.save_device_memory_profile(f"{workspace}/memory.pprof", cfg.device)
        log.shutdown()

    # Viewing Perfetto locally (for dev):
    # After program runs, follow the link printed in the terminal to view trace in your browser.
    # Or manually upload /tmp/profile-data to https://ui.perfetto.dev

    # Viewing Perfetto remotely (TPU, SSH):
    # If running on a remote server/TPU, set up an SSH tunnel before running main.py:
    # ssh -L 9001:127.0.0.1:9001 <your-user>@<remote-host>
    # Then, open the provided Perfetto link in your local browser.
    #
    # For Google Cloud TPU:
    # gcloud compute ssh <tpu-instance-name> -- -L 9001:127.0.0.1:9001
