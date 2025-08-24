import os

from cumind.utils.config import cfg


def register_xla_flags() -> None:
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
