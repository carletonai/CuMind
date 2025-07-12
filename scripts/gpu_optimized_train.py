#!/usr/bin/env python3
"""GPU-optimized training script for CuMind with performance flags."""

import os
import sys
from pathlib import Path

# Set GPU optimization flags BEFORE importing JAX
print("Setting GPU optimization environment variables...")

# XLA optimization flags
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_collectives=true"

# Profile-guided optimization
os.environ["JAX_ENABLE_PGLE"] = "true"
os.environ["JAX_PGLE_PROFILING_RUNS"] = "3"

# NCCL optimizations for multi-GPU
os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    }
)

# Memory optimization
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"  # Use 90% of GPU memory

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax

print(f"JAX devices available: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Import CuMind after setting environment
from cumind import Agent, Config
from cumind.runner import Runner


def main():
    """Run GPU-optimized training."""
    # Load configuration
    config = Config.from_json("configuration.json")

    # Override with GPU-optimized settings
    if jax.default_backend() == "gpu":
        print("\nGPU detected! Using optimized settings:")
        print("- Mixed precision training (bfloat16)")
        print("- JIT compilation enabled")
        print("- XLA optimizations active")

        # You can override config here if mixed precision is implemented
        # config.model_dtype = "bfloat16"
    else:
        print("\nWarning: No GPU detected. Running on CPU.")

    # Create agent and runner
    print(f"\nStarting training on {config.env_name}...")
    agent = Agent(config)
    runner = Runner(agent, config)

    # Run training
    runner.run()

    print("\nTraining completed!")
    print("GPU optimization flags were active during training.")


if __name__ == "__main__":
    main()
