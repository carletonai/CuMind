# GPU Performance Optimization Guide for CuMind

This document provides comprehensive GPU performance optimization tips for the CuMind project, based on JAX best practices and analysis of the current codebase.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Environment Configuration](#environment-configuration)
3. [Code Optimizations](#code-optimizations)
4. [Implementation Examples](#implementation-examples)
5. [Benchmarking](#benchmarking)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Essential GPU Optimizations

1. **Enable XLA optimizations** by setting environment variables before running:
```bash
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true'
export JAX_ENABLE_PGLE=true
export JAX_PGLE_PROFILING_RUNS=3
```

2. **Use mixed precision** by configuring dtypes in `configuration.json`:
```json
"Data Types": {
    "model_dtype": "bfloat16",
    "action_dtype": "int32", 
    "target_dtype": "float32"
}
```

3. **Run with one process per GPU**:
```bash
# For single GPU
python -m cumind train

# For multi-GPU (example with 4 GPUs)
python -m cumind train --num-devices 4
```

## Environment Configuration

### XLA Compiler Flags

Set these environment variables for optimal GPU performance:

```python
import os

# Basic XLA optimizations
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '  # Enable Triton GEMM kernels
    '--xla_gpu_enable_latency_hiding_scheduler=true '  # Better scheduling
    '--xla_gpu_enable_async_collectives=true '  # Async communication
)

# Profile-Guided Latency Estimation
os.environ['JAX_ENABLE_PGLE'] = 'true'
os.environ['JAX_PGLE_PROFILING_RUNS'] = '3'  # Number of profiling runs

# NCCL communication optimization (for multi-GPU)
os.environ.update({
    "NCCL_LL128_BUFFSIZE": "-2",
    "NCCL_LL_BUFFSIZE": "-2", 
    "NCCL_PROTO": "SIMPLE,LL,LL128",
})
```

### Pipeline Parallelism (Advanced)

For large models with pipeline parallelism:

```bash
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer='' --xla_disable_hlo_passes=collective-permute-motion --xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE"
```

## Code Optimizations

### 1. JIT Compilation

Add JIT compilation to performance-critical functions:

```python
# In agent.py
import jax

class Agent:
    @partial(jax.jit, static_argnames=['training'])
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        # ... existing code ...

# In trainer.py
class Trainer:
    @jax.jit
    def _loss_fn(self, params, target_params, batch):
        # ... existing code ...
```

### 2. Mixed Precision Training

Use bfloat16 for faster computation on modern GPUs (A100, V100):

```python
# In network.py
def __init__(self, ..., model_dtype: str = "bfloat16"):
    self.dtype = get_dtype(model_dtype)
    
    # Use mixed precision in layers
    self.dense = nnx.Dense(
        in_features, 
        out_features,
        dtype=self.dtype,  # Computation in bfloat16
        param_dtype=jnp.float32  # Parameters in float32
    )
```

### 3. Batch Processing Optimization

Utilize the existing `batched_apply` function for memory-efficient processing:

```python
# In trainer.py
from ..utils.jax_utils import batched_apply

# Process large batches in chunks
predictions = batched_apply(
    self.network.initial_inference,
    observations,
    batch_size=32  # Process 32 samples at a time
)
```

### 4. Device Placement

Explicitly place computations on GPU:

```python
# In agent.py
import jax

# Check available devices
print(f"Available devices: {jax.devices()}")

# Place arrays on GPU
observation_gpu = jax.device_put(observation, jax.devices('gpu')[0])
```

## Implementation Examples

### Example 1: Optimized Training Step

```python
# In trainer.py
import functools

class Trainer:
    def __init__(self, config: Config):
        # ... existing code ...
        
        # JIT compile the training step
        self._train_step_jit = jax.jit(self._train_step)
    
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def _train_step(self, params, opt_state, batch):
        """JIT-compiled training step with donated buffers."""
        grads = jax.grad(self._loss_fn)(params, self.target_params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
```

### Example 2: Mixed Precision Network

```python
# In config.py
@chex.dataclass
class Config:
    # ... existing fields ...
    
    # GPU optimization settings
    use_mixed_precision: bool = True
    jit_compile: bool = True
    xla_flags: str = "--xla_gpu_triton_gemm_any=True"
```

### Example 3: Multi-GPU Training

```python
# In runner.py
import jax
import jax.numpy as jnp
from jax import pmap

class DistributedRunner:
    def __init__(self, config: Config):
        self.num_devices = jax.device_count()
        print(f"Using {self.num_devices} GPUs")
        
        # Replicate model across devices
        self.train_step = pmap(self._train_step, axis_name='devices')
    
    def train_epoch(self, data):
        # Shard data across devices
        data_per_device = data.reshape(self.num_devices, -1, *data.shape[1:])
        
        # Parallel training step
        new_params = self.train_step(self.params, data_per_device)
```

## Benchmarking

### Performance Monitoring Script

Create `scripts/benchmark_gpu.py`:

```python
import time
import jax
import jax.numpy as jnp
from cumind import Agent, Config

def benchmark_inference(config: Config, num_runs: int = 1000):
    """Benchmark inference speed on GPU."""
    agent = Agent(config)
    observation = jnp.ones(config.observation_shape)
    
    # Warm-up
    for _ in range(10):
        _ = agent.select_action(observation, training=False)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = agent.select_action(observation, training=False)
    
    # Force completion
    jax.block_until_ready(agent.network.state)
    
    elapsed = time.time() - start_time
    print(f"Inference speed: {num_runs / elapsed:.2f} steps/second")

if __name__ == "__main__":
    # Test different configurations
    configs = [
        Config(model_dtype="float32"),
        Config(model_dtype="bfloat16"),
    ]
    
    for config in configs:
        print(f"\nTesting with dtype: {config.model_dtype}")
        benchmark_inference(config)
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM) Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Enable memory optimization: `os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'`

2. **Slow Compilation**
   - Cache compiled functions: `jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)`
   - Reduce JIT compilation overhead by batching operations

3. **Mixed Precision Instability**
   - Keep loss scaling in float32
   - Use float32 for batch normalization statistics
   - Monitor for NaN/Inf values

4. **Multi-GPU Synchronization**
   - Ensure all devices have same batch size
   - Use `psum` for gradient aggregation
   - Check NCCL environment variables

### Profiling Tools

Use JAX profiler to identify bottlenecks:

```python
# Enable profiling
import jax.profiler

# Profile a training step
with jax.profiler.trace("/tmp/jax-trace"):
    agent.train_step(batch)

# View in TensorBoard
# tensorboard --logdir=/tmp/jax-trace
```

## Future Optimizations

### Planned Improvements

1. **Automatic Mixed Precision (AMP)**
   - Implement dynamic loss scaling
   - Add automatic dtype conversion

2. **Fused Kernels**
   - Combine operations for better memory bandwidth
   - Use XLA fusion hints

3. **Quantization**
   - INT8 inference for deployment
   - Quantization-aware training

4. **Advanced Parallelism**
   - Model parallelism for large networks
   - Pipeline parallelism for sequential models

## References

- [JAX GPU Performance Tips](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
- [JAX JIT Compilation](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [XLA Optimization Passes](https://www.tensorflow.org/xla/operation_semantics)

---

Last updated: 2025-07-04  
Tested with: JAX 0.4.x, CUDA 12.x, CuMind v0.1.x