## CuMind- Profiling Tools and Performance Metrics

# Systems Monitoring

# Runtime Analysis

### Tool: **_cProfile_** - Ishar Ghura
Docs
Python built-in deterministic profiler, provides low-overhead way to profile programs and
determine where time is being spent
**Investigation** :
- Built-in: import cProfile
- Measures functional call count, total time, cumulative time
- Outputs can be read via pstats or visualized with SnakeViz or gprof2dot
- Lightweight and suitable for CPU-bound performance bottlenecks
- Can help pinpoint slow model forward passes, environment simulations, or
MCTS evaluations
- Works well even with JAX python frontends, but wont capture jax compiled jit
functions internally
**Analysis:**
1. Ex: cProfile on compute-heavy components like SelfPlay.run_episode() or
Trainer.train_step()
a. Limitations: cProfile will only measure python time, it wont peek into @jit,
but:
- Can identify python overhead or inefficient data prep
- Profile bottlenecks in logic like sampling, logging, checkpointing
It’s pretty easy to use and gives an overview of all slow functions
2. Use @profile decorators or you can manually wrap MCTS search, model
inference, and environment steps
3. Combine with visualization: SnakeViz
a. pip install snakeviz
b. snakeviz cumind_profile.prof
Limitations:


JAX’s @jit functions execute in compiled XLA which cannot be traced by cProfile, to
profile JAX internals you need tools like jax.profiler.start_trace(“output_dir”)
So, cProfile is still highly useful for profiling the python layer: data pipeline, MCTS,
logging, trainer orchestration