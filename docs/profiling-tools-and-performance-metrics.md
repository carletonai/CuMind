## CuMind- Profiling Tools and Performance Metrics

# Systems Monitoring
### Tool: **_PSUTIL (Process and System Utilities)_** - Muhammed Imran

- Python library used to retrieve information on system utilization, providing metrics
    on a few main system areas:
       1. CPU
          - cpu_percent() → % CPU usage (overall or per core)
          - cpu_times() → Time spent by CPU in user/system/idle mode
          - cpu_count() → Number of cores (logical & physical)
          - cpu_freq() → Current CPU frequency
          - cpu_stats() → CPU context switches, interrupts, etc.
       2. Memory (RAM)
          - virtual_memory() → Total, available, used, percent, etc.
          - swap_memory() → Swap/paging memory stats
       3. Disk
          - disk_usage(path) → Total, used, free space on a disk
          - disk_partitions() → Mounted disk partitions
          - disk_io_counters() → Read/write counts, bytes, time, etc.
       4. Network
          - net_io_counters() → Total bytes sent/received, packets, errors
          - net_if_addrs() → IP addresses per network interface
          - net_if_stats() → Interface speed, up/down status


- net_connections() → Open network connections (sockets)
_Analysis: PSUTIL (Process and System Utilities)_ - Muhammed Imran
Psutil only requires one line to install on Linux, Windows and macOS:
**Pip install psutil**
It can be used anywhere and most, if not all of the required systems monitoring areas
are covered by this library and the functions it provides, which can easily be used in
print functions wherever needed, anywhere in CuMind.

# Runtime Analysis
### Tool: **_line_profiler_** - Muhammed Imran

- A Python performance profiling tool that shows how much time each individual
    line of your function takes to run, so unlike general profilers, like CProfile, which
    shows how long each function takes to run, it gives a more magnified view of
    exactly what is causing program latency issues.
- Best used when you know a function is slow (after using CProfile) and you need
    the exact line that is causing the issue.
- When working on CPU-heavy code like loops or math heavy logic (think CuMind
    🤪!)
_Analysis: line_profiler_ - Muhammed Imran
*THE FOLLOWING IS THE LEGACY INSTALLATION, VIEW MODERN ON GITHUB
REPO: pyutils/line_profiler: Line-by-line profiling for Python
- Install with **pip install line_profiler**
- Then add @profile decorator in the line right above the function you want to
profile, then run script with **kernprof** (kernprof -l file.py)
- Finally view results with:
**python -m line_profiler file.py.lprof**
For our project, line_profiler would be invaluable and should be used in conjunction with
CProfile, as it would be redundant running it everywhere.


Line Profiling should be noted to have some drawbacks however, so that should
influence where it is used:

1. Not good for multi-threaded/process code, as it can show incomplete/confusing
    results or nothing at all
2. Line profiling only measures the time Python spends waiting for GPU calls to
    finish, not measuring actual GPU computation time if it happens asynchonously.
   
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
