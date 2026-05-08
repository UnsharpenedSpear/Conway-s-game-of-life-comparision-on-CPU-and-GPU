# Conway's Game of Life — CPU vs GPU Benchmark

Measures **execution time**, **power consumption** (GPU board + CPU package),
and **performance-per-watt** for a 4096×4096 toroidal GoL grid over 300 generations.

---

## Quick start

### Run the benchmark

```bash
# 1. Detect your GPU architecture (check compute capability)
make info
# e.g. output: NVIDIA GeForce RTX 3070, 8.6, 220.00 W, 8192 MiB

# 2. Build  (-arch=native detects automatically; CUDA ≥ 11.6)
make

# 3. Run benchmark + plot
make plot

# Or just run without plotting
make run
```

### Run the interactive viewer

```bash
make interactive
```

This launches an interactive Game of Life simulator with visual display, pattern selection, 
generation counter, speed control, and real-time CPU/memory usage monitoring.

---

## Interactive Viewer Features

The interactive viewer (`interactive_gol.py`) provides:

### Visual Display
- **Live cell rendering** — Green cells on a dark grid
- **Real-time generation counter** — Shows current tick number
- **Population display** — Number of alive cells
- **Speed indicator** — Current ticks per second

### Pattern Selection
Press the following keys to place patterns at the grid center:
- **R** — Fill with random pattern (~30% density)
- **C** — Clear all cells
- **G** — Glider (moving, period 4)
- **B** — Blinker (oscillator, period 2)
- **K** — Block (2×2 still life)
- **T** — Toad (oscillator, period 2)
- **P** — Pentomino (complex pattern)

### Simulation Control
- **SPACE** — Play/Pause simulation
- **+/-** — Increase/Decrease simulation speed (1–60 ticks/second)
- **ESC** — Quit viewer

### Monitoring
- **Real-time CPU/Memory graph** — Shows your system's usage over time
- **FPS counter** — Display refresh rate
- **Status indicator** — PAUSED (red) or RUNNING (green)

---

## Prerequisites

| Requirement | How to check |
|---|---|
| NVIDIA GPU + driver | `nvidia-smi` |
| CUDA Toolkit (≥ 11.6) | `nvcc --version` |
| libnvidia-ml | ships with the driver (no install needed) |
| Python 3 + dependencies | `pip install -r requirements.txt` |

### Install Python dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install matplotlib pandas pygame psutil numpy
```

---

## Benchmark — How it works

### Workload — Conway's Game of Life (toroidal)
- **CPU**: Single-threaded C++ nested loops with modular boundary wrapping.
- **GPU**: CUDA kernel with 32×32 thread blocks and a shared-memory tile
  (34×34 including the 1-cell halo). Each cell update is one thread.
  Shared memory eliminates redundant global reads for neighbour cells.

### Power measurement

| Source | API | What it measures |
|---|---|---|
| GPU board | NVML `nvmlDeviceGetPowerUsage()` | Total GPU board power (mW → W) |
| CPU package | Intel RAPL sysfs (`energy_uj` delta / Δt) | CPU die + DRAM power (W) |

A background thread samples both every **50 ms** throughout each run.

### Key metrics

| Metric | Formula |
|---|---|
| Throughput | `(WIDTH × HEIGHT × GENS) / time_s` in Gcell/s |
| System power | `avg_gpu_power + avg_cpu_power` (W) |
| **Perf/Watt** | `throughput_gcell_s / system_power_W` (Gcell/s/W) |

---

## Understanding the benchmark results

- **Speedup** alone tells you how much faster the GPU is.
- **Performance-per-watt** tells you if that speed comes at an acceptable energy cost.
  A GPU that is 50× faster but draws 100× more power is *less* efficient.
- The **power trace plots** reveal idle vs active power baselines and how quickly
  the GPU ramps to full power at the start of the run.

### Typical results (RTX 3070, i7-10875H)

| Metric | CPU | GPU |
|---|---|---|
| Time | ~38 s | ~0.4 s |
| Throughput | ~1.3 Gcell/s | ~125 Gcell/s |
| System power | ~42 W | ~175 W |
| **Perf/Watt** | ~0.031 | ~0.71 |

GPU is ~100× faster and ~23× more energy-efficient for this workload.

---

## Troubleshooting

**RAPL unavailable (CPU power shows -1)**  
Check permissions: `ls -la /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`  
Fix: `sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`  
AMD CPUs: the code tries `/sys/class/powercap/amd-energy/` automatically.

**libnvidia-ml not found**  
```bash
sudo ldconfig
# or add the cuda lib path:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**`-arch=native` unsupported**  
Use an explicit arch: `ARCH := -arch=sm_86` in Makefile.

