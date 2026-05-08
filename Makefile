# ── Conway's Game of Life — CUDA Benchmark Makefile ──────────────────────────
#
# Requirements:
#   - CUDA Toolkit (tested 11.6+)
#   - libnvidia-ml (ships with the NVIDIA driver; usually in /usr/lib or /usr/local/cuda/lib64)
#   - Python 3 + matplotlib + pandas (for plot_results.py)
#
# Finding your GPU architecture:
#   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
#   e.g.  7.5 → sm_75 (Turing, RTX 20xx)
#         8.6 → sm_86 (Ampere, RTX 30xx)
#         8.9 → sm_89 (Ada, RTX 40xx)
#         9.0 → sm_90 (Hopper)
#
# -arch=native  detects automatically (CUDA 11.6+, recommended)
# ─────────────────────────────────────────────────────────────────────────────

NVCC      := nvcc
BIN       := benchmark
SRC       := benchmark.cu

# Compiler flags
ARCH      := -arch=native          # auto-detect; replace with e.g. -arch=sm_86 if needed
OPT       := -O3 --use_fast_math
STD       := -std=c++17
WFLAGS    := -Xcompiler -Wall,-Wextra
NVCCFLAGS := $(ARCH) $(OPT) $(STD) $(WFLAGS)

# Linker flags — link against NVML (NVIDIA Management Library)
LIBS      := -lnvidia-ml

# CUDA toolkit paths (usually /usr/local/cuda)
CUDA_PATH ?= /usr/local/cuda
INCLUDES  := -I$(CUDA_PATH)/include
LDFLAGS   := -L/usr/local/cuda/lib64 -lnvidia-ml

# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all clean run plot info interactive

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "Build OK → ./$(BIN)"

# Run the benchmark and then plot
run: $(BIN)
	@echo "═══ Running benchmark ═══════════════════════════════════════════"
	./$(BIN)

plot: run
	@echo "═══ Plotting results ════════════════════════════════════════════"
	python3 plot_results.py

# Run the interactive Game of Life viewer
interactive:
	@echo "═══ Starting interactive Game of Life viewer ═════════════════════"
	python3 interactive_gol.py

# Print GPU info before building
info:
	@nvidia-smi --query-gpu=name,compute_cap,power.limit,memory.total \
	            --format=csv,noheader

clean:
	rm -f $(BIN) summary.csv power_series.csv *.png
