// =============================================================================
// Conway's Game of Life — CPU vs GPU Benchmark
// Measures: execution time, power consumption, performance-per-watt
//
// Power sources:
//   GPU  → NVML (nvidia-ml)           — direct hardware register read
//   CPU  → Intel RAPL via sysfs       — energy counter delta / time
//          (AMD: use /sys/class/powercap/amd_energy/ or hwmon instead)
//
// Command-line options:
//   --pattern <type>    Random starting pattern: random (default), glider, block, blinker
//   --density <0-1>     Density for random pattern (default: 0.30)
//   --generations <N>   Number of generations to simulate (default: 300)
//   --help              Show usage
// =============================================================================

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ── Configuration ─────────────────────────────────────────────────────────────
// Grid size must be a multiple of BLOCK_DIM for the shared-memory kernel.
static constexpr int WIDTH        = 4096;   // cells wide
static constexpr int HEIGHT       = 4096;   // cells tall
static constexpr int BLOCK_DIM    = 32;     // CUDA block dimension (32×32 = 1024 threads)
static constexpr int SAMPLE_MS    = 50;     // power sample interval (ms)
static constexpr int GPU_WARMUP   = 5;      // warm-up iterations before timing

// Command-line configurable
static int GENERATIONS = 300;    // default steps to simulate
static std::string PATTERN_TYPE = "random";  // default pattern
static double PATTERN_DENSITY = 0.30;        // default density for random pattern

// ── CUDA / NVML error helpers ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e)          \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define NVML_CHECK(call)                                                    \
    do {                                                                    \
        nvmlReturn_t _r = (call);                                           \
        if (_r != NVML_SUCCESS) {                                           \
            std::cerr << "NVML warning: " << nvmlErrorString(_r) << "\n";  \
        }                                                                   \
    } while (0)

// ── Power monitoring ──────────────────────────────────────────────────────────
struct PowerSample {
    double gpu_w;   // GPU board power (W), from NVML
    double cpu_w;   // CPU package power (W), derived from RAPL delta
};

class PowerMonitor {
public:
    explicit PowerMonitor(int gpu_index = 0) {
        nvmlReturn_t r = nvmlInit_v2();
        nvml_ok_ = (r == NVML_SUCCESS);
        if (nvml_ok_) {
            NVML_CHECK(nvmlDeviceGetHandleByIndex(gpu_index, &dev_));
            char name[128];
            nvmlDeviceGetName(dev_, name, sizeof(name));
            gpu_name_ = name;

            unsigned int tdp_mw = 0;
            nvmlDeviceGetPowerManagementLimit(dev_, &tdp_mw);
            gpu_tdp_w_ = tdp_mw / 1000.0;
        }
    }

    ~PowerMonitor() {
        if (nvml_ok_) nvmlShutdown();
    }

    const std::string& gpu_name()  const { return gpu_name_; }
    double             gpu_tdp_w() const { return gpu_tdp_w_; }

    void start() {
        samples_.clear();
        running_ = true;
        thread_ = std::thread([this] { loop(); });
    }

    std::vector<PowerSample> stop() {
        running_ = false;
        thread_.join();
        return samples_;
    }

private:
    nvmlDevice_t             dev_{};
    bool                     nvml_ok_ = false;
    std::string              gpu_name_;
    double                   gpu_tdp_w_ = 0;
    std::atomic<bool>        running_{false};
    std::thread              thread_;
    std::vector<PowerSample> samples_;

    // RAPL: read the package energy counter in microjoules
    // Returns -1 if unavailable (AMD CPU, no permission, etc.)
    static double rapl_uj() {
        // Try Intel RAPL first
        std::ifstream f("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
        if (f.is_open()) {
            double uj; f >> uj; return uj;
        }
        // AMD fallback path (kernel 5.8+ amd_energy driver)
        f.open("/sys/class/powercap/amd-energy/amd-energy:0/energy_uj");
        if (f.is_open()) {
            double uj; f >> uj; return uj;
        }
        return -1.0;
    }

    void loop() {
        double prev_uj = rapl_uj();
        auto   prev_t  = std::chrono::steady_clock::now();

        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SAMPLE_MS));

            // ── GPU power via NVML ───────────────────────────────────────────
            double gpu_w = -1.0;
            if (nvml_ok_) {
                unsigned int mw = 0;
                if (nvmlDeviceGetPowerUsage(dev_, &mw) == NVML_SUCCESS)
                    gpu_w = mw / 1000.0;
            }

            // ── CPU power via RAPL delta ─────────────────────────────────────
            double cur_uj = rapl_uj();
            auto   cur_t  = std::chrono::steady_clock::now();
            double dt     = std::chrono::duration<double>(cur_t - prev_t).count();

            double cpu_w = -1.0;
            if (cur_uj >= 0 && prev_uj >= 0 && dt > 0)
                cpu_w = (cur_uj - prev_uj) * 1e-6 / dt;  // µJ → J, then / s = W

            prev_uj = cur_uj;
            prev_t  = cur_t;

            samples_.push_back({gpu_w, cpu_w});
        }
    }
};

// ── CPU Game of Life (single-threaded, baseline) ──────────────────────────────
// Uses toroidal (wrapping) boundary conditions.
void cpu_gol_step(const uint8_t* __restrict__ in,
                        uint8_t* __restrict__ out,
                  int w, int h)
{
    for (int y = 0; y < h; ++y) {
        int yn = (y - 1 + h) % h;
        int yp = (y + 1)     % h;
        for (int x = 0; x < w; ++x) {
            int xn = (x - 1 + w) % w;
            int xp = (x + 1)     % w;

            int live = in[yn*w + xn] + in[yn*w + x] + in[yn*w + xp]
                     + in[ y*w + xn]                + in[ y*w + xp]
                     + in[yp*w + xn] + in[yp*w + x] + in[yp*w + xp];

            uint8_t c    = in[y*w + x];
            out[y*w + x] = static_cast<uint8_t>(
                (c && (live == 2 || live == 3)) || (!c && live == 3));
        }
    }
}

// ── GPU Kernel — shared memory tile with 1-cell halo ─────────────────────────
// Each 32×32 block loads a 34×34 shared tile (including wrapping neighbours).
// This eliminates redundant global memory reads for halo cells.
__global__ void gol_kernel(const uint8_t* __restrict__ in,
                                  uint8_t* __restrict__ out,
                            int w, int h)
{
    // Shared memory tile: centre (BLOCK_DIM × BLOCK_DIM) + 1-cell border
    __shared__ uint8_t s[BLOCK_DIM + 2][BLOCK_DIM + 2];

    const int gx = blockIdx.x * blockDim.x + threadIdx.x;   // global column
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;    // global row
    const int tx = threadIdx.x + 1;                          // shared column (1-indexed)
    const int ty = threadIdx.y + 1;                          // shared row   (1-indexed)

    // Wrapping helpers
    const int gxL = (gx - 1 + w) % w,  gxR = (gx + 1) % w;
    const int gyT = (gy - 1 + h) % h,  gyB = (gy + 1) % h;

    // Centre cell
    s[ty][tx] = in[gy * w + gx];

    // Edge halos (one thread per edge handles the extra column/row)
    if (threadIdx.x == 0)              s[ty][0]           = in[gy * w + gxL];
    if (threadIdx.x == blockDim.x - 1) s[ty][BLOCK_DIM+1] = in[gy * w + gxR];
    if (threadIdx.y == 0)              s[0][tx]           = in[gyT * w + gx];
    if (threadIdx.y == blockDim.y - 1) s[BLOCK_DIM+1][tx] = in[gyB * w + gx];

    // Corner halos (only 4 threads per block write these)
    if (threadIdx.x == 0 && threadIdx.y == 0)
        s[0][0]                       = in[gyT * w + gxL];
    if (threadIdx.x == blockDim.x-1 && threadIdx.y == 0)
        s[0][BLOCK_DIM+1]             = in[gyT * w + gxR];
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y-1)
        s[BLOCK_DIM+1][0]             = in[gyB * w + gxL];
    if (threadIdx.x == blockDim.x-1 && threadIdx.y == blockDim.y-1)
        s[BLOCK_DIM+1][BLOCK_DIM+1]   = in[gyB * w + gxR];

    __syncthreads();

    if (gx >= w || gy >= h) return;

    // Count live neighbours from shared tile
    int live = s[ty-1][tx-1] + s[ty-1][tx] + s[ty-1][tx+1]
             + s[ty  ][tx-1]               + s[ty  ][tx+1]
             + s[ty+1][tx-1] + s[ty+1][tx] + s[ty+1][tx+1];

    uint8_t c     = s[ty][tx];
    out[gy*w + gx] = static_cast<uint8_t>(
        (c && (live == 2 || live == 3)) || (!c && live == 3));
}

// ── Statistics ────────────────────────────────────────────────────────────────
static double vec_mean(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
static double vec_max(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    return *std::max_element(v.begin(), v.end());
}

// ── Pattern Initialization ───────────────────────────────────────────────────
static void init_random_pattern(std::vector<uint8_t>& grid, double density) {
    std::mt19937 rng(42);
    std::bernoulli_distribution bern(density);
    for (auto& c : grid) c = static_cast<uint8_t>(bern(rng));
}

static void init_glider_pattern(std::vector<uint8_t>& grid) {
    // Place gliders at regular intervals
    for (int y = 0; y < HEIGHT; y += 50) {
        for (int x = 0; x < WIDTH; x += 50) {
            // Glider pattern
            grid[(y + 1)*WIDTH + (x + 2)] = 1;
            grid[(y + 2)*WIDTH + (x + 3)] = 1;
            grid[(y + 3)*WIDTH + (x + 1)] = 1;
            grid[(y + 3)*WIDTH + (x + 2)] = 1;
            grid[(y + 3)*WIDTH + (x + 3)] = 1;
        }
    }
}

static void init_block_pattern(std::vector<uint8_t>& grid) {
    // Place 2×2 blocks at regular intervals
    for (int y = 0; y < HEIGHT; y += 40) {
        for (int x = 0; x < WIDTH; x += 40) {
            grid[y*WIDTH + x] = 1;
            grid[y*WIDTH + (x+1)] = 1;
            grid[(y+1)*WIDTH + x] = 1;
            grid[(y+1)*WIDTH + (x+1)] = 1;
        }
    }
}

static void init_blinker_pattern(std::vector<uint8_t>& grid) {
    // Place horizontal blinkers at regular intervals
    for (int y = 0; y < HEIGHT; y += 40) {
        for (int x = 0; x < WIDTH; x += 40) {
            grid[y*WIDTH + x] = 1;
            grid[y*WIDTH + (x+1)] = 1;
            grid[y*WIDTH + (x+2)] = 1;
        }
    }
}

// ── Command-line Parsing ─────────────────────────────────────────────────────
static void parse_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "\nOptions:\n"
                      << "  --pattern <type>      Starting pattern: random, glider, block, blinker\n"
                      << "                        (default: random)\n"
                      << "  --density <0-1>       Density for random pattern (default: 0.30)\n"
                      << "  --generations <N>     Number of generations (default: 300)\n"
                      << "  --help                Show this message\n"
                      << "\nExample:\n"
                      << "  " << argv[0] << " --pattern glider --generations 500\n";
            std::exit(0);
        } else if (arg == "--pattern" && i + 1 < argc) {
            PATTERN_TYPE = argv[++i];
        } else if (arg == "--density" && i + 1 < argc) {
            PATTERN_DENSITY = std::stod(argv[++i]);
            PATTERN_DENSITY = std::max(0.0, std::min(1.0, PATTERN_DENSITY));
        } else if (arg == "--generations" && i + 1 < argc) {
            GENERATIONS = std::max(1, std::stoi(argv[++i]));
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static void write_power_csv(const std::string& path,
                             const std::vector<PowerSample>& cpu_run,
                             const std::vector<PowerSample>& gpu_run)
{
    std::ofstream f(path);
    f << "phase,sample_index,time_s,gpu_power_w,cpu_power_w\n";
    f << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < cpu_run.size(); ++i)
        f << "cpu," << i << "," << i * SAMPLE_MS / 1000.0
          << "," << cpu_run[i].gpu_w << "," << cpu_run[i].cpu_w << "\n";
    for (size_t i = 0; i < gpu_run.size(); ++i)
        f << "gpu," << i << "," << i * SAMPLE_MS / 1000.0
          << "," << gpu_run[i].gpu_w << "," << gpu_run[i].cpu_w << "\n";
}

static void write_summary_csv(const std::string& path,
                               const std::string& label,
                               double time_s, double throughput,
                               double avg_gpu_w, double peak_gpu_w,
                               double avg_cpu_w, double peak_cpu_w,
                               double total_w,   double ppw,
                               bool first)
{
    std::ofstream f(path, first ? std::ios::out : std::ios::app);
    if (first)
        f << "device,time_s,throughput_gcells_s,"
          << "avg_gpu_power_w,peak_gpu_power_w,"
          << "avg_cpu_power_w,peak_cpu_power_w,"
          << "total_power_w,perf_per_watt_gcells_s_w\n";
    f << std::fixed << std::setprecision(4)
      << label     << "," << time_s     << "," << throughput  << ","
      << avg_gpu_w << "," << peak_gpu_w << ","
      << avg_cpu_w << "," << peak_cpu_w << ","
      << total_w   << "," << ppw        << "\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // ── Parse command-line arguments ───────────────────────────────────────
    parse_args(argc, argv);

    // ── Setup ──────────────────────────────────────────────────────────────
    const int N = WIDTH * HEIGHT;

    // Print device info
    {
        int dev;  cudaGetDevice(&dev);
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        std::cout << "GPU : " << p.name
                  << "  (SMs=" << p.multiProcessorCount
                  << ", Mem=" << p.totalGlobalMem / (1<<20) << " MiB)\n";
    }

    PowerMonitor pm;
    if (!pm.gpu_name().empty())
        std::cout << "NVML: " << pm.gpu_name()
                  << "  TDP=" << pm.gpu_tdp_w() << " W\n";

    std::cout << "Grid : " << WIDTH << " × " << HEIGHT
              << "  (" << N / 1e6 << " Mcells)\n"
              << "Gens : " << GENERATIONS << "\n"
              << "Total: " << (double)N * GENERATIONS / 1e9 << " Gcell-updates\n"
              << "Pattern: " << PATTERN_TYPE;
    if (PATTERN_TYPE == "random") {
        std::cout << " (density=" << PATTERN_DENSITY << ")";
    }
    std::cout << "\n\n";

    // Initialise grid with selected pattern
    std::vector<uint8_t> h_init(N, 0);
    {
        if (PATTERN_TYPE == "random") {
            init_random_pattern(h_init, PATTERN_DENSITY);
        } else if (PATTERN_TYPE == "glider") {
            init_glider_pattern(h_init);
        } else if (PATTERN_TYPE == "block") {
            init_block_pattern(h_init);
        } else if (PATTERN_TYPE == "blinker") {
            init_blinker_pattern(h_init);
        } else {
            std::cerr << "Unknown pattern: " << PATTERN_TYPE << "\n";
            std::cerr << "Use --help for valid options.\n";
            return 1;
        }
    }

    // ── CPU benchmark ──────────────────────────────────────────────────────
    std::cout << "━━━ CPU benchmark ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::vector<uint8_t> cpu_a = h_init, cpu_b(N);

    pm.start();
    auto cpu_t0 = std::chrono::steady_clock::now();

    for (int g = 0; g < GENERATIONS; ++g) {
        cpu_gol_step(cpu_a.data(), cpu_b.data(), WIDTH, HEIGHT);
        std::swap(cpu_a, cpu_b);
    }

    auto cpu_t1   = std::chrono::steady_clock::now();
    auto cpu_pwr  = pm.stop();
    double cpu_s  = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();
    std::cout << "  Done in " << cpu_s << " s\n";

    // ── GPU benchmark ──────────────────────────────────────────────────────
    std::cout << "━━━ GPU benchmark ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    uint8_t *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N));
    CUDA_CHECK(cudaMalloc(&d_b, N));
    CUDA_CHECK(cudaMemcpy(d_a, h_init.data(), N, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((WIDTH  + BLOCK_DIM - 1) / BLOCK_DIM,
              (HEIGHT + BLOCK_DIM - 1) / BLOCK_DIM);

    // Warm-up (fills caches, spins up clocks)
    std::cout << "  Warming up (" << GPU_WARMUP << " iters)...\n";
    for (int g = 0; g < GPU_WARMUP; ++g) {
        gol_kernel<<<grid, block>>>(d_a, d_b, WIDTH, HEIGHT);
        std::swap(d_a, d_b);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset to initial state for fair comparison
    CUDA_CHECK(cudaMemcpy(d_a, h_init.data(), N, cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    pm.start();
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int g = 0; g < GENERATIONS; ++g) {
        gol_kernel<<<grid, block>>>(d_a, d_b, WIDTH, HEIGHT);
        std::swap(d_a, d_b);
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    auto gpu_pwr = pm.stop();

    float gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_end));
    double gpu_s = gpu_ms / 1000.0;
    std::cout << "  Done in " << gpu_s << " s\n\n";

    // Cleanup GPU resources
    cudaFree(d_a); cudaFree(d_b);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);

    // ── Aggregate power samples ────────────────────────────────────────────
    auto filter_valid = [](const std::vector<PowerSample>& s, bool gpu) {
        std::vector<double> out;
        for (auto& p : s) {
            double v = gpu ? p.gpu_w : p.cpu_w;
            if (v > 0) out.push_back(v);
        }
        return out;
    };

    auto cpu_gpu_w = filter_valid(cpu_pwr, true);
    auto cpu_cpu_w = filter_valid(cpu_pwr, false);
    auto gpu_gpu_w = filter_valid(gpu_pwr, true);
    auto gpu_cpu_w = filter_valid(gpu_pwr, false);

    double cpu_avg_gpu = vec_mean(cpu_gpu_w), cpu_pk_gpu = vec_max(cpu_gpu_w);
    double cpu_avg_cpu = vec_mean(cpu_cpu_w), cpu_pk_cpu = vec_max(cpu_cpu_w);
    double gpu_avg_gpu = vec_mean(gpu_gpu_w), gpu_pk_gpu = vec_max(gpu_gpu_w);
    double gpu_avg_cpu = vec_mean(gpu_cpu_w), gpu_pk_cpu = vec_max(gpu_cpu_w);

    double cpu_total_w = (cpu_avg_gpu > 0 ? cpu_avg_gpu : 0)
                       + (cpu_avg_cpu > 0 ? cpu_avg_cpu : 0);
    double gpu_total_w = (gpu_avg_gpu > 0 ? gpu_avg_gpu : 0)
                       + (gpu_avg_cpu > 0 ? gpu_avg_cpu : 0);

    // ── Derived metrics ────────────────────────────────────────────────────
    double total_cells   = static_cast<double>(N) * GENERATIONS;
    double cpu_gcell_s   = total_cells / cpu_s / 1e9;
    double gpu_gcell_s   = total_cells / gpu_s / 1e9;
    double speedup       = cpu_s / gpu_s;

    // Performance-per-watt: Gcell/s per watt of total system power
    double cpu_ppw = cpu_total_w > 0 ? cpu_gcell_s / cpu_total_w : -1;
    double gpu_ppw = gpu_total_w > 0 ? gpu_gcell_s / gpu_total_w : -1;


    auto W = [](int w) { return std::setw(w); };
    auto COL_W = 14;
    std::cout << std::fixed << std::setprecision(4);

    std::cout
        << "╔═══════════════════════════════════════════════════════════════╗\n"
        << "║              CONWAY'S GAME OF LIFE — BENCHMARK RESULTS        ║\n"
        << "╠═══════════════════════╦═══════════════╦═══════════════════════╣\n"
        << "║  Metric               ║     CPU       ║     GPU               ║\n"
        << "╠═══════════════════════╬═══════════════╬═══════════════════════╣\n";

    auto row = [&](const std::string& lbl, double c, double g, const std::string& unit, int prec=2) {
        std::cout << "║  " << std::left << std::setw(21) << lbl << "║"
                  << std::right << std::setw(COL_W) << std::fixed << std::setprecision(prec) << c
                  << " ║" << std::setw(COL_W) << g << "  " << std::left << std::setw(7) << unit << "║\n";
    };

    row("Execution time (s)",   cpu_s,       gpu_s,       "s",        3);
    row("Throughput (Gcell/s)", cpu_gcell_s, gpu_gcell_s, "Gcell/s",  2);
    row("Speedup",              1.0,         speedup,     "x",        1);
    std::cout << "╠═══════════════════════╬═══════════════╬═══════════════════════╣\n";
    row("Avg GPU power (W)",    cpu_avg_gpu, gpu_avg_gpu, "W",        1);
    row("Peak GPU power (W)",   cpu_pk_gpu,  gpu_pk_gpu,  "W",        1);
    row("Avg CPU power (W)",    cpu_avg_cpu, gpu_avg_cpu, "W",        1);
    row("Total system power",   cpu_total_w, gpu_total_w, "W",        1);
    std::cout << "╠═══════════════════════╬═══════════════╬═══════════════════════╣\n";
    row("Perf/Watt (Gc/s/W)",  cpu_ppw,     gpu_ppw,     "Gc/s/W",   3);
    std::cout << "╚═══════════════════════╩═══════════════╩═══════════════════════╝\n\n";

    if (cpu_ppw > 0 && gpu_ppw > 0) {
        double eff_ratio = gpu_ppw / cpu_ppw;
        std::cout << "→  GPU throughput is " << std::fixed << std::setprecision(1)
                  << speedup << "× faster than CPU.\n";
        std::cout << "→  GPU is " << eff_ratio << "× more efficient (performance-per-watt).\n";
        if (eff_ratio < 1.0)
            std::cout << "   (GPU uses more power than the speedup justifies at this workload size.)\n";
    }

    // ── Write CSVs ────────────────────────────────────────────────────────
    write_power_csv("power_series.csv", cpu_pwr, gpu_pwr);
    write_summary_csv("summary.csv", "CPU", cpu_s, cpu_gcell_s,
                      cpu_avg_gpu, cpu_pk_gpu, cpu_avg_cpu, cpu_pk_cpu,
                      cpu_total_w, cpu_ppw, true);
    write_summary_csv("summary.csv", "GPU", gpu_s, gpu_gcell_s,
                      gpu_avg_gpu, gpu_pk_gpu, gpu_avg_cpu, gpu_pk_cpu,
                      gpu_total_w, gpu_ppw, false);

    std::cout << "\nResults written to summary.csv and power_series.csv\n";
    return 0;
}
