#!/usr/bin/env python3
"""
plot_results.py — Analyse and visualise benchmark output from benchmark.cu

Reads:
  summary.csv      — aggregated metrics per device
  power_series.csv — time-series power samples during each run

Produces:
  results_overview.png  — 4-panel figure
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── Colour palette ────────────────────────────────────────────────────────────
CPU_COL = "#4C9BE8"   # blue
GPU_COL = "#E8834C"   # orange
POWER_GPU_COL = "#E84C4C"
POWER_CPU_COL = "#4CE87A"

# ── Load data ─────────────────────────────────────────────────────────────────
def load():
    for f in ("summary.csv", "power_series.csv"):
        if not pathlib.Path(f).exists():
            sys.exit(f"[error] {f} not found — run `make run` first.")
    summary = pd.read_csv("summary.csv")
    power   = pd.read_csv("power_series.csv")
    return summary, power

# ── Helpers ───────────────────────────────────────────────────────────────────
def bar_pair(ax, labels, cpu_val, gpu_val, ylabel, title, fmt=".2f", unit=""):
    x = np.arange(len(labels))
    bars_c = ax.bar(x - 0.2, cpu_val, 0.38, label="CPU", color=CPU_COL, zorder=3)
    bars_g = ax.bar(x + 0.2, gpu_val, 0.38, label="GPU", color=GPU_COL, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    for bars in (bars_c, bars_g):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                        f"{h:{fmt}}{unit}", ha="center", va="bottom", fontsize=7.5)

def power_timeseries(ax, power, phase, label):
    sub = power[power.phase == phase].copy()
    if sub.empty:
        ax.set_title(f"{label} (no data)"); return
    t = sub["time_s"].values
    gpu_w = sub["gpu_power_w"].replace(-1, np.nan)
    cpu_w = sub["cpu_power_w"].replace(-1, np.nan)
    ax.plot(t, gpu_w, color=POWER_GPU_COL, lw=1.6, label="GPU board power")
    ax.plot(t, cpu_w, color=POWER_CPU_COL, lw=1.6, label="CPU package power")
    ax.fill_between(t, 0, gpu_w.fillna(0), alpha=0.15, color=POWER_GPU_COL)
    ax.fill_between(t, 0, cpu_w.fillna(0), alpha=0.15, color=POWER_CPU_COL)
    # annotate mean
    if not gpu_w.isna().all():
        m = gpu_w.mean()
        ax.axhline(m, color=POWER_GPU_COL, ls="--", lw=0.9, alpha=0.7)
        ax.text(t[-1], m, f" {m:.1f}W", va="center", fontsize=7.5, color=POWER_GPU_COL)
    if not cpu_w.isna().all():
        m = cpu_w.mean()
        ax.axhline(m, color=POWER_CPU_COL, ls="--", lw=0.9, alpha=0.7)
        ax.text(t[-1], m, f" {m:.1f}W", va="center", fontsize=7.5, color=POWER_CPU_COL)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Power (W)",  fontsize=9)
    ax.set_title(f"Power trace — {label} run", fontweight="bold")
    ax.legend(fontsize=8)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    summary, power = load()

    # Ensure we have both rows
    cpu_row = summary[summary.device == "CPU"].iloc[0]
    gpu_row = summary[summary.device == "GPU"].iloc[0]

    speedup = cpu_row["time_s"] / gpu_row["time_s"]
    eff_ratio = (gpu_row["perf_per_watt_gcells_s_w"] /
                 cpu_row["perf_per_watt_gcells_s_w"]) if cpu_row["perf_per_watt_gcells_s_w"] > 0 else 0

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Conway's Game of Life — CPU vs GPU Analysis", fontsize=13, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig)

    # ── Panel 1: Throughput ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    devices = ["Throughput\n(Gcell/s)"]
    bar_pair(ax1, devices,
             [cpu_row["throughput_gcells_s"]],
             [gpu_row["throughput_gcells_s"]],
             "Gcell/s", f"Throughput  ({speedup:.1f}× GPU speedup)", fmt=".2f")

    # ── Panel 2: Power breakdown ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ["GPU board", "CPU pkg", "Total"]
    c_vals = [cpu_row["avg_gpu_power_w"], cpu_row["avg_cpu_power_w"], cpu_row["total_power_w"]]
    g_vals = [gpu_row["avg_gpu_power_w"], gpu_row["avg_cpu_power_w"], gpu_row["total_power_w"]]
    bar_pair(ax2, categories, c_vals, g_vals, "Watts", "Average Power Draw", fmt=".1f", unit="W")

    # ── Panel 3: Performance-per-watt ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ppw_label = ["Perf/Watt\n(Gcell/s/W)"]
    bar_pair(ax3, ppw_label,
             [cpu_row["perf_per_watt_gcells_s_w"]],
             [gpu_row["perf_per_watt_gcells_s_w"]],
             "Gcell/s/W", f"Efficiency  ({eff_ratio:.1f}× GPU)", fmt=".3f")

    # ── Panel 4: CPU run power trace ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    power_timeseries(ax4, power, "cpu", "CPU workload")

    # ── Panel 5: GPU run power trace ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    power_timeseries(ax5, power, "gpu", "GPU workload")

    # ── Annotation box ────────────────────────────────────────────────────────
    ann = (
        f"Grid: {summary.attrs.get('grid', '?')}     "
        f"Speedup: {speedup:.1f}×     "
        f"GPU efficiency ratio: {eff_ratio:.2f}×     "
        f"CPU time: {cpu_row['time_s']:.2f}s     GPU time: {gpu_row['time_s']:.3f}s"
    )
    fig.text(0.5, -0.01, ann, ha="center", fontsize=8.5, style="italic", color="grey")

    out = "results_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
