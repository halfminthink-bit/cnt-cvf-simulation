"""Step 3-B: モンテカルロ MC 実行スクリプト.

実験条件（15 min サンプル、24 s/drop、h_film=25 nm）で N=10000 本の CNT を
シミュレートし、G ピーク比を計算する。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cnt_cvf.config import (
    CNT_DIAMETER,
    DROP_VOLUME,
    EFFECTIVE_AREA,
    PORE_DENSITY,
    TEMPERATURE,
    VISCOSITY,
)
from cnt_cvf.runner import build_pe_to_costheta_table, run_monte_carlo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3-B Monte Carlo simulation")
    parser.add_argument(
        "--length_csv",
        default=str(PROJECT_ROOT / "data/length_distributions/15min_pos01_final_lengths.csv"),
        help="長さ分布 CSV パス",
    )
    parser.add_argument("--n_cnt", type=int, default=10000, help="CNT 本数")
    parser.add_argument("--drop_interval", type=float, default=24.0, help="drop 間隔 [s]")
    parser.add_argument("--h_film", type=float, default=25e-9, help="膜上高さ範囲 [m]")
    parser.add_argument("--area_size", type=float, default=5e-6, help="ポア生成領域 [m]")
    parser.add_argument("--seed_table", type=int, default=0, help="Pe テーブル用シード")
    parser.add_argument("--seed_mc", type=int, default=42, help="MC 本番用シード")
    parser.add_argument(
        "--pe_n_real", type=int, default=20, help="Pe テーブルの realization 数"
    )
    parser.add_argument(
        "--pe_n_steps", type=int, default=20000, help="Pe テーブルの積分ステップ数"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Step 3-B: Monte Carlo CNT orientation simulation")
    print("=" * 60)
    print(f"  length_csv   : {args.length_csv}")
    print(f"  n_cnt        : {args.n_cnt}")
    print(f"  drop_interval: {args.drop_interval} s")
    print(f"  h_film       : {args.h_film * 1e9:.1f} nm")
    print(f"  area_size    : {args.area_size * 1e6:.1f} µm")
    print(f"  seed_table   : {args.seed_table}")
    print(f"  seed_mc      : {args.seed_mc}")
    print()

    # ── Pe テーブル構築 ────────────────────────────────────────────────────────
    print("Building Pe → <cos²θ> table ...")
    pe_values = np.logspace(-2, 4, 50)
    rng_table = np.random.default_rng(args.seed_table)
    pe_table = build_pe_to_costheta_table(
        pe_values=pe_values,
        n_realization=args.pe_n_real,
        n_steps=args.pe_n_steps,
        rng=rng_table,
    )
    print(f"  Pe grid: [{pe_values[0]:.2e}, {pe_values[-1]:.2e}], {len(pe_values)} points")
    print()

    # ── MC 本番 ───────────────────────────────────────────────────────────────
    print("Running Monte Carlo ...")
    rng_mc = np.random.default_rng(args.seed_mc)
    result = run_monte_carlo(
        n_cnt=args.n_cnt,
        h_film=args.h_film,
        area_size=args.area_size,
        length_csv=args.length_csv,
        drop_interval=args.drop_interval,
        pe_table=pe_table,
        rng=rng_mc,
        n_z_layers=5,
    )
    print("  Done.")
    print()

    # ── 統計計算 ──────────────────────────────────────────────────────────────
    lengths_nm = result["lengths"] * 1e9
    gamma_dots = result["gamma_dots"]
    pe_vals = result["pe_values"]
    cos2_arr = result["angles"]
    g_ratio = result["g_ratio"]
    exp_value = 1.59  # 15 min サンプルの実験値

    stats = {
        "Pe_mean": float(np.mean(pe_vals)),
        "Pe_median": float(np.median(pe_vals)),
        "Pe_p99": float(np.percentile(pe_vals, 99)),
        "gamma_mean": float(np.mean(gamma_dots)),
        "gamma_median": float(np.median(gamma_dots)),
        "L_mean_nm": float(np.mean(lengths_nm)),
        "L_median_nm": float(np.median(lengths_nm)),
        "cos2_mean": float(np.mean(cos2_arr)),
        "cos2_median": float(np.median(cos2_arr)),
        "g_ratio_predicted": g_ratio,
        "g_ratio_experimental": exp_value,
        "g_ratio_relative_error_pct": (g_ratio - exp_value) / exp_value * 100,
    }

    # ── summary.txt 出力 ──────────────────────────────────────────────────────
    summary_path = results_dir / "step3b_summary.txt"
    lines = [
        "=" * 60,
        "Step 3-B Summary",
        "=" * 60,
        "",
        "─ Input parameters ─",
        f"  length_csv   : {args.length_csv}",
        f"  n_cnt        : {args.n_cnt}",
        f"  drop_interval: {args.drop_interval} s",
        f"  h_film       : {args.h_film * 1e9:.1f} nm",
        f"  area_size    : {args.area_size * 1e6:.1f} µm",
        f"  seed_table   : {args.seed_table}",
        f"  seed_mc      : {args.seed_mc}",
        f"  U_macro      : {result['U_macro'] * 1e6:.2f} µm/s",
        f"  Q_pore       : {result['Q_pore']:.3e} m³/s",
        "",
        "─ Predicted vs Experimental ─",
        f"  G ratio predicted  : {g_ratio:.4f}",
        f"  G ratio experiment : {exp_value:.4f}  (15 min, 24 s/drop)",
        f"  Relative error     : {stats['g_ratio_relative_error_pct']:+.1f}%",
        "",
        "─ Statistics (all CNTs) ─",
        f"  <Pe>  mean   : {stats['Pe_mean']:.4f}",
        f"  <Pe>  median : {stats['Pe_median']:.4f}",
        f"  Pe    99th%  : {stats['Pe_p99']:.4f}",
        f"  <γ̇>  mean   : {stats['gamma_mean']:.1f} s⁻¹",
        f"  <γ̇>  median : {stats['gamma_median']:.1f} s⁻¹",
        f"  <L>   mean   : {stats['L_mean_nm']:.1f} nm",
        f"  <L>   median : {stats['L_median_nm']:.1f} nm",
        f"  <cos²θ> mean   : {stats['cos2_mean']:.4f}",
        f"  <cos²θ> median : {stats['cos2_median']:.4f}",
        "",
        "─ G ratio by z-layer (bottom→top) ─",
    ]
    z_bins = result["z_bins"]
    for k, gr in result["g_ratio_per_layer"].items():
        z_lo = z_bins[k] * 1e9
        z_hi = z_bins[k + 1] * 1e9
        lines.append(f"  Layer {k} ({z_lo:.1f}–{z_hi:.1f} nm): G = {gr:.4f}")
    lines += ["", "=" * 60]

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text + "\n")
    print(summary_text)

    # ── 図 1: z 層別 G ピーク比 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    layer_keys = sorted(result["g_ratio_per_layer"].keys())
    z_centers_nm = [(z_bins[k] + z_bins[k + 1]) / 2 * 1e9 for k in layer_keys]
    g_layers = [result["g_ratio_per_layer"][k] for k in layer_keys]
    ax.plot(z_centers_nm, g_layers, "o-", color="steelblue")
    ax.axhline(g_ratio, color="gray", linestyle="--", label=f"Overall G={g_ratio:.3f}")
    ax.axhline(exp_value, color="red", linestyle=":", label=f"Exp. G={exp_value:.2f}")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("G peak ratio")
    ax.set_title("Orientation vs z-layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "step3b_layer_breakdown.png", dpi=150)
    plt.close(fig)

    # ── 図 2: 長さ bin 別 G ピーク比 ─────────────────────────────────────────
    L_bins_nm = np.percentile(lengths_nm, [0, 20, 40, 60, 80, 100])
    L_bins = L_bins_nm * 1e-9
    g_L = []
    counts_L = []
    for j in range(len(L_bins) - 1):
        mask = (result["lengths"] >= L_bins[j]) & (result["lengths"] < L_bins[j + 1])
        if mask.sum() > 0:
            g_L.append(
                float(
                    (10 * cos2_arr[mask] + (1 - cos2_arr[mask])).dot(result["lengths"][mask])
                    / ((cos2_arr[mask] + 10 * (1 - cos2_arr[mask])).dot(result["lengths"][mask]))
                )
            )
        else:
            g_L.append(float("nan"))
        counts_L.append(int(mask.sum()))

    L_centers_nm = [(L_bins_nm[j] + L_bins_nm[j + 1]) / 2 for j in range(len(L_bins) - 1)]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(L_centers_nm, g_L, "o-", color="steelblue", label="G ratio")
    ax2.bar(L_centers_nm, counts_L, width=(L_bins_nm[1] - L_bins_nm[0]) * 0.8,
            alpha=0.3, color="orange", label="count")
    ax1.axhline(exp_value, color="red", linestyle=":", label=f"Exp. {exp_value:.2f}")
    ax1.set_xlabel("L [nm] (bin center)")
    ax1.set_ylabel("G peak ratio", color="steelblue")
    ax2.set_ylabel("Count", color="orange")
    ax1.set_title("Orientation vs CNT length")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "step3b_length_breakdown.png", dpi=150)
    plt.close(fig)

    # ── 図 3: Pe ヒストグラム ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    pe_plot = pe_vals[pe_vals > 0]
    ax.hist(pe_plot, bins=np.logspace(
        np.log10(max(pe_plot.min(), 1e-4)), np.log10(pe_plot.max()), 50
    ), color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xscale("log")
    for val, label, color in [
        (np.median(pe_plot), f"median={np.median(pe_plot):.3f}", "green"),
        (np.mean(pe_plot), f"mean={np.mean(pe_plot):.3f}", "blue"),
        (np.percentile(pe_plot, 99), f"99th%={np.percentile(pe_plot,99):.3f}", "red"),
    ]:
        ax.axvline(val, color=color, linestyle="--", label=label)
    ax.set_xlabel("Pe [-]")
    ax.set_ylabel("Count")
    ax.set_title("Pe distribution (all CNTs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "step3b_pe_distribution.png", dpi=150)
    plt.close(fig)

    # ── 図 4: <cos²θ>(z) 連続プロット ────────────────────────────────────────
    n_z_fine = 20
    z_bins_fine = np.linspace(0.0, args.h_film, n_z_fine + 1)
    z_centers_fine = (z_bins_fine[:-1] + z_bins_fine[1:]) / 2
    cos2_z = []
    for k in range(n_z_fine):
        mask = (
            (result["positions"][:, 2] >= z_bins_fine[k])
            & (result["positions"][:, 2] < z_bins_fine[k + 1])
        )
        if mask.sum() > 0:
            cos2_z.append(float(np.mean(cos2_arr[mask])))
        else:
            cos2_z.append(float("nan"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_centers_fine * 1e9, cos2_z, "o-", color="steelblue")
    ax.axhline(0.5, color="gray", linestyle=":", label="isotropic (0.5)")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("<cos²θ>")
    ax.set_title("<cos²θ> vs z (20 bins)")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "step3b_predicted_vs_z.png", dpi=150)
    plt.close(fig)

    print(f"\nOutput files:")
    print(f"  {results_dir / 'step3b_summary.txt'}")
    print(f"  {results_dir / 'step3b_layer_breakdown.png'}")
    print(f"  {results_dir / 'step3b_length_breakdown.png'}")
    print(f"  {results_dir / 'step3b_pe_distribution.png'}")
    print(f"  {results_dir / 'step3b_predicted_vs_z.png'}")


if __name__ == "__main__":
    main()
