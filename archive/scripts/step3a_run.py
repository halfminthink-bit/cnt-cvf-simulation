"""Step 3-A: 流れ場モデルの可視化と sanity check.

実験条件（24 s/drop）で以下を出力する:
1. Q_total, Q_pore, N_pores の値
2. 単一ポア周辺の γ̇(r, z) ヒートマップ
3. 多ポア（正方格子、ピッチ 600 nm）での γ̇ ヒートマップ
4. γ̇ vs r の 1D プロット（複数 z）
5. z=10 nm 面の γ̇ ヒストグラム（ランダムポア配置）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# プロジェクトルートを sys.path に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from cnt_cvf.config import (
    BOLTZMANN,
    BROERSMA_GAMMA_CORRECTION,
    CNT_DIAMETER,
    CNT_LENGTH,
    DROP_VOLUME,
    EFFECTIVE_AREA,
    PORE_DENSITY,
    PORE_RADIUS,
    TEMPERATURE,
    VISCOSITY,
)
from cnt_cvf.flow_field import (
    generate_pore_lattice,
    macroscopic_flow_rate,
    per_pore_flow_rate,
    strain_rate,
    velocity_field,
)
from cnt_cvf.dynamics import rotational_diffusivity

RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DT_DROP = 24.0  # drop 間隔 [s]（実験条件）
A = PORE_RADIUS  # 50 nm


def compute_umacro(Q_total: float) -> float:
    return Q_total / EFFECTIVE_AREA


def build_summary(
    Q_total: float,
    Q_pore: float,
    N_pores: int,
    U_macro: float,
    gd_single_z50: float,
    gd_lattice_midpoint: float,
    hist_stats: dict,
) -> str:
    D_r = rotational_diffusivity(CNT_LENGTH, CNT_DIAMETER, VISCOSITY, TEMPERATURE)

    lines = [
        "=== Step 3-A Summary ===",
        "",
        f"実験条件: dt_drop = {DT_DROP} s/drop",
        f"Q_total  = {Q_total:.4e} m³/s",
        f"N_pores  = {N_pores:.3e}",
        f"Q_pore   = {Q_pore:.4e} m³/s",
        f"U_macro  = {U_macro*1e6:.4f} µm/s",
        "",
        "--- 単一ポア真上 z=50 nm ---",
        f"γ̇ = {gd_single_z50:.3e} s⁻¹",
        "",
        "--- 多ポア（正方格子 ピッチ 600 nm）ポア間中点 z=10 nm ---",
        f"γ̇ = {gd_lattice_midpoint:.3e} s⁻¹",
        "",
        "--- γ̇ ヒストグラム統計（z=10 nm、ランダム配置 N=10000 点） ---",
        f"中央値  : {hist_stats['median']:.3e} s⁻¹",
        f"平均    : {hist_stats['mean']:.3e} s⁻¹",
        f"std     : {hist_stats['std']:.3e} s⁻¹",
        f"上位 1% : {hist_stats['p99']:.3e} s⁻¹",
        "",
        "--- Pe 推定（L=230 nm の D_r = {:.3e} s⁻¹ で割る） ---".format(D_r),
        f"Pe（γ̇ 中央値）: {hist_stats['median'] / D_r:.4f}",
        f"Pe（γ̇ 平均）  : {hist_stats['mean'] / D_r:.4f}",
        f"Pe（γ̇ 99%）   : {hist_stats['p99'] / D_r:.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    # ── 基本量計算 ────────────────────────────────────────────────────────────
    Q_total = macroscopic_flow_rate(DROP_VOLUME, DT_DROP)
    Q_pore = per_pore_flow_rate(Q_total, PORE_DENSITY, EFFECTIVE_AREA)
    N_pores = int(PORE_DENSITY * EFFECTIVE_AREA)
    U_macro = compute_umacro(Q_total)

    print(f"Q_total = {Q_total:.4e} m³/s")
    print(f"N_pores = {N_pores:.3e}")
    print(f"Q_pore  = {Q_pore:.4e} m³/s")
    print(f"U_macro = {U_macro*1e6:.4f} µm/s")

    # ── 1. 単一ポアの γ̇ ヒートマップ ─────────────────────────────────────────
    single_pore = np.array([[0.0, 0.0, 0.0]])

    nx, nz = 80, 50
    r_range = np.linspace(-1000e-9, 1000e-9, nx)
    z_range = np.linspace(5e-9, 500e-9, nz)
    GD_single = np.zeros((nz, nx))

    for iz, z in enumerate(z_range):
        for ix, r in enumerate(r_range):
            x = np.array([r, 0.0, z])
            GD_single[iz, ix] = strain_rate(x, single_pore, Q_pore, A, U_macro=0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    r_nm = r_range * 1e9
    z_nm = z_range * 1e9
    im = ax.pcolormesh(
        r_nm, z_nm, GD_single,
        norm=mcolors.LogNorm(vmin=1e1, vmax=1e5),
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label=r"$\dot{\gamma}$ [s$^{-1}$]")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("z [nm]")
    ax.set_title("γ̇(r, z)  –  Single pore (point sink model)")
    plt.tight_layout()
    out_single = RESULTS_DIR / "step3a_gamma_single_pore.png"
    fig.savefig(out_single, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_single}")

    # ── 2. 多ポア（正方格子）の γ̇ ヒートマップ ──────────────────────────────
    PITCH = 600e-9
    area_lattice = 4500e-9  # 4.5 µm 程度の正方形領域を生成
    pore_lattice = generate_pore_lattice(PORE_DENSITY, area_lattice, mode="square")
    # 格子を中心に移動
    pore_lattice[:, :2] -= area_lattice / 2

    nx2, nz2 = 100, 50
    r_range2 = np.linspace(-1500e-9, 1500e-9, nx2)
    z_range2 = np.linspace(5e-9, 500e-9, nz2)
    GD_lattice = np.zeros((nz2, nx2))

    cutoff = 5 * PITCH
    for iz, z in enumerate(z_range2):
        for ix, r in enumerate(r_range2):
            x = np.array([r, 0.0, z])
            GD_lattice[iz, ix] = strain_rate(
                x, pore_lattice, Q_pore, A, U_macro=0.0, cutoff=cutoff
            )

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(
        r_range2 * 1e9, z_range2 * 1e9, GD_lattice,
        norm=mcolors.LogNorm(vmin=1e1, vmax=1e5),
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label=r"$\dot{\gamma}$ [s$^{-1}$]")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("z [nm]")
    ax.set_title(f"γ̇(x, z)  –  Square lattice (pitch={PITCH*1e9:.0f} nm, y=0 slice)")
    plt.tight_layout()
    out_lattice = RESULTS_DIR / "step3a_gamma_lattice.png"
    fig.savefig(out_lattice, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_lattice}")

    # ── 3. γ̇ vs r（1D、単一ポア、複数 z） ───────────────────────────────────
    r_1d = np.linspace(50e-9, 1500e-9, 60)
    z_levels = [10e-9, 50e-9, 100e-9]
    colors = ["tab:red", "tab:orange", "tab:blue"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for z_val, c in zip(z_levels, colors):
        gd_vals = []
        for r in r_1d:
            x = np.array([r, 0.0, z_val])
            gd_vals.append(strain_rate(x, single_pore, Q_pore, A, U_macro=0.0))
        ax.plot(r_1d * 1e9, gd_vals, color=c, label=f"z = {z_val*1e9:.0f} nm")

    ax.set_yscale("log")
    ax.set_xlabel("r from pore center [nm]")
    ax.set_ylabel(r"$\dot{\gamma}$ [s$^{-1}$]")
    ax.set_title("γ̇ vs r  –  Single pore")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_r = RESULTS_DIR / "step3a_gamma_vs_r.png"
    fig.savefig(out_r, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_r}")

    # ── 4. z=10 nm 面での γ̇ ヒストグラム（ランダム配置） ────────────────────
    rng = np.random.default_rng(42)
    area_rand = 10e-6  # 10 µm 角の領域にランダムにポアを配置
    pore_rand = generate_pore_lattice(PORE_DENSITY, area_rand, mode="random", rng=rng)
    pore_rand[:, :2] -= area_rand / 2  # 中心に移動

    N_sample = 10_000
    z_hist = 10e-9
    # サンプル点は中央 3 µm 角にランダム散布（端効果を避ける）
    xy_sample = rng.uniform(-1.5e-6, 1.5e-6, size=(N_sample, 2))
    gd_hist = np.zeros(N_sample)
    cutoff_hist = 5 * PITCH

    for i in range(N_sample):
        x = np.array([xy_sample[i, 0], xy_sample[i, 1], z_hist])
        gd_hist[i] = strain_rate(
            x, pore_rand, Q_pore, A, U_macro=0.0, cutoff=cutoff_hist
        )

    hist_stats = {
        "median": float(np.median(gd_hist)),
        "mean": float(np.mean(gd_hist)),
        "std": float(np.std(gd_hist)),
        "p99": float(np.percentile(gd_hist, 99)),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.logspace(np.log10(max(gd_hist.min(), 1.0)), np.log10(gd_hist.max()), 60)
    ax.hist(gd_hist, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(hist_stats["median"], color="red", linestyle="--", label=f"median = {hist_stats['median']:.2e}")
    ax.axvline(hist_stats["mean"], color="orange", linestyle="-.", label=f"mean = {hist_stats['mean']:.2e}")
    ax.axvline(hist_stats["p99"], color="purple", linestyle=":", label=f"99th pct = {hist_stats['p99']:.2e}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\dot{\gamma}$ [s$^{-1}$]")
    ax.set_ylabel("Count")
    ax.set_title(f"γ̇ distribution at z={z_hist*1e9:.0f} nm  (random pore layout, N={N_sample})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_hist = RESULTS_DIR / "step3a_gamma_histogram.png"
    fig.savefig(out_hist, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_hist}")

    # ── 単一ポア z=50 nm の γ̇ ────────────────────────────────────────────────
    gd_single_z50 = strain_rate(
        np.array([0.0, 0.0, 50e-9]), single_pore, Q_pore, A, U_macro=0.0
    )

    # ── 多ポア格子、ポア間中点（ピッチ 300 nm）z=10 nm ──────────────────────
    gd_lattice_midpoint = strain_rate(
        np.array([PITCH / 2, 0.0, 10e-9]), pore_lattice, Q_pore, A, U_macro=0.0,
        cutoff=cutoff
    )

    # ── summary.txt 出力 ──────────────────────────────────────────────────────
    summary = build_summary(
        Q_total, Q_pore, N_pores, U_macro,
        gd_single_z50, gd_lattice_midpoint, hist_stats
    )
    print("\n" + summary)
    out_txt = RESULTS_DIR / "step3a_summary.txt"
    out_txt.write_text(summary, encoding="utf-8")
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
