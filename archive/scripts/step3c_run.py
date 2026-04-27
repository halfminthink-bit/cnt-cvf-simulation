"""Step 3-C スイープ 1: log-normal 中央値 μ_L スイープ.

人工 log-normal 分布の中央値 μ_L を変化させて G ピーク比の依存性を調べる。
物理モデルは Step 3-B と同一。σ_log = 0.6 固定、N_cnt=10000、5 シード平均。
実測 3 サンプル（8/15/30 min CSV）の予測値も同じ MC で計算して重ねる。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnt_cvf.config import (
    CNT_DIAMETER,
    DROP_VOLUME,
    EFFECTIVE_AREA,
    PORE_DENSITY,
    TEMPERATURE,
    VISCOSITY,
)
from cnt_cvf.flow_field import (
    DEFAULT_CUTOFF,
    PORE_RADIUS,
    generate_pore_lattice,
    macroscopic_flow_rate,
    per_pore_flow_rate,
    strain_rate_batch,
)
from cnt_cvf.length_dist import (
    load_length_distribution,
    sample_lengths,
    sample_lognormal_lengths,
)
from cnt_cvf.dynamics import rotational_diffusivity
from cnt_cvf.runner import build_pe_to_costheta_table, _compute_g_ratio

# ── 実験設定（Step 3-B と同一） ──────────────────────────────────────────────
DROP_INTERVAL = 24.0        # s
H_FILM = 25e-9              # m
AREA_SIZE = 5e-6            # m
N_CNT = 10000
N_SEEDS = 5
SIGMA_LOG = 0.6             # log-normal の対数標準偏差
CNT_AREA_MARGIN = 1e-6      # m

# μ_L スイープ点 [nm]
MEDIAN_NM_SWEEP = [100, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000]

# 実測 CSV と実験値
CSV_SAMPLES = [
    {
        "label": "8min",
        "csv": str(PROJECT_ROOT / "data/length_distributions/8min_pos03_final_lengths.csv"),
        "exp_g": 1.78,
    },
    {
        "label": "15min",
        "csv": str(PROJECT_ROOT / "data/length_distributions/15min_pos01_final_lengths.csv"),
        "exp_g": 1.59,
    },
    {
        "label": "30min",
        "csv": str(PROJECT_ROOT / "data/length_distributions/30min_sup_final_lengths.csv"),
        "exp_g": 1.20,
    },
]

# μ_L = 500 nm の参考ライン（合成的に達成可能な上限）
ACHIEVABLE_LIMIT_NM = 500.0


def run_mc_from_lengths(
    lengths_m: np.ndarray,
    rng: np.random.Generator,
    pe_table: dict,
    drop_interval: float = DROP_INTERVAL,
    h_film: float = H_FILM,
    area_size: float = AREA_SIZE,
    n_cnt: int = N_CNT,
) -> dict:
    """事前サンプル済みの長さ配列を使って MC を 1 回走らせ、G 比等を返す.

    Args:
        lengths_m: 長さ配列 [m]. shape (n_cnt,).
        rng: 乱数生成器。
        pe_table: build_pe_to_costheta_table の戻り値。
        drop_interval: drop 間隔 [s]。
        h_film: CNT 存在高さ範囲 [m]。
        area_size: ポア生成領域の一辺 [m]。
        n_cnt: CNT 本数。

    Returns:
        dict with keys: g_ratio, pe_values, gamma_dots, lengths
    """
    Q_total = macroscopic_flow_rate(DROP_VOLUME, drop_interval)
    Q_pore = per_pore_flow_rate(Q_total, PORE_DENSITY, EFFECTIVE_AREA)
    U_macro = Q_total / EFFECTIVE_AREA

    pore_positions = generate_pore_lattice(PORE_DENSITY, area_size, mode="random", rng=rng)

    lo, hi = CNT_AREA_MARGIN, area_size - CNT_AREA_MARGIN
    xy = rng.uniform(lo, hi, size=(n_cnt, 2))
    z = rng.uniform(0.0, h_film, size=n_cnt)
    positions = np.column_stack([xy, z])

    gamma_dots = strain_rate_batch(
        positions, pore_positions, Q_pore,
        a=PORE_RADIUS, U_macro=U_macro, cutoff=DEFAULT_CUTOFF,
    )
    dr_arr = np.array([
        rotational_diffusivity(l, CNT_DIAMETER, VISCOSITY, TEMPERATURE)
        for l in lengths_m
    ])
    pe_values = gamma_dots / dr_arr

    cos2_arr = pe_table["cos2"](pe_values).astype(float)
    sin2_arr = 1.0 - cos2_arr
    g_ratio = _compute_g_ratio(lengths_m, cos2_arr, sin2_arr)

    return {
        "g_ratio": g_ratio,
        "pe_values": pe_values,
        "gamma_dots": gamma_dots,
        "lengths": lengths_m,
    }


def sweep_lognormal(pe_table: dict) -> list[dict]:
    """μ_L スイープを 5 シード平均で実行する.

    Returns:
        各 μ_L の dict リスト。keys: median_nm, sigma_log,
        predicted_g_mean, predicted_g_std, median_pe, mean_pe, n_seeds
    """
    records = []
    n_total = len(MEDIAN_NM_SWEEP)
    for idx, median_nm in enumerate(MEDIAN_NM_SWEEP):
        g_vals = []
        pe_medians = []
        pe_means = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + seed * 100)
            lengths_m = sample_lognormal_lengths(
                median_nm, SIGMA_LOG, N_CNT, rng, min_length_nm=100.0
            )
            rng_mc = np.random.default_rng(2000 + seed * 100)
            res = run_mc_from_lengths(lengths_m, rng_mc, pe_table)
            g_vals.append(res["g_ratio"])
            pe_medians.append(float(np.median(res["pe_values"])))
            pe_means.append(float(np.mean(res["pe_values"])))

        records.append({
            "median_nm": median_nm,
            "sigma_log": SIGMA_LOG,
            "predicted_g_mean": float(np.mean(g_vals)),
            "predicted_g_std": float(np.std(g_vals, ddof=1)),
            "median_pe": float(np.mean(pe_medians)),
            "mean_pe": float(np.mean(pe_means)),
            "n_seeds": N_SEEDS,
        })
        print(f"  [{idx+1}/{n_total}] μ_L={median_nm:5d} nm  G={np.mean(g_vals):.4f} ± {np.std(g_vals, ddof=1):.4f}")
    return records


def sweep_csv_samples(pe_table: dict) -> list[dict]:
    """実測 CSV サンプル 3 点を 5 シード平均で MC 計算する.

    Returns:
        各サンプルの dict リスト。keys: label, csv, exp_g,
        predicted_g_mean, predicted_g_std, csv_median_nm, sigma_log_empirical,
        median_pe, mean_pe
    """
    records = []
    for sample in CSV_SAMPLES:
        length_dist = load_length_distribution(sample["csv"])
        csv_median_nm = float(np.median(length_dist * 1e9))
        sigma_log_emp = float(np.std(np.log(length_dist * 1e9)))

        g_vals = []
        pe_medians = []
        pe_means = []
        for seed in range(N_SEEDS):
            rng_len = np.random.default_rng(3000 + seed * 100)
            lengths_m = sample_lengths(length_dist, N_CNT, rng_len)
            rng_mc = np.random.default_rng(4000 + seed * 100)
            res = run_mc_from_lengths(lengths_m, rng_mc, pe_table)
            g_vals.append(res["g_ratio"])
            pe_medians.append(float(np.median(res["pe_values"])))
            pe_means.append(float(np.mean(res["pe_values"])))

        rec = {
            "label": sample["label"],
            "csv": sample["csv"],
            "exp_g": sample["exp_g"],
            "predicted_g_mean": float(np.mean(g_vals)),
            "predicted_g_std": float(np.std(g_vals, ddof=1)),
            "csv_median_nm": csv_median_nm,
            "sigma_log_empirical": sigma_log_emp,
            "median_pe": float(np.mean(pe_medians)),
            "mean_pe": float(np.mean(pe_means)),
        }
        records.append(rec)
        print(f"  {sample['label']:6s}  median={csv_median_nm:.1f}nm  G_pred={np.mean(g_vals):.4f}  G_exp={sample['exp_g']:.2f}")
    return records


def save_csv(sweep_records: list[dict], results_dir: Path) -> None:
    df = pd.DataFrame(sweep_records)[
        ["median_nm", "sigma_log", "predicted_g_mean", "predicted_g_std",
         "median_pe", "mean_pe", "n_seeds"]
    ]
    df.to_csv(results_dir / "step3c_sweep_median.csv", index=False)


def save_plot(
    sweep_records: list[dict],
    csv_records: list[dict],
    results_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # 背景色：達成可能域（μ_L ≤ 500 nm）vs 参照域
    ax.axvspan(0, ACHIEVABLE_LIMIT_NM, alpha=0.07, color="steelblue", label=None)
    ax.axvspan(ACHIEVABLE_LIMIT_NM, 3000, alpha=0.04, color="gray", label=None)
    ax.axvline(ACHIEVABLE_LIMIT_NM, color="gray", linestyle="--", linewidth=1.2,
               label=f"μ_L = {ACHIEVABLE_LIMIT_NM:.0f} nm (achievable limit)")

    # 青線: log-normal スイープ
    medians = [r["median_nm"] for r in sweep_records]
    g_mean = np.array([r["predicted_g_mean"] for r in sweep_records])
    g_std = np.array([r["predicted_g_std"] for r in sweep_records])
    ax.errorbar(
        medians, g_mean, yerr=g_std,
        fmt="o-", color="steelblue", capsize=4, linewidth=1.8,
        label=f"log-normal sweep (σ_log={SIGMA_LOG})",
        zorder=3,
    )

    # 緑水平線: 実験値
    exp_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for i, sample in enumerate(CSV_SAMPLES):
        ax.axhline(
            sample["exp_g"], linestyle=":", linewidth=1.4,
            color=exp_colors[i],
            label=f"{sample['label']} exp G={sample['exp_g']:.2f}",
        )

    # 赤マーカー: 実測 CSV の MC 予測値（横軸は CSV の実測 median）
    for i, rec in enumerate(csv_records):
        ax.errorbar(
            rec["csv_median_nm"], rec["predicted_g_mean"],
            yerr=rec["predicted_g_std"],
            fmt="D", color=exp_colors[i], markersize=9,
            capsize=4, zorder=5,
            label=f"{rec['label']} pred G={rec['predicted_g_mean']:.3f}",
        )

    ax.set_xscale("log")
    ax.set_xlim(80, 2500)
    ax.set_ylim(bottom=1.0)
    ax.set_xlabel("Median CNT length μ_L [nm]", fontsize=12)
    ax.set_ylabel("Raman G peak ratio", fontsize=12)
    ax.set_title(
        f"Step 3-C: G ratio vs log-normal median  (σ_log={SIGMA_LOG}, N={N_CNT}, seeds={N_SEEDS})",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "step3c_sweep_median.png", dpi=150)
    plt.close(fig)


def save_summary(
    sweep_records: list[dict],
    csv_records: list[dict],
    results_dir: Path,
) -> str:
    g_means = [r["predicted_g_mean"] for r in sweep_records]
    best_idx = int(np.argmax(g_means))
    best_median = sweep_records[best_idx]["median_nm"]
    best_g = sweep_records[best_idx]["predicted_g_mean"]

    # σ_log の報告
    sigma_log_note = (
        f"  指定値 σ_log={SIGMA_LOG:.1f} に対して実測値は:"
        + "".join(
            f"\n    {r['label']}: σ_log={r['sigma_log_empirical']:.3f}  (median={r['csv_median_nm']:.1f} nm)"
            for r in csv_records
        )
        + f"\n  → 指定値は 8min ({csv_records[0]['sigma_log_empirical']:.3f}) に近く、"
          f"15min ({csv_records[1]['sigma_log_empirical']:.3f}) や "
          f"30min ({csv_records[2]['sigma_log_empirical']:.3f}) よりやや小さい。"
    )

    lines = [
        "=" * 68,
        "Step 3-C Summary: log-normal median sweep",
        "=" * 68,
        "",
        "─ 設定 ─",
        f"  σ_log (固定)   : {SIGMA_LOG}",
        f"  N_cnt          : {N_CNT}",
        f"  N_seeds        : {N_SEEDS}",
        f"  drop_interval  : {DROP_INTERVAL} s",
        f"  h_film         : {H_FILM*1e9:.0f} nm",
        f"  area_size      : {AREA_SIZE*1e6:.0f} µm",
        "",
        "─ σ_log の妥当性メモ ─",
        sigma_log_note,
        "",
        "─ log-normal スイープ結果 ─",
        f"  {'median_nm':>10}  {'G_mean':>8}  {'G_std':>7}  {'Pe_median':>10}",
        f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*10}",
    ]
    for r in sweep_records:
        lines.append(
            f"  {r['median_nm']:>10.0f}  {r['predicted_g_mean']:>8.4f}  "
            f"{r['predicted_g_std']:>7.4f}  {r['median_pe']:>10.4f}"
        )

    lines += [
        "",
        "─ 特定 μ_L での予測値 ─",
    ]
    for target in [200, 300, 500]:
        for r in sweep_records:
            if r["median_nm"] == target:
                lines.append(
                    f"  μ_L = {target:4d} nm:  G = {r['predicted_g_mean']:.4f} ± {r['predicted_g_std']:.4f}"
                )
    lines += [
        "",
        f"  最大 G: {best_g:.4f}  (μ_L = {best_median} nm)",
        f"  達成可能域 (μ_L ≤ {ACHIEVABLE_LIMIT_NM:.0f} nm) での最大値:",
    ]
    achievable = [r for r in sweep_records if r["median_nm"] <= ACHIEVABLE_LIMIT_NM]
    if achievable:
        best_ach = max(achievable, key=lambda r: r["predicted_g_mean"])
        lines.append(
            f"    G = {best_ach['predicted_g_mean']:.4f} ± {best_ach['predicted_g_std']:.4f}  "
            f"(μ_L = {best_ach['median_nm']} nm)"
        )
    lines += [
        "",
        "─ 実測 CSV の MC 予測値 ─",
        f"  {'label':>6}  {'csv_median':>11}  {'G_pred':>8}  {'G_std':>7}  {'G_exp':>6}  {'diff':>8}",
        f"  {'-'*6}  {'-'*11}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*8}",
    ]
    for r in csv_records:
        diff = r["predicted_g_mean"] - r["exp_g"]
        lines.append(
            f"  {r['label']:>6}  {r['csv_median_nm']:>10.1f}nm  "
            f"{r['predicted_g_mean']:>8.4f}  {r['predicted_g_std']:>7.4f}  "
            f"{r['exp_g']:>6.2f}  {diff:>+8.4f}"
        )
    lines += [
        "",
        "─ 観察事項 ─",
    ]

    # 単調増加かどうか判定
    g_arr = np.array(g_means)
    is_monotone = bool(np.all(np.diff(g_arr) >= 0))
    if is_monotone:
        lines.append("  * G ピーク比は μ_L に対してほぼ単調増加。明確な極大値なし。")
    else:
        max_idx_local = int(np.argmax(g_arr))
        lines.append(f"  * G ピーク比は μ_L = {sweep_records[max_idx_local]['median_nm']} nm 付近で極大値。")

    lines += [
        f"  * 達成可能域 (μ_L ≤ {ACHIEVABLE_LIMIT_NM:.0f} nm) 内: G_max ≈ {best_ach['predicted_g_mean']:.3f}",
        "  * 実測 CSV の予測値は理論曲線（log-normal スイープ）と比較できる。",
        "    → 乗る/外れる程度でモデルの説明能力を評価する。",
        "",
        "=" * 68,
    ]
    text = "\n".join(lines)
    (results_dir / "step3c_summary.txt").write_text(text + "\n")
    return text


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Step 3-C: log-normal median sweep")
    print("=" * 60)
    print(f"  σ_log={SIGMA_LOG}, N_cnt={N_CNT}, N_seeds={N_SEEDS}")
    print(f"  μ_L sweep: {MEDIAN_NM_SWEEP} nm")
    print()

    # ── Pe テーブル構築（1 回だけ、全スイープで共有） ─────────────────────────
    print("Building Pe → <cos²θ> table ...")
    pe_values = np.logspace(-2, 4, 50)
    rng_table = np.random.default_rng(0)
    pe_table = build_pe_to_costheta_table(
        pe_values=pe_values,
        n_realization=20,
        n_steps=20000,
        rng=rng_table,
    )
    print(f"  Done. Pe grid: [{pe_values[0]:.2e}, {pe_values[-1]:.2e}]")
    print()

    # ── log-normal スイープ ──────────────────────────────────────────────────
    print("Running log-normal sweep ...")
    sweep_records = sweep_lognormal(pe_table)
    print()

    # ── CSV サンプルの MC ────────────────────────────────────────────────────
    print("Running MC for CSV samples ...")
    csv_records = sweep_csv_samples(pe_table)
    print()

    # ── 出力 ────────────────────────────────────────────────────────────────
    save_csv(sweep_records, results_dir)
    save_plot(sweep_records, csv_records, results_dir)
    summary_text = save_summary(sweep_records, csv_records, results_dir)

    print(summary_text)
    print()
    print("Output files:")
    print(f"  {results_dir / 'step3c_sweep_median.csv'}")
    print(f"  {results_dir / 'step3c_sweep_median.png'}")
    print(f"  {results_dir / 'step3c_summary.txt'}")


if __name__ == "__main__":
    main()
