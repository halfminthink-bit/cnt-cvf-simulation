"""Step 1 実行スクリプト: Kawai式による配向度テーブル・グラフを生成する."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src ディレクトリをパスに追加（スクリプト直接実行時）
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from cnt_cvf.orientation import g_peak_ratio, g_peak_ratio_grid

# ── パラメータ定義（SI 単位） ─────────────────────────────────────────────────

GROOVE_WIDTHS_NM = [200, 250, 300, 334, 400, 500]   # W [nm]
LENGTH_MIN_NM = 50
LENGTH_MAX_NM = 1000
LENGTH_STEP_NM = 50                                 # L グリッド [nm]

EXP_W_NM = 334.0    # 実験条件 溝幅 [nm]
EXP_L_NM = 230.0    # 実験条件 CNT長さ中央値 [nm]
EXP_RATIO = 1.78    # 実験値 G ピーク比 [-]

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── グリッド計算 ──────────────────────────────────────────────────────────────

groove_widths_m = np.array(GROOVE_WIDTHS_NM) * 1e-9
lengths_nm = np.arange(LENGTH_MIN_NM, LENGTH_MAX_NM + 1, LENGTH_STEP_NM)
lengths_m = lengths_nm * 1e-9

ratio_grid = g_peak_ratio_grid(lengths_m, groove_widths_m)

# ── CSV 出力 ──────────────────────────────────────────────────────────────────

csv_path = RESULTS_DIR / "step1_kawai_table.csv"
df = pd.DataFrame(
    ratio_grid,
    index=pd.Index(lengths_nm, name="L_nm"),
    columns=pd.Index(GROOVE_WIDTHS_NM, name="W_nm"),
)
df.to_csv(csv_path, float_format="%.6f")

# ── グラフ出力 ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

colors = plt.cm.tab10(np.linspace(0, 0.6, len(GROOVE_WIDTHS_NM)))

for j, (W_nm, color) in enumerate(zip(GROOVE_WIDTHS_NM, colors)):
    W_m = W_nm * 1e-9
    lw_ratio = lengths_m / W_m
    ax.plot(lw_ratio, ratio_grid[:, j], label=f"W = {W_nm} nm", color=color, linewidth=1.8)

# 補助線
ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.0, label="L/W = 1 (boundary)")
ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.0, label="Orientation = 1")
ax.axhline(y=10.0, color="black", linestyle="-.", linewidth=1.0, label="Orientation = 10 (asymptotic limit)")

# 実験値の水平線
ax.axhline(y=EXP_RATIO, color="red", linestyle="--", linewidth=1.0,
           alpha=0.6, label=f"Exp. value {EXP_RATIO}")

# 実験条件マーカー（W=334nm, L=230nm → L/W = 230/334 ≈ 0.689）
exp_lw = EXP_L_NM / EXP_W_NM
exp_orientation = g_peak_ratio(EXP_L_NM * 1e-9, EXP_W_NM * 1e-9)
ax.scatter(
    [exp_lw], [exp_orientation],
    s=80, color="red", zorder=5,
    label=f"Exp. cond. W={EXP_W_NM:.0f}nm, L={EXP_L_NM:.0f}nm\n(orientation={exp_orientation:.3f})",
)

ax.set_xscale("log")
ax.set_xlabel("L / W  [-]", fontsize=12)
ax.set_ylabel("Orientation  I_par / I_perp  [-]", fontsize=12)
ax.set_title("Kawai formula: CNT orientation vs L/W ratio", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(left=0.05)
ax.set_ylim(0.5, 10.5)
ax.grid(True, which="both", alpha=0.3)

png_path = RESULTS_DIR / "step1_kawai_orientation_vs_LW.png"
fig.tight_layout()
fig.savefig(png_path, dpi=150)
plt.close(fig)

# ── サマリー出力 ──────────────────────────────────────────────────────────────

print("=" * 60)
print("Step 1 完了: Kawai式による配向度テーブル・グラフを生成しました")
print(f"  CSV  : {csv_path}")
print(f"  PNG  : {png_path}")
print()

# 実験条件（W=334nm, L=230nm）の行を抜粋
print(f"実験条件 W=334nm, L=230nm:")
print(f"  L/W  = {EXP_L_NM / EXP_W_NM:.4f}")
print(f"  配向度 = {exp_orientation:.6f}  (L < W → 自由回転)")
print()

# CSVの抜粋（L_nm = 200〜300 付近、W=334nm 列）
w334_col = GROOVE_WIDTHS_NM.index(334)
print("CSVの抜粋 (W=334nm 列, L=200〜300nm):")
print(f"{'L [nm]':>10}  {'配向度':>10}")
for L_nm in [200, 230, 250, 300, 334, 335, 400, 500]:
    idx = int(round((L_nm - LENGTH_MIN_NM) / LENGTH_STEP_NM))
    if 0 <= idx < len(lengths_nm) and lengths_nm[idx] == L_nm:
        val = ratio_grid[idx, w334_col]
        marker = "  ← 実験条件" if L_nm == 230 else ""
        print(f"{L_nm:>10}  {val:>10.6f}{marker}")
    else:
        # グリッドにない値は直接計算
        val = g_peak_ratio(L_nm * 1e-9, 334e-9)
        marker = "  ← 実験条件" if L_nm == 230 else ""
        print(f"{L_nm:>10}  {val:>10.6f}{marker}")
print("=" * 60)
