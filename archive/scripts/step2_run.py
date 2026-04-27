"""Step 2 実行スクリプト: 2D Langevin シミュレーションによる CNT 配向解析."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from cnt_cvf.config import (
    BOUNDARY_LAYER_THICKNESS,
    CNT_DIAMETER,
    CNT_LENGTH,
    TEMPERATURE,
    VISCOSITY,
)
from cnt_cvf.dynamics import (
    peclet_number,
    rotational_diffusivity,
    simulate_single_cnt,
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 実験条件パラメータ ─────────────────────────────────────────────────────────

L = CNT_LENGTH       # 230 nm
d = CNT_DIAMETER     # 0.78 nm
eta = VISCOSITY      # 1e-3 Pa·s
T = TEMPERATURE      # 298.15 K
delta = BOUNDARY_LAYER_THICKNESS  # 100 nm

U_24s = 6.6e-6       # 24 s/drop での平均流速 [m/s]
gamma_dot_exp = U_24s / delta  # せん断速度 γ̇ = U/δ [s⁻¹]

D_r = rotational_diffusivity(L, d, eta, T)
Pe_exp = peclet_number(gamma_dot_exp, L, d, eta, T)

# ── Pe 軸の設定 ────────────────────────────────────────────────────────────────

PE_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]


# ── 無次元シミュレーション（N 本ベクトル化）──────────────────────────────────────

def _sim_ensemble(
    Pe: float, N: int, t_star_max: float, rng: np.random.Generator
) -> float:
    """N 本のアンサンブルを無次元時間で走らせ、後半の <cos 2θ> を返す.

    dθ/dt* = -Pe sin²θ + sqrt(2) ξ*(t*)  を Euler-Maruyama で積分する。
    """
    # 安定性: Pe·dt* < 0.002、かつ sqrt(2*dt*) < 0.1 rad/step
    dt_star = min(0.002 / max(1.0, Pe), 0.005)
    n_steps = int(t_star_max / dt_star)
    noise_std = np.sqrt(2.0 * dt_star)

    theta = rng.uniform(-np.pi, np.pi, N)   # ランダム初期条件

    n_half = n_steps // 2
    # 前半: 平衡化（結果を捨てる）
    for _ in range(n_half):
        theta = theta - Pe * np.sin(theta) ** 2 * dt_star + noise_std * rng.standard_normal(N)

    # 後半: 統計収集
    cos2_sum = np.zeros(N)
    n_prod = n_steps - n_half
    for _ in range(n_prod):
        theta = theta - Pe * np.sin(theta) ** 2 * dt_star + noise_std * rng.standard_normal(N)
        cos2_sum += np.cos(2.0 * theta)

    return float(np.mean(cos2_sum / n_prod))


# ── 1. 実験条件での Pe 計算 ───────────────────────────────────────────────────

print("=" * 60)
print("Step 2: 2D Langevin シミュレーション")
print("=" * 60)
print()
print(f"実験条件パラメータ:")
print(f"  L    = {L*1e9:.1f} nm")
print(f"  d    = {d*1e9:.2f} nm")
print(f"  η    = {eta:.3e} Pa·s")
print(f"  T    = {T:.2f} K")
print(f"  U    = {U_24s*1e6:.1f} µm/s  (24 s/drop)")
print(f"  δ    = {delta*1e9:.0f} nm  (仮定値)")
print(f"  γ̇    = U/δ = {gamma_dot_exp:.1f} s⁻¹")
print(f"  D_r  = {D_r:.2f} rad²/s")
print(f"  τ_r  = 1/(2D_r) = {1/(2*D_r)*1e3:.4f} ms")
print(f"  Pe_exp = γ̇/D_r = {Pe_exp:.5f}")
print()
print("⚠️  疑問点（γ̇ と δ の仮定について）:")
print("   Pe_exp ≈ {:.4f} << 1 → 回転拡散が支配的。".format(Pe_exp))
print("   γ̇ = U/δ の境界層モデルでは、δ=100nm は「ポア入口付近の局所的な")
print("   流れ加速域」の推定。実際の膜近傍せん断場が不明なため、この値は")
print("   粗い仮定。Pe=1 になるには γ̇ ≈ {:.0f} s⁻¹ が必要。".format(D_r))
print()

# ── 2. θ(t) 軌跡プロット ──────────────────────────────────────────────────────

rng_traj = np.random.default_rng(0)
fig_traj, axes = plt.subplots(len(PE_VALUES), 1, figsize=(9, 10), sharex=False)

traj_results: dict[float, tuple[np.ndarray, np.ndarray]] = {}

for ax, Pe in zip(axes, PE_VALUES):
    gamma_dot = Pe * D_r
    # 安定性: Pe·dt* < 0.002
    dt_star = min(0.002 / max(1.0, Pe), 0.005)
    t_star_max = 5.0
    dt = dt_star / D_r
    t_max = t_star_max / D_r
    n_steps_traj = int(t_star_max / dt_star)

    t_arr, theta_arr = simulate_single_cnt(
        L, d, eta, T,
        gamma_dot=gamma_dot,
        t_max=t_max, dt=dt,
        theta0=np.pi / 4,
        rng=np.random.default_rng(int(rng_traj.integers(1_000_000))),
    )
    traj_results[Pe] = (t_arr, theta_arr)

    t_star = t_arr * D_r  # 無次元時間
    theta_deg = np.degrees(theta_arr % np.pi - np.pi / 2)  # [-90, 90] deg に折り畳み

    ax.plot(t_star, np.degrees(theta_arr), linewidth=0.8, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("θ [deg]", fontsize=8)
    ax.set_title(f"Pe = {Pe}", fontsize=9)
    ax.set_ylim(-200, 200)

axes[-1].set_xlabel("t* = D_r · t  (dimensionless time)", fontsize=10)
fig_traj.suptitle("2D Jeffery+Langevin: θ(t) trajectories (θ₀ = π/4)", fontsize=12)
fig_traj.tight_layout()
traj_path = RESULTS_DIR / "step2_theta_trajectories.png"
fig_traj.savefig(traj_path, dpi=150)
plt.close(fig_traj)
print(f"  PNG saved: {traj_path}")

# ── 3. 秩序パラメータ S = <cos 2θ> vs Pe ─────────────────────────────────────

N_ENS = 200
T_STAR_MAX = 5.0

rng_ens = np.random.default_rng(42)
order_params: dict[float, float] = {}

print(f"\n秩序パラメータ S = <cos 2θ>  (N={N_ENS} trajectories per Pe):")
print(f"{'Pe':>8}  {'S = <cos 2θ>':>14}  {'n_steps':>10}")
for Pe in PE_VALUES:
    S = _sim_ensemble(Pe, N=N_ENS, t_star_max=T_STAR_MAX, rng=rng_ens)
    order_params[Pe] = S
    dt_star = min(0.002 / max(1.0, Pe), 0.005)
    n_steps_disp = int(T_STAR_MAX / dt_star)
    print(f"{Pe:>8.1f}  {S:>14.6f}  {n_steps_disp:>10d}")

print()

# ── 4. S vs Pe プロット ───────────────────────────────────────────────────────

Pe_arr = np.array(PE_VALUES)
S_arr = np.array([order_params[p] for p in PE_VALUES])

fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
ax_s.semilogx(Pe_arr, S_arr, "o-", color="steelblue", linewidth=2, markersize=7,
               label=f"MC (N={N_ENS})")

# 実験条件の Pe を縦線
ax_s.axvline(x=Pe_exp, color="red", linestyle="--", linewidth=1.5,
             label=f"Pe_exp = {Pe_exp:.4f}  (U={U_24s*1e6:.1f}µm/s, δ={delta*1e9:.0f}nm)")

ax_s.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax_s.axhline(1, color="black", linestyle="-.", linewidth=0.8, label="S = 1 (perfect)")

ax_s.set_xlabel("Peclet number  Pe = γ̇ / D_r  [-]", fontsize=11)
ax_s.set_ylabel("Order parameter  S = <cos 2θ>  [-]", fontsize=11)
ax_s.set_title("Orientational order parameter vs Pe (2D Jeffery+Langevin)", fontsize=12)
ax_s.legend(fontsize=9)
ax_s.set_ylim(-0.2, 1.1)
ax_s.grid(True, which="both", alpha=0.3)

order_path = RESULTS_DIR / "step2_order_vs_Pe.png"
fig_s.tight_layout()
fig_s.savefig(order_path, dpi=150)
plt.close(fig_s)
print(f"  PNG saved: {order_path}")

# ── 5. summary.txt 出力 ──────────────────────────────────────────────────────

summary_path = RESULTS_DIR / "step2_summary.txt"
lines = [
    "=" * 60,
    "Step 2 Summary: 2D Jeffery+Langevin Simulation",
    "=" * 60,
    "",
    "--- Experimental Condition ---",
    f"  L        = {L*1e9:.2f} nm",
    f"  d        = {d*1e9:.3f} nm",
    f"  eta      = {eta:.3e} Pa*s",
    f"  T        = {T:.2f} K",
    f"  U (24s)  = {U_24s*1e6:.2f} um/s",
    f"  delta    = {delta*1e9:.1f} nm  (assumed boundary layer)",
    f"  gamma_dot= {gamma_dot_exp:.2f} s^-1",
    f"  D_r      = {D_r:.4f} rad^2/s",
    f"  tau_r    = {1/(2*D_r)*1e3:.4f} ms",
    f"  Pe_exp   = {Pe_exp:.6f}",
    "",
    "--- Orientation vs Pe (N=200 ensemble, t*=5) ---",
    f"  {'Pe':>10}  {'<cos 2θ>':>12}",
]
for Pe in PE_VALUES:
    lines.append(f"  {Pe:>10.1f}  {order_params[Pe]:>12.6f}")
lines += [
    "",
    "--- Notes ---",
    f"  Pe_exp << 1: diffusion dominates at experimental conditions.",
    f"  Pe = 1 requires gamma_dot = {D_r:.0f} s^-1  (much larger than {gamma_dot_exp:.1f} s^-1).",
    f"  The assumed delta = {delta*1e9:.0f} nm (boundary layer) is a rough estimate.",
    f"  If actual shear rate near pore entrance is much higher, Pe could reach ~1.",
]

summary_text = "\n".join(lines)
summary_path.write_text(summary_text)
print(f"  TXT saved: {summary_path}")
print()
print("=" * 60)
print("Step 2 完了")
print("=" * 60)
