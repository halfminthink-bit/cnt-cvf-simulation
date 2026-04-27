"""モンテカルロ実行 + Pe → <cos²θ> テーブル構築（Step 3-B）."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import interp1d

from cnt_cvf.config import (
    BOLTZMANN,
    CNT_DIAMETER,
    EFFECTIVE_AREA,
    PORE_DENSITY,
    TEMPERATURE,
    VISCOSITY,
    DROP_VOLUME,
)
from cnt_cvf.dynamics import rotational_diffusivity, simulate_single_cnt
from cnt_cvf.flow_field import (
    generate_pore_lattice,
    macroscopic_flow_rate,
    per_pore_flow_rate,
    strain_rate_batch,
    DEFAULT_CUTOFF,
    PORE_RADIUS,
)
from cnt_cvf.length_dist import load_length_distribution, sample_lengths


def _steady_cos2theta(
    pe: float,
    dr: float,
    n_realization: int,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """単一 Pe での定常 <cos²θ> と <sin²θ> を Langevin シミュで推定する.

    タイムステップは Pe と Dr の両方に適応的に設定する:
        τ_eq = 1 / max(γ̇, Dr)  （平衡化の特徴的時間スケール）
        dt = 0.01 × τ_eq

    これにより高 Pe でも Euler-Maruyama が安定する。

    Args:
        pe: Péclet 数 [-]。
        dr: 回転拡散係数 [rad²/s]。正値であること。
        n_realization: 並走する軌跡本数。
        n_steps: 1 本あたりのステップ数（固定）。後半半分を平均に使う。
        rng: 乱数生成器。

    Returns:
        (mean_cos2, mean_sin2): 定常状態の <cos²θ>, <sin²θ>。
    """
    gamma_dot = pe * dr
    # 高 Pe では γ̇ が Dr を上回るので、γ̇ で時間スケールを決める
    tau_eq = 1.0 / max(gamma_dot, dr)
    dt = 0.01 * tau_eq
    half_idx = n_steps // 2

    cos2_vals = []
    sin2_vals = []
    noise_std = math.sqrt(2.0 * dr * dt)
    for _ in range(n_realization):
        theta = rng.uniform(-math.pi, math.pi)
        cos2_sum = 0.0
        sin2_sum = 0.0
        for step in range(n_steps):
            drift = -gamma_dot * math.sin(theta) ** 2 * dt
            theta = theta + drift + noise_std * rng.standard_normal()
            if step >= half_idx:
                cos2_sum += math.cos(theta) ** 2
                sin2_sum += math.sin(theta) ** 2
        n_half = n_steps - half_idx
        cos2_vals.append(cos2_sum / n_half)
        sin2_vals.append(sin2_sum / n_half)

    return float(np.mean(cos2_vals)), float(np.mean(sin2_vals))


def build_pe_to_costheta_table(
    pe_values: np.ndarray,
    n_realization: int = 20,
    n_steps: int = 20000,
    rng: np.random.Generator | None = None,
) -> dict[str, interp1d]:
    """Pe → <cos²θ>, <sin²θ> の interpolation テーブルを構築する.

    各 Pe グリッド点で n_realization 本の Langevin 軌跡を走らせ、
    後半半分の時間平均から定常値を推定する。

    タイムステップは Pe に適応的: dt = 0.01 / max(γ̇, Dr)。
    D_r = 1 の正規化系で計算（Pe = γ̇ / D_r = γ̇）。

    Args:
        pe_values: Pe グリッド点の 1-D 配列 [-]。正値のみ。shape (K,)。
        n_realization: 各 Pe での並走軌跡数。
        n_steps: 1 軌跡あたりの積分ステップ数（後半半分を平均に使う）。
        rng: 乱数生成器。None なら np.random.default_rng() を使用。

    Returns:
        dict with keys:
            'cos2': interp1d (Pe → <cos²θ>), fill_value extrapolated
            'sin2': interp1d (Pe → <sin²θ>), fill_value extrapolated
            'pe_grid': pe_values の配列
            'cos2_raw': 生の <cos²θ> 配列
            'sin2_raw': 生の <sin²θ> 配列
    """
    if rng is None:
        rng = np.random.default_rng()

    dr_ref = 1.0  # 正規化系

    cos2_table = np.empty(len(pe_values))
    sin2_table = np.empty(len(pe_values))

    for k, pe in enumerate(pe_values):
        c2, s2 = _steady_cos2theta(
            pe=float(pe),
            dr=dr_ref,
            n_realization=n_realization,
            n_steps=n_steps,
            rng=rng,
        )
        cos2_table[k] = c2
        sin2_table[k] = s2

    interp_cos2 = interp1d(
        pe_values, cos2_table, kind="linear",
        bounds_error=False,
        fill_value=(cos2_table[0], cos2_table[-1]),
    )
    interp_sin2 = interp1d(
        pe_values, sin2_table, kind="linear",
        bounds_error=False,
        fill_value=(sin2_table[0], sin2_table[-1]),
    )
    return {"cos2": interp_cos2, "sin2": interp_sin2, "pe_grid": pe_values,
            "cos2_raw": cos2_table, "sin2_raw": sin2_table}


def run_monte_carlo(
    n_cnt: int,
    h_film: float,
    area_size: float,
    drop_interval: float,
    pe_table: dict,
    rng: np.random.Generator,
    length_csv: str | None = None,
    lengths_m: np.ndarray | None = None,
    cnt_diameter: float = CNT_DIAMETER,
    viscosity: float = VISCOSITY,
    temperature: float = TEMPERATURE,
    pore_density: float = PORE_DENSITY,
    drop_volume: float = DROP_VOLUME,
    effective_area: float = EFFECTIVE_AREA,
    pore_radius: float = PORE_RADIUS,
    cutoff: float = DEFAULT_CUTOFF,
    cnt_area_margin: float = 1e-6,
    n_z_layers: int = 5,
) -> dict:
    """N 本の CNT を膜上に配置し、定常配向から G ピーク比を計算する.

    長さ分布は length_csv または lengths_m のどちらか一方を指定する。
    lengths_m が与えられた場合は CSV ロードをスキップして直接使用する。

    Args:
        n_cnt: CNT の本数。正値であること。
        h_film: CNT が存在する膜上の高さ範囲 [m]。z ∈ [0, h_film]。
        area_size: ポアを生成する正方形領域の一辺 [m]。
        drop_interval: drop 間隔 Δt_drop [s]。正値であること。
        pe_table: build_pe_to_costheta_table の戻り値 dict。
        rng: 乱数生成器（テーブル構築とは別シードを推奨）。
        length_csv: 長さ分布 CSV のパス。lengths_m と排他。
        lengths_m: 事前サンプル済みの長さ配列 [m]。shape (n_cnt,)。
            CSV ロードをスキップしたい場合（Step 3-C 等）に使用。
        cnt_diameter: CNT 直径 [m]。
        viscosity: 溶媒粘度 [Pa·s]。
        temperature: 温度 [K]。
        pore_density: ポア密度 [m⁻²]。
        drop_volume: 1 drop の体積 [m³]。
        effective_area: 有効面積 [m²]。
        pore_radius: 正則化半径 [m]。
        cutoff: γ̇ 計算のカットオフ距離 [m]。
        cnt_area_margin: CNT サンプリング領域の余白 [m]。
            CNT は [margin, area_size-margin]² からサンプル。
        n_z_layers: z 方向の層分割数。

    Returns:
        dict with keys:
            'angles':        shape (n_cnt,) - 各 CNT の <cos²θ> 定常値
            'lengths':       shape (n_cnt,) [m]
            'positions':     shape (n_cnt, 3) [m]
            'gamma_dots':    shape (n_cnt,) [s⁻¹]
            'pe_values':     shape (n_cnt,) [-]
            'g_ratio':       float - ラマン G ピーク比（長さ重みづけ）
            'g_ratio_per_layer': dict {layer_idx: g_ratio}
    """
    if n_cnt <= 0:
        raise ValueError(f"n_cnt must be positive, got {n_cnt}")
    if h_film <= 0:
        raise ValueError(f"h_film must be positive, got {h_film}")
    if drop_interval <= 0:
        raise ValueError(f"drop_interval must be positive, got {drop_interval}")
    if length_csv is None and lengths_m is None:
        raise ValueError("Either length_csv or lengths_m must be provided")

    # 流速の計算
    Q_total = macroscopic_flow_rate(drop_volume, drop_interval)
    Q_pore = per_pore_flow_rate(Q_total, pore_density, effective_area)
    U_macro = Q_total / effective_area

    # ポア配置（5µm × 5µm 全域）
    pore_positions = generate_pore_lattice(pore_density, area_size, mode="random", rng=rng)

    # 長さ分布のロード & サンプリング
    if lengths_m is not None:
        lengths = np.asarray(lengths_m, dtype=float)
        if len(lengths) != n_cnt:
            raise ValueError(
                f"lengths_m must have {n_cnt} elements, got {len(lengths)}"
            )
    else:
        length_dist = load_length_distribution(length_csv)  # type: ignore[arg-type]
        lengths = sample_lengths(length_dist, n_cnt, rng)

    # CNT 位置のサンプリング（中央の [margin, area_size-margin]² 内）
    lo = cnt_area_margin
    hi = area_size - cnt_area_margin
    if lo >= hi:
        lo, hi = 0.0, area_size
    xy = rng.uniform(lo, hi, size=(n_cnt, 2))
    z = rng.uniform(0.0, h_film, size=n_cnt)
    positions = np.column_stack([xy, z])

    # 各 CNT の γ̇ と Pe を計算（全 CNT をベクトル化して一括処理）
    gamma_dots = strain_rate_batch(
        positions,
        pore_positions,
        Q_pore,
        a=pore_radius,
        U_macro=U_macro,
        cutoff=cutoff,
    )
    dr_arr = np.array([
        rotational_diffusivity(l, cnt_diameter, viscosity, temperature)
        for l in lengths
    ])
    pe_values = gamma_dots / dr_arr

    # Pe → <cos²θ> の interpolation
    cos2_arr = pe_table["cos2"](pe_values).astype(float)
    sin2_arr = 1.0 - cos2_arr  # cos²θ + sin²θ = 1

    # G ピーク比（長さ重みづけ）
    g_ratio = _compute_g_ratio(lengths, cos2_arr, sin2_arr)

    # z 層別 G ピーク比
    z_bins = np.linspace(0.0, h_film, n_z_layers + 1)
    g_ratio_per_layer: dict[int, float] = {}
    for k in range(n_z_layers):
        mask = (positions[:, 2] >= z_bins[k]) & (positions[:, 2] < z_bins[k + 1])
        if mask.sum() > 0:
            g_ratio_per_layer[k] = _compute_g_ratio(
                lengths[mask], cos2_arr[mask], sin2_arr[mask]
            )
        else:
            g_ratio_per_layer[k] = float("nan")

    return {
        "angles": cos2_arr,
        "lengths": lengths,
        "positions": positions,
        "gamma_dots": gamma_dots,
        "pe_values": pe_values,
        "g_ratio": g_ratio,
        "g_ratio_per_layer": g_ratio_per_layer,
        "z_bins": z_bins,
        "Q_pore": Q_pore,
        "U_macro": U_macro,
    }


def _compute_g_ratio(
    lengths: np.ndarray,
    cos2: np.ndarray,
    sin2: np.ndarray,
) -> float:
    """長さ重みづけ G ピーク比を計算する.

    .. math::

        G = \\frac{\\sum_i L_i (10 \\cos^2\\theta_i + \\sin^2\\theta_i)}
                  {\\sum_i L_i (\\cos^2\\theta_i + 10 \\sin^2\\theta_i)}

    Args:
        lengths: shape (N,) [m]
        cos2: shape (N,) - <cos²θ>
        sin2: shape (N,) - <sin²θ>

    Returns:
        G ピーク比 [-]。
    """
    numerator = np.sum(lengths * (10.0 * cos2 + sin2))
    denominator = np.sum(lengths * (cos2 + 10.0 * sin2))
    return float(numerator / denominator)
