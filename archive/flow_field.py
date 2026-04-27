"""ポア入口収束流の流れ場モデル（Step 3-A）.

座標系:
    - 膜面を z=0 とし、z>0 がバルク溶液側
    - 流れは z>0 から z=0 へ向かう（-z 方向）
    - ポアは z=0 の平面上に分布

点シンクモデル（半空間）:
    単一ポア i の作る速度場:

    .. math::

        \\mathbf{v}_i(\\mathbf{x}) = -\\frac{Q_{\\text{pore}}}{2\\pi}
            \\frac{\\mathbf{x} - \\mathbf{P}_i}{|\\mathbf{x} - \\mathbf{P}_i|^3}

    係数 2π は半空間（全空間なら 4π）。

ひずみ速度テンソルと局所せん断率:

    .. math::

        E_{ij} = \\frac{1}{2}\\left(\\frac{\\partial v_i}{\\partial x_j}
                + \\frac{\\partial v_j}{\\partial x_i}\\right)

        \\dot{\\gamma} = \\sqrt{2 \\sum_{ij} E_{ij}^2}
"""

from __future__ import annotations

import math

import numpy as np

from cnt_cvf.config import (
    DROP_VOLUME,
    EFFECTIVE_AREA,
    PORE_DENSITY,
    PORE_SIZE,
)

# ポア半径（正則化に使用）
PORE_RADIUS = PORE_SIZE / 2  # 50 nm [m]

# デフォルトのカットオフ距離 [m]（ポア間ピッチの 5 倍程度）
DEFAULT_CUTOFF = 3e-6  # 3 µm

# 数値微分のデフォルトステップ [m]
DEFAULT_H_FINITE_DIFF = 1e-9  # 1 nm


def macroscopic_flow_rate(V_drop: float, dt_drop: float) -> float:
    """マクロ流量 Q_total を返す.

    .. math::

        Q_{\\text{total}} = V_{\\text{drop}} / \\Delta t_{\\text{drop}}

    Args:
        V_drop: 1 drop の体積 [m³]。正値であること。
        dt_drop: drop 間隔 [s]。正値であること。

    Returns:
        Q_total: 全体流量 [m³/s]。
    """
    if V_drop <= 0:
        raise ValueError(f"V_drop must be positive, got {V_drop}")
    if dt_drop <= 0:
        raise ValueError(f"dt_drop must be positive, got {dt_drop}")
    return V_drop / dt_drop


def per_pore_flow_rate(
    Q_total: float, n_pore_density: float, A_eff: float
) -> float:
    """単一ポアあたりの流量 Q_pore を返す.

    .. math::

        Q_{\\text{pore}} = Q_{\\text{total}} / (n_{\\text{density}} \\times A_{\\text{eff}})

    Args:
        Q_total: 全体流量 [m³/s]。正値であること。
        n_pore_density: ポア密度 [m⁻²]。正値であること。
        A_eff: 有効面積 [m²]。正値であること。

    Returns:
        Q_pore: ポアあたりの流量 [m³/s]。
    """
    if Q_total <= 0:
        raise ValueError(f"Q_total must be positive, got {Q_total}")
    if n_pore_density <= 0:
        raise ValueError(f"n_pore_density must be positive, got {n_pore_density}")
    if A_eff <= 0:
        raise ValueError(f"A_eff must be positive, got {A_eff}")
    N_pores = n_pore_density * A_eff
    return Q_total / N_pores


def sink_velocity(
    x: np.ndarray,
    P: np.ndarray,
    Q_pore: float,
    a: float = PORE_RADIUS,
) -> np.ndarray:
    """単一ポアが作る速度ベクトルを返す（点シンク、半空間）.

    .. math::

        \\mathbf{v}_i(\\mathbf{x}) = -\\frac{Q_{\\text{pore}}}{2\\pi}
            \\frac{\\mathbf{x} - \\mathbf{P}_i}{R_{\\text{eff}}^3}

        R_{\\text{eff}} = \\max(|\\mathbf{x} - \\mathbf{P}_i|, a)

    Args:
        x: 観測点の座標 (3,) [m]。
        P: ポア位置の座標 (3,) [m]。z 成分は 0 を想定。
        Q_pore: ポアあたりの流量 [m³/s]。正値であること。
        a: 正則化半径（ポア半径）[m]。正値であること。

    Returns:
        速度ベクトル (3,) [m/s]。
    """
    if Q_pore <= 0:
        raise ValueError(f"Q_pore must be positive, got {Q_pore}")
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")

    dx = x - P
    r = float(np.linalg.norm(dx))
    R_eff = max(r, a)
    return -(Q_pore / (2.0 * math.pi)) * dx / R_eff**3


def velocity_field(
    x: np.ndarray,
    pore_positions: np.ndarray,
    Q_pore: float,
    a: float = PORE_RADIUS,
    U_macro: float = 0.0,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """全ポアの重ね合わせ + バルク均一流の速度ベクトルを返す.

    .. math::

        \\mathbf{v}(\\mathbf{x}) = -U_{\\text{macro}}\\,\\hat{z}
            + \\sum_{i} \\mathbf{v}_i(\\mathbf{x})

    バルク流は -z 方向（膜面へ向かう向き）。

    Args:
        x: 観測点の座標 (3,) [m]。
        pore_positions: ポア座標の配列 (N, 3) [m]。
        Q_pore: ポアあたりの流量 [m³/s]。正値であること。
        a: 正則化半径（ポア半径）[m]。正値であること。
        U_macro: マクロ流速の大きさ [m/s]。
        cutoff: 和を取るカットオフ距離 [m]。これより遠いポアは無視。

    Returns:
        速度ベクトル (3,) [m/s]。
    """
    # バルク流（-z 方向）
    v = np.array([0.0, 0.0, -U_macro])

    for P in pore_positions:
        dx = x - P
        r = float(np.linalg.norm(dx))
        if r < cutoff:
            v = v + sink_velocity(x, P, Q_pore, a)

    return v


def strain_rate(
    x: np.ndarray,
    pore_positions: np.ndarray,
    Q_pore: float,
    a: float = PORE_RADIUS,
    U_macro: float = 0.0,
    cutoff: float = DEFAULT_CUTOFF,
    h: float = DEFAULT_H_FINITE_DIFF,
) -> float:
    """局所せん断率 γ̇ を数値微分（中心差分）で計算して返す.

    ひずみ速度テンソル:

    .. math::

        E_{ij} = \\frac{1}{2}\\left(\\frac{\\partial v_i}{\\partial x_j}
                + \\frac{\\partial v_j}{\\partial x_i}\\right)

    局所せん断率:

    .. math::

        \\dot{\\gamma} = \\sqrt{2 \\sum_{ij} E_{ij}^2}

    Args:
        x: 観測点の座標 (3,) [m]。
        pore_positions: ポア座標の配列 (N, 3) [m]。
        Q_pore: ポアあたりの流量 [m³/s]。正値であること。
        a: 正則化半径（ポア半径）[m]。正値であること。
        U_macro: マクロ流速の大きさ [m/s]。
        cutoff: カットオフ距離 [m]。
        h: 数値微分のステップ幅 [m]。

    Returns:
        γ̇ [s⁻¹]。非負。
    """
    # 速度勾配テンソル ∂v_i/∂x_j を中心差分で計算
    grad_v = np.zeros((3, 3))  # grad_v[i, j] = ∂v_i/∂x_j
    for j in range(3):
        x_plus = x.copy()
        x_plus[j] += h
        x_minus = x.copy()
        x_minus[j] -= h
        v_plus = velocity_field(x_plus, pore_positions, Q_pore, a, U_macro, cutoff)
        v_minus = velocity_field(x_minus, pore_positions, Q_pore, a, U_macro, cutoff)
        grad_v[:, j] = (v_plus - v_minus) / (2.0 * h)

    # ひずみ速度テンソル E_ij = (∂v_i/∂x_j + ∂v_j/∂x_i) / 2
    E = 0.5 * (grad_v + grad_v.T)

    # γ̇ = sqrt(2 * E:E)
    gamma_dot = float(np.sqrt(2.0 * np.sum(E**2)))
    return gamma_dot


def generate_pore_lattice(
    n_density: float,
    area_size: float,
    mode: str = "random",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """指定密度でポア座標を生成する（z=0 固定）.

    Args:
        n_density: ポア密度 [m⁻²]。正値であること。
        area_size: 生成する正方形領域の一辺 [m]。正値であること。
        mode: 'random'（Poisson 配置）または 'square'（正方格子）。
        rng: 乱数生成器（mode='random' のとき使用）。
             None の場合は np.random.default_rng() を使用。

    Returns:
        ポア座標の配列 (N, 3) [m]。各行が (x, y, 0)。
    """
    if n_density <= 0:
        raise ValueError(f"n_density must be positive, got {n_density}")
    if area_size <= 0:
        raise ValueError(f"area_size must be positive, got {area_size}")
    if mode not in ("random", "square"):
        raise ValueError(f"mode must be 'random' or 'square', got {mode!r}")

    if mode == "square":
        pitch = 1.0 / math.sqrt(n_density)
        n_side = int(area_size / pitch)
        xs = np.arange(n_side) * pitch
        ys = np.arange(n_side) * pitch
        xx, yy = np.meshgrid(xs, ys)
        positions = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    else:
        if rng is None:
            rng = np.random.default_rng()
        N_expected = int(n_density * area_size**2)
        xy = rng.uniform(0.0, area_size, size=(N_expected, 2))
        positions = np.column_stack([xy, np.zeros(N_expected)])

    return positions


def velocity_field_batch(
    positions: np.ndarray,
    pore_positions: np.ndarray,
    Q_pore: float,
    a: float = PORE_RADIUS,
    U_macro: float = 0.0,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """複数の観測点に対して速度ベクトルを一括計算する（numpy ベクトル化版）.

    ``velocity_field`` と同じ物理モデル。Python ループを持たず、
    numpy broadcasting で M 点を同時に処理する。

    Args:
        positions: 観測点の座標配列 (M, 3) [m]。
        pore_positions: ポア座標の配列 (N, 3) [m]。
        Q_pore: ポアあたりの流量 [m³/s]。正値であること。
        a: 正則化半径 [m]。正値であること。
        U_macro: マクロ流速の大きさ [m/s]。
        cutoff: カットオフ距離 [m]。

    Returns:
        速度ベクトルの配列 (M, 3) [m/s]。
    """
    M = positions.shape[0]
    # dx[m, n, :] = positions[m] - pore_positions[n]
    dx = positions[:, None, :] - pore_positions[None, :, :]  # (M, N, 3)
    r = np.linalg.norm(dx, axis=2)  # (M, N)
    r_eff = np.maximum(r, a)  # (M, N) 正則化
    prefactor = -Q_pore / (2.0 * math.pi) / r_eff**3  # (M, N)
    v_sink = prefactor[:, :, None] * dx  # (M, N, 3)
    in_cutoff = (r < cutoff)[:, :, None]  # (M, N, 1)
    v = np.where(in_cutoff, v_sink, 0.0).sum(axis=1)  # (M, 3)
    v[:, 2] -= U_macro  # バルク流（-z 方向）
    return v


def strain_rate_batch(
    positions: np.ndarray,
    pore_positions: np.ndarray,
    Q_pore: float,
    a: float = PORE_RADIUS,
    U_macro: float = 0.0,
    cutoff: float = DEFAULT_CUTOFF,
    h: float = DEFAULT_H_FINITE_DIFF,
) -> np.ndarray:
    """複数の観測点に対して局所せん断率 γ̇ を一括計算する（numpy ベクトル化版）.

    ``strain_rate`` と同じ物理モデル（中心差分）を M 点に対して同時に適用する。

    Args:
        positions: 観測点の座標配列 (M, 3) [m]。
        pore_positions: ポア座標の配列 (N, 3) [m]。
        Q_pore: ポアあたりの流量 [m³/s]。
        a: 正則化半径 [m]。
        U_macro: マクロ流速の大きさ [m/s]。
        cutoff: カットオフ距離 [m]。
        h: 数値微分のステップ幅 [m]。

    Returns:
        γ̇ の配列 (M,) [s⁻¹]。非負。
    """
    M = positions.shape[0]
    grad_v = np.zeros((M, 3, 3))  # grad_v[m, i, j] = ∂v_i/∂x_j

    for j in range(3):
        pos_p = positions.copy()
        pos_p[:, j] += h
        pos_m = positions.copy()
        pos_m[:, j] -= h
        v_p = velocity_field_batch(pos_p, pore_positions, Q_pore, a, U_macro, cutoff)
        v_m = velocity_field_batch(pos_m, pore_positions, Q_pore, a, U_macro, cutoff)
        grad_v[:, :, j] = (v_p - v_m) / (2.0 * h)  # (M, 3)

    E = 0.5 * (grad_v + grad_v.transpose(0, 2, 1))  # (M, 3, 3)
    gamma_dot = np.sqrt(2.0 * np.sum(E**2, axis=(1, 2)))  # (M,)
    return gamma_dot
