"""Kawai式によるラマン G ピーク比（配向度）の計算（Step 1）."""

from __future__ import annotations

import numpy as np
from scipy import integrate


def theta_max(length: float, groove_width: float) -> float:
    """溝内で許容される CNT の最大傾き角 θ_max を返す.

    L > W のとき θ_max = arcsin(W / L)。
    L ≤ W のとき π/2（実質的な自由回転を示す番兵値として返す）。

    Args:
        length: CNT 長さ L [m]。正値であること。
        groove_width: 溝幅 W [m]。正値であること。

    Returns:
        θ_max [rad]、範囲 (0, π/2]。
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if groove_width <= 0:
        raise ValueError(f"groove_width must be positive, got {groove_width}")

    if length <= groove_width:
        return np.pi / 2
    return np.arcsin(groove_width / length)


def g_peak_ratio(length: float, groove_width: float) -> float:
    """Kawai 式で G ピーク比 I_∥ / I_⊥ を計算する.

    溝内の CNT 角度分布を一様分布 θ ∈ [-θ_max, θ_max] と仮定:

    .. math::

        \\frac{I_\\parallel}{I_\\perp}
        = \\frac{\\int_{-\\theta_{\\max}}^{\\theta_{\\max}}
                  (10\\cos^2\\theta + \\sin^2\\theta)\\,d\\theta}
                 {\\int_{-\\theta_{\\max}}^{\\theta_{\\max}}
                  (\\cos^2\\theta + 10\\sin^2\\theta)\\,d\\theta}

    L ≤ W のとき自由回転（配向なし）→ 1.0 を返す。

    係数 10 はラマン偏光依存項（軸方向 vs 直交方向の G band 強度比）。

    Args:
        length: CNT 長さ L [m]。正値であること。
        groove_width: 溝幅 W [m]。正値であること。

    Returns:
        I_∥ / I_⊥ [-]、範囲 [1.0, 10.0)。
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if groove_width <= 0:
        raise ValueError(f"groove_width must be positive, got {groove_width}")

    if length <= groove_width:
        return 1.0

    t_max = theta_max(length, groove_width)

    def integrand_parallel(theta: float) -> float:
        return 10.0 * np.cos(theta) ** 2 + np.sin(theta) ** 2

    def integrand_perp(theta: float) -> float:
        return np.cos(theta) ** 2 + 10.0 * np.sin(theta) ** 2

    i_parallel, _ = integrate.quad(integrand_parallel, -t_max, t_max)
    i_perp, _ = integrate.quad(integrand_perp, -t_max, t_max)

    return i_parallel / i_perp


def g_peak_ratio_grid(
    lengths: np.ndarray,
    groove_widths: np.ndarray,
) -> np.ndarray:
    """(L, W) グリッド全体の G ピーク比を返す.

    Args:
        lengths: CNT 長さの 1-D 配列 L [m]。shape (N,)。
        groove_widths: 溝幅の 1-D 配列 W [m]。shape (M,)。

    Returns:
        G ピーク比の 2-D 配列、shape (N, M)。
        行インデックスが lengths、列インデックスが groove_widths に対応。
    """
    result = np.empty((len(lengths), len(groove_widths)))
    for i, L in enumerate(lengths):
        for j, W in enumerate(groove_widths):
            result[i, j] = g_peak_ratio(L, W)
    return result
