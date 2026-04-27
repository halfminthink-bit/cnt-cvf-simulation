"""2D Langevin 方程式による CNT 配向ダイナミクス（Step 2）.

支配方程式（2D、θ は z 軸からの傾き角）:

    dθ/dt = -γ̇ sin²(θ) + sqrt(2 D_r) ξ(t)

第 1 項は Jeffery 方程式（L/d → ∞ 極限）。
ξ(t) は白色ガウス雑音 <ξ(t)ξ(t')> = δ(t-t')。
"""

from __future__ import annotations

import math

import numpy as np

from cnt_cvf.config import BOLTZMANN, BROERSMA_GAMMA_CORRECTION


def rotational_drag(length: float, diameter: float, viscosity: float) -> float:
    """剛体棒の回転抵抗係数（Broersma 近似）を返す.

    .. math::

        \\zeta_r = \\frac{\\pi \\eta L^3}{3 (\\ln(L/d) - \\gamma_c)}

    Args:
        length: CNT 長さ L [m]。正値であること。
        diameter: CNT 直径 d [m]。正値であること。
        viscosity: 溶媒粘度 η [Pa·s]。正値であること。

    Returns:
        回転抵抗係数 ζ_r [N·m·s]。
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    if viscosity <= 0:
        raise ValueError(f"viscosity must be positive, got {viscosity}")

    log_term = math.log(length / diameter) - BROERSMA_GAMMA_CORRECTION
    return math.pi * viscosity * length**3 / (3.0 * log_term)


def rotational_diffusivity(
    length: float,
    diameter: float,
    viscosity: float,
    temperature: float,
) -> float:
    """回転拡散係数 D_r を返す.

    .. math::

        D_r = \\frac{k_B T}{\\zeta_r}
            = \\frac{3 k_B T (\\ln(L/d) - \\gamma_c)}{\\pi \\eta L^3}

    Args:
        length: CNT 長さ L [m]。正値であること。
        diameter: CNT 直径 d [m]。正値であること。
        viscosity: 溶媒粘度 η [Pa·s]。正値であること。
        temperature: 温度 T [K]。正値であること。

    Returns:
        回転拡散係数 D_r [rad²/s]。
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    zeta_r = rotational_drag(length, diameter, viscosity)
    return BOLTZMANN * temperature / zeta_r


def peclet_number(
    gamma_dot: float,
    length: float,
    diameter: float,
    viscosity: float,
    temperature: float,
) -> float:
    """回転 Péclet 数 Pe = γ̇ / D_r を返す.

    Pe >> 1: 対流（せん断）支配でCNTが配向する。
    Pe << 1: 回転拡散支配でCNTがランダム配向を保つ。

    Args:
        gamma_dot: せん断速度 γ̇ [s⁻¹]。正値であること。
        length: CNT 長さ L [m]。正値であること。
        diameter: CNT 直径 d [m]。正値であること。
        viscosity: 溶媒粘度 η [Pa·s]。正値であること。
        temperature: 温度 T [K]。正値であること。

    Returns:
        回転 Péclet 数 Pe [-]。
    """
    if gamma_dot < 0:
        raise ValueError(f"gamma_dot must be non-negative, got {gamma_dot}")

    D_r = rotational_diffusivity(length, diameter, viscosity, temperature)
    return gamma_dot / D_r


def simulate_single_cnt(
    length: float,
    diameter: float,
    viscosity: float,
    temperature: float,
    gamma_dot: float,
    t_max: float,
    dt: float,
    theta0: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """2D Langevin 方程式を Euler-Maruyama 法で数値積分し、θ(t) を返す.

    支配方程式:

    .. math::

        \\frac{d\\theta}{dt} = -\\dot{\\gamma} \\sin^2(\\theta) + \\sqrt{2 D_r}\\, \\xi(t)

    θ は z 軸（勾配方向）からの傾き角。

    Euler-Maruyama スキーム:

    .. math::

        \\theta_{n+1} = \\theta_n
            - \\dot{\\gamma} \\sin^2(\\theta_n)\\, dt
            + \\sqrt{2 D_r\\, dt}\\, N(0,1)

    Args:
        length: CNT 長さ L [m]。正値であること。
        diameter: CNT 直径 d [m]。正値であること。
        viscosity: 溶媒粘度 η [Pa·s]。正値であること。
        temperature: 温度 T [K]。正値であること。
        gamma_dot: せん断速度 γ̇ [s⁻¹]。非負であること。
        t_max: 積分総時間 [s]。正値であること。
        dt: タイムステップ [s]。正値、かつ γ̇·dt << 1 を満たすこと。
        theta0: 初期角度 θ₀ [rad]。
        rng: NumPy の乱数生成器 (np.random.Generator)。

    Returns:
        (t, theta): 時刻配列 t [s]（shape (N+1,)）と
                    角度配列 theta [rad]（shape (N+1,)）。
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if gamma_dot < 0:
        raise ValueError(f"gamma_dot must be non-negative, got {gamma_dot}")

    D_r = rotational_diffusivity(length, diameter, viscosity, temperature)
    n_steps = int(t_max / dt)

    theta = np.empty(n_steps + 1)
    theta[0] = theta0
    noise_std = math.sqrt(2.0 * D_r * dt)

    for i in range(n_steps):
        drift = -gamma_dot * math.sin(theta[i]) ** 2 * dt
        theta[i + 1] = theta[i] + drift + noise_std * rng.standard_normal()

    t = np.arange(n_steps + 1, dtype=float) * dt
    return t, theta
