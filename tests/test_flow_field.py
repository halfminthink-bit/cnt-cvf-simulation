"""flow_field.py の物理的極限値テスト（Step 3-A）."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnt_cvf.config import (
    DROP_VOLUME,
    EFFECTIVE_AREA,
    PORE_DENSITY,
    PORE_RADIUS,
)
from cnt_cvf.flow_field import (
    DEFAULT_CUTOFF,
    macroscopic_flow_rate,
    per_pore_flow_rate,
    sink_velocity,
    strain_rate,
    velocity_field,
    generate_pore_lattice,
)

# ── テスト用の固定パラメータ ──────────────────────────────────────────────────

DT_DROP = 24.0  # drop 間隔 [s]
Q_TOTAL = macroscopic_flow_rate(DROP_VOLUME, DT_DROP)
Q_PORE = per_pore_flow_rate(Q_TOTAL, PORE_DENSITY, EFFECTIVE_AREA)
A = PORE_RADIUS

SINGLE_PORE = np.array([[0.0, 0.0, 0.0]])  # 原点に 1 つのポア


# ── 流量保存テスト ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("R_hemi", [200e-9, 500e-9, 1000e-9])
def test_flux_conservation_hemisphere(R_hemi: float) -> None:
    """単一ポア真上の半球面で v·n を積分すると Q_pore に等しくなる.

    半球（z>0 側）の表面積分を Monte Carlo で評価する。
    """
    rng = np.random.default_rng(42)
    N_mc = 50_000

    # 半球面上の一様サンプリング（z>0）
    phi = rng.uniform(0.0, 2.0 * math.pi, N_mc)
    cos_theta = rng.uniform(0.0, 1.0, N_mc)  # z>0 なので cos_theta ∈ [0,1]
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    xs = R_hemi * sin_theta * np.cos(phi)
    ys = R_hemi * sin_theta * np.sin(phi)
    zs = R_hemi * cos_theta

    # 各点で速度を計算し、外向き法線（半径方向）との内積を取る
    flux_sum = 0.0
    for i in range(N_mc):
        xi = np.array([xs[i], ys[i], zs[i]])
        n_hat = xi / R_hemi  # 外向き単位法線
        v = sink_velocity(xi, SINGLE_PORE[0], Q_PORE, A)
        flux_sum += float(np.dot(v, n_hat))

    # 半球面積で正規化（MC 積分）
    S_hemi = 2.0 * math.pi * R_hemi**2
    flux = flux_sum / N_mc * S_hemi

    # 点シンクの流量は Q_pore に等しい（符号は内向き = 負）
    # v·n の積分は -Q_pore になるはず
    assert abs(flux + Q_PORE) / Q_PORE < 0.05, (
        f"Flux conservation failed at R={R_hemi*1e9:.0f} nm: "
        f"integral={flux:.3e}, -Q_pore={-Q_PORE:.3e}"
    )


def test_velocity_far_field_converges_to_macro() -> None:
    """十分遠方ではバルク流 U_macro に収束する."""
    U_macro = Q_TOTAL / EFFECTIVE_AREA
    x_far = np.array([0.0, 0.0, 1e-3])  # 1 mm（ポア間ピッチの ~1600 倍）
    v = velocity_field(x_far, SINGLE_PORE, Q_PORE, A, U_macro, cutoff=DEFAULT_CUTOFF)
    # ポア寄与は無視できるほど小さいはず（cutoff 外なのでゼロ）
    assert abs(v[0]) < 1e-12, f"v_x should be ~0 far away, got {v[0]}"
    assert abs(v[1]) < 1e-12, f"v_y should be ~0 far away, got {v[1]}"
    assert abs(v[2] + U_macro) / U_macro < 1e-6, (
        f"v_z should equal -U_macro far away, got {v[2]}, expected {-U_macro}"
    )


def test_symmetry_above_single_pore() -> None:
    """単一ポア正上方では v_x = v_y = 0、v_z < 0 になる."""
    x_above = np.array([0.0, 0.0, 200e-9])
    v = velocity_field(x_above, SINGLE_PORE, Q_PORE, A, U_macro=0.0)
    assert abs(v[0]) < 1e-15, f"v_x should be 0 above pore, got {v[0]}"
    assert abs(v[1]) < 1e-15, f"v_y should be 0 above pore, got {v[1]}"
    assert v[2] < 0, f"v_z should be negative (downward) above pore, got {v[2]}"


def test_gamma_dot_decreases_with_r() -> None:
    """z=10 nm 固定でポア中心から遠ざかると γ̇ が単調減少する."""
    z = 10e-9
    r_values = [60e-9, 100e-9, 200e-9, 400e-9, 800e-9]
    gamma_dots = []
    for r in r_values:
        x = np.array([r, 0.0, z])
        gd = strain_rate(x, SINGLE_PORE, Q_PORE, A, U_macro=0.0)
        gamma_dots.append(gd)

    for i in range(len(gamma_dots) - 1):
        assert gamma_dots[i] > gamma_dots[i + 1], (
            f"γ̇ not monotonically decreasing: "
            f"r={r_values[i]*1e9:.0f} nm → {gamma_dots[i]:.2e}, "
            f"r={r_values[i+1]*1e9:.0f} nm → {gamma_dots[i+1]:.2e}"
        )


def test_gamma_dot_scaling_1_over_r3() -> None:
    """十分遠方では γ̇ ∝ 1/R³ のスケーリングになる（単一ポア）.

    R が 2 倍になると γ̇ が 1/8 になることを確認。
    """
    z1, z2 = 500e-9, 1000e-9
    x1 = np.array([0.0, 0.0, z1])
    x2 = np.array([0.0, 0.0, z2])
    gd1 = strain_rate(x1, SINGLE_PORE, Q_PORE, A, U_macro=0.0)
    gd2 = strain_rate(x2, SINGLE_PORE, Q_PORE, A, U_macro=0.0)
    ratio = gd1 / gd2
    expected_ratio = (z2 / z1) ** 3  # = 8.0
    assert abs(ratio - expected_ratio) / expected_ratio < 0.10, (
        f"1/R³ scaling failed: ratio={ratio:.3f}, expected={expected_ratio:.3f}"
    )


def test_gamma_dot_magnitude_near_pore() -> None:
    """単一ポア真上 z=ポア半径 での γ̇ が 10³〜10⁴ s⁻¹ のオーダーにある."""
    x = np.array([0.0, 0.0, A])  # z = 50 nm = ポア半径
    gd = strain_rate(x, SINGLE_PORE, Q_PORE, A, U_macro=0.0)
    assert 1e2 < gd < 1e6, (
        f"γ̇ at z=a should be 10³~10⁴ s⁻¹, got {gd:.2e}"
    )


def test_generate_pore_lattice_square_density() -> None:
    """正方格子モードで生成したポア数が期待値に近い."""
    area = 5e-6  # 5 µm
    positions = generate_pore_lattice(PORE_DENSITY, area, mode="square")
    expected_N = int(PORE_DENSITY * area**2)
    # 格子は int(area/pitch) × int(area/pitch) なので多少ずれる
    assert abs(len(positions) - expected_N) / expected_N < 0.20, (
        f"Square lattice pore count mismatch: {len(positions)} vs {expected_N}"
    )
    # z 座標はすべて 0
    assert np.all(positions[:, 2] == 0.0)


def test_generate_pore_lattice_random() -> None:
    """ランダムモードで生成したポア数と z 座標を確認."""
    rng = np.random.default_rng(0)
    area = 5e-6
    positions = generate_pore_lattice(PORE_DENSITY, area, mode="random", rng=rng)
    expected_N = int(PORE_DENSITY * area**2)
    assert len(positions) == expected_N
    assert np.all(positions[:, 2] == 0.0)


def test_invalid_inputs_raise() -> None:
    """物理的に不正な入力で ValueError が上がる."""
    with pytest.raises(ValueError):
        macroscopic_flow_rate(-1.0, 24.0)
    with pytest.raises(ValueError):
        macroscopic_flow_rate(DROP_VOLUME, 0.0)
    with pytest.raises(ValueError):
        per_pore_flow_rate(Q_TOTAL, -1.0, EFFECTIVE_AREA)
    with pytest.raises(ValueError):
        sink_velocity(np.zeros(3), np.zeros(3), Q_PORE, a=-1.0)
    with pytest.raises(ValueError):
        generate_pore_lattice(-1.0, 5e-6)
    with pytest.raises(ValueError):
        generate_pore_lattice(PORE_DENSITY, 5e-6, mode="invalid")
