"""dynamics.py のユニットテスト（Step 2）."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnt_cvf.dynamics import (
    peclet_number,
    rotational_diffusivity,
    rotational_drag,
    simulate_single_cnt,
)


# ── rotational_drag ────────────────────────────────────────────────────────────

class TestRotationalDrag:
    def test_proportional_to_L_cubed(self) -> None:
        """ζ_r ∝ L³ を確認（L を 2 倍にすると ~8 倍、ln 項補正で実際は 7〜8）.

        ζ_r = π η L³ / (3(ln(L/d) - γ_c)) なので、L³ と ln(L/d) の両方が
        変化する。L=230nm → 460nm で ln 項の比は ≈ 0.88 のため
        実際の比は 8 × 0.88 ≈ 7.0 になる。
        """
        d, eta = 0.78e-9, 1e-3
        zeta1 = rotational_drag(230e-9, d, eta)
        zeta2 = rotational_drag(460e-9, d, eta)
        ratio = zeta2 / zeta1
        # L³ 純粋スケールなら 8、ln 補正で 6.5〜8.5 の範囲内
        assert 6.5 < ratio < 8.5

    def test_positive_for_valid_inputs(self) -> None:
        assert rotational_drag(230e-9, 0.78e-9, 1e-3) > 0

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError):
            rotational_drag(0.0, 0.78e-9, 1e-3)

    def test_invalid_diameter_raises(self) -> None:
        with pytest.raises(ValueError):
            rotational_drag(230e-9, -1e-9, 1e-3)

    def test_invalid_viscosity_raises(self) -> None:
        with pytest.raises(ValueError):
            rotational_drag(230e-9, 0.78e-9, 0.0)


# ── rotational_diffusivity ─────────────────────────────────────────────────────

class TestRotationalDiffusivity:
    def test_decreases_with_longer_CNT(self) -> None:
        """D_r ∝ 1/L³（長いほど小さい）."""
        d, eta, T = 0.78e-9, 1e-3, 298.15
        Dr_short = rotational_diffusivity(230e-9, d, eta, T)
        Dr_long = rotational_diffusivity(500e-9, d, eta, T)
        assert Dr_long < Dr_short

    def test_order_of_magnitude_for_230nm(self) -> None:
        """L=230 nm での D_r は ≈ 1000〜3000 rad²/s のオーダーであること."""
        Dr = rotational_diffusivity(230e-9, 0.78e-9, 1e-3, 298.15)
        assert 500.0 < Dr < 5000.0

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            rotational_diffusivity(230e-9, 0.78e-9, 1e-3, 0.0)


# ── peclet_number ──────────────────────────────────────────────────────────────

class TestPecletNumber:
    def test_numerical_value(self) -> None:
        """既知のパラメータで Pe を手計算と照合する.

        L=230nm, d=0.78nm, η=1e-3 Pa·s, T=298.15 K, γ̇=1624 s⁻¹:
        D_r ≈ 1624 rad²/s → Pe ≈ 1.0 前後（同オーダー確認）.
        """
        import cnt_cvf.config as cfg
        D_r = rotational_diffusivity(
            cfg.CNT_LENGTH, cfg.CNT_DIAMETER, cfg.VISCOSITY, cfg.TEMPERATURE
        )
        gamma_dot = D_r  # これで Pe = 1 になるはず
        Pe = peclet_number(gamma_dot, cfg.CNT_LENGTH, cfg.CNT_DIAMETER,
                           cfg.VISCOSITY, cfg.TEMPERATURE)
        assert Pe == pytest.approx(1.0, rel=1e-6)

    def test_zero_shear_gives_zero_pe(self) -> None:
        Pe = peclet_number(0.0, 230e-9, 0.78e-9, 1e-3, 298.15)
        assert Pe == 0.0

    def test_invalid_gamma_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            peclet_number(-1.0, 230e-9, 0.78e-9, 1e-3, 298.15)


# ── simulate_single_cnt ────────────────────────────────────────────────────────

class TestSimulateSingleCnt:
    """物理的に既知の極限で simulate_single_cnt を検証する."""

    # 共通パラメータ（L=230nm, d=0.78nm, η=1e-3, T=298.15 K）
    L, d, eta, T = 230e-9, 0.78e-9, 1e-3, 298.15

    def _make_rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_output_shapes(self) -> None:
        """t と theta の shape が一致すること."""
        Dr = rotational_diffusivity(self.L, self.d, self.eta, self.T)
        dt = 1e-4 / Dr
        t_max = 10 / Dr
        t, theta = simulate_single_cnt(
            self.L, self.d, self.eta, self.T,
            gamma_dot=0.0, t_max=t_max, dt=dt,
            theta0=0.0, rng=self._make_rng(),
        )
        assert t.shape == theta.shape
        assert t[0] == pytest.approx(0.0)
        assert abs(t[-1] - t_max) < 2 * dt

    def test_pe_zero_isotropic(self) -> None:
        """Pe = 0（γ̇ = 0）で長時間後の <cos 2θ> が 0 に近い（等方化）.

        N=100 本、t*=5 回転拡散時間、dt*=0.005（n_steps=1000）で確認。
        """
        Dr = rotational_diffusivity(self.L, self.d, self.eta, self.T)
        # dt* = 0.005 → dt = 0.005/D_r, t_max = 5/D_r → n_steps=1000
        dt = 5e-3 / Dr
        t_max = 5.0 / Dr
        rng = self._make_rng(seed=0)
        N = 100
        theta_finals = np.array([
            simulate_single_cnt(
                self.L, self.d, self.eta, self.T,
                gamma_dot=0.0, t_max=t_max, dt=dt,
                theta0=float(rng.uniform(-math.pi, math.pi)),
                rng=np.random.default_rng(int(rng.integers(1e9))),
            )[1][-1]
            for _ in range(N)
        ])
        order_param = float(np.mean(np.cos(2 * theta_finals)))
        assert abs(order_param) < 0.2   # 統計的許容誤差

    def test_large_pe_aligns_with_gradient(self) -> None:
        """Pe = 500 で θ が 0 近傍に収束する（決定論的項支配）.

        Pe*dt* = 0.001（安定）、t* = 0.5（n_steps = 250000、約 0.2 s）。
        <cos 2θ> は ~0.89 程度、> 0.8 を確認する。
        """
        Dr = rotational_diffusivity(self.L, self.d, self.eta, self.T)
        Pe = 500.0
        gamma_dot = Pe * Dr
        # dt* = 0.001/Pe → Pe*dt* = 0.001（安定性確保）
        dt = 0.001 / (Pe * Dr)   # = dt*/Dr
        t_max = 0.5 / Dr         # t* = 0.5
        _, theta = simulate_single_cnt(
            self.L, self.d, self.eta, self.T,
            gamma_dot=gamma_dot, t_max=t_max, dt=dt,
            theta0=math.pi / 4,
            rng=self._make_rng(),
        )
        cos2_final = float(np.mean(np.cos(2 * theta[-len(theta)//4:])))
        assert cos2_final > 0.8

    def test_invalid_t_max_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_single_cnt(self.L, self.d, self.eta, self.T,
                                gamma_dot=0.0, t_max=-1.0, dt=1e-7,
                                theta0=0.0, rng=self._make_rng())

    def test_invalid_gamma_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_single_cnt(self.L, self.d, self.eta, self.T,
                                gamma_dot=-1.0, t_max=1e-3, dt=1e-7,
                                theta0=0.0, rng=self._make_rng())
