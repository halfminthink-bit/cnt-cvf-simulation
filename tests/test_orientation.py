"""Kawai式 orientation.py のユニットテスト."""

import numpy as np
import pytest

from cnt_cvf.orientation import g_peak_ratio, theta_max


# ── theta_max の境界テスト ────────────────────────────────────────────────────

class TestThetaMax:
    def test_L_less_than_W_returns_pi_over_2(self) -> None:
        assert theta_max(100e-9, 200e-9) == pytest.approx(np.pi / 2)

    def test_L_equal_W_returns_pi_over_2(self) -> None:
        assert theta_max(334e-9, 334e-9) == pytest.approx(np.pi / 2)

    def test_L_greater_than_W(self) -> None:
        L, W = 500e-9, 334e-9
        expected = np.arcsin(W / L)
        assert theta_max(L, W) == pytest.approx(expected)

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError):
            theta_max(-1e-9, 334e-9)

    def test_invalid_width_raises(self) -> None:
        with pytest.raises(ValueError):
            theta_max(300e-9, 0.0)


# ── g_peak_ratio の物理的極限テスト ──────────────────────────────────────────

class TestGPeakRatio:
    def test_L_less_than_W_returns_1(self) -> None:
        """L < W → 自由回転 → 配向度 = 1.0（厳密）."""
        assert g_peak_ratio(100e-9, 334e-9) == pytest.approx(1.0)

    def test_L_equal_W_returns_1(self) -> None:
        """L = W → 境界 → 配向度 = 1.0."""
        assert g_peak_ratio(334e-9, 334e-9) == pytest.approx(1.0)

    def test_L_slightly_above_W_is_close_to_1(self) -> None:
        """L = 1.01 W → 配向度は 1 より大きく低配向域に留まる.

        arcsin(W/1.01W) ≈ 81.9° のため積分値は約 1.17 となる（1.01 未満にはならない）。
        Kawai 式の非線形性による正しい挙動。
        """
        ratio = g_peak_ratio(1.01 * 334e-9, 334e-9)
        assert 1.0 < ratio < 1.3

    def test_L_much_larger_than_W_approaches_10(self) -> None:
        """L >> W → θ_max → 0 → 配向度 → 10."""
        ratio = g_peak_ratio(1e6 * 334e-9, 334e-9)
        assert ratio == pytest.approx(10.0, rel=1e-4)

    def test_ratio_at_least_1(self) -> None:
        """I_∥/I_⊥ ≥ 1（θ_max ≤ π/2 のとき常に成立）."""
        for L_nm in [100, 200, 335, 500, 1000, 10000]:
            ratio = g_peak_ratio(L_nm * 1e-9, 334e-9)
            assert ratio >= 1.0 - 1e-12

    def test_monotone_increasing_with_L(self) -> None:
        """同じ W で L を増やすと配向度が単調増加."""
        W = 334e-9
        lengths = np.array([335, 400, 500, 700, 1000]) * 1e-9
        ratios = [g_peak_ratio(L, W) for L in lengths]
        for a, b in zip(ratios, ratios[1:]):
            assert b >= a

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError):
            g_peak_ratio(0.0, 334e-9)

    def test_invalid_width_raises(self) -> None:
        with pytest.raises(ValueError):
            g_peak_ratio(300e-9, -1e-9)

    def test_experimental_condition_W334_L230(self) -> None:
        """W=334nm, L=230nm → L < W なので配向度 = 1.0."""
        # CNT長さ中央値 230 nm < 溝幅 334 nm → Kawai式では配向なし
        ratio = g_peak_ratio(230e-9, 334e-9)
        assert ratio == pytest.approx(1.0)
