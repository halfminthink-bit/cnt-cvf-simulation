"""runner モジュールのユニットテスト."""

import math

import numpy as np
import pytest

from cnt_cvf.runner import (
    _compute_g_ratio,
    build_pe_to_costheta_table,
    run_monte_carlo,
)
from cnt_cvf.config import CNT_DIAMETER, TEMPERATURE, VISCOSITY


# ─── Pe テーブルの極限値テスト ───────────────────────────────────────────────

def _build_table(pe_values, n_realization=30, n_steps=20000):
    rng = np.random.default_rng(0)
    return build_pe_to_costheta_table(
        pe_values=pe_values,
        n_realization=n_realization,
        n_steps=n_steps,
        rng=rng,
    )


def test_pe_table_zero_limit():
    """Pe → 0 では <cos²θ> → 0.5（等方分布）."""
    table = _build_table(np.array([0.001]))
    cos2 = float(table["cos2"](0.001))
    assert abs(cos2 - 0.5) < 0.05, f"Expected ~0.5, got {cos2:.4f}"


def test_pe_table_large_pe():
    """Pe >> 1 では <cos²θ> > 0.8（配向が強まる）."""
    table = _build_table(np.array([100.0]), n_realization=30, n_steps=40000)
    cos2 = float(table["cos2"](100.0))
    assert cos2 > 0.8, f"Expected >0.8, got {cos2:.4f}"


def test_pe_table_monotone():
    """<cos²θ> は Pe に対してほぼ単調非減少（許容誤差 0.05 以内）."""
    pe_vals = np.logspace(-1, 2, 8)
    table = _build_table(pe_vals, n_realization=30, n_steps=30000)
    cos2_vals = table["cos2"](pe_vals)
    diffs = np.diff(cos2_vals)
    assert np.all(diffs >= -0.05), f"Non-monotone at: {diffs}"


# ─── G ピーク比の極限値テスト ────────────────────────────────────────────────

def test_g_ratio_isotropic():
    """<cos²θ>=0.5（等方）では G 比 = 1.0。"""
    lengths = np.ones(100) * 200e-9
    cos2 = np.full(100, 0.5)
    sin2 = np.full(100, 0.5)
    ratio = _compute_g_ratio(lengths, cos2, sin2)
    assert abs(ratio - 1.0) < 1e-10, f"Expected 1.0, got {ratio}"


def test_g_ratio_perfectly_aligned():
    """<cos²θ>=1（完全配向）では G 比 = 10.0。"""
    lengths = np.ones(100) * 200e-9
    cos2 = np.ones(100)
    sin2 = np.zeros(100)
    ratio = _compute_g_ratio(lengths, cos2, sin2)
    assert abs(ratio - 10.0) < 1e-10, f"Expected 10.0, got {ratio}"


# ─── run_monte_carlo の極限値テスト ─────────────────────────────────────────

def _minimal_pe_table():
    """Pe=0 に対応するテーブル（等方）を手動で作る。"""
    from scipy.interpolate import interp1d
    pe = np.array([0.0, 1.0])
    cos2_vals = np.array([0.5, 0.5])
    sin2_vals = np.array([0.5, 0.5])
    return {
        "cos2": interp1d(pe, cos2_vals, bounds_error=False, fill_value=(0.5, 0.5)),
        "sin2": interp1d(pe, sin2_vals, bounds_error=False, fill_value=(0.5, 0.5)),
        "pe_grid": pe,
        "cos2_raw": cos2_vals,
        "sin2_raw": sin2_vals,
    }


def test_mc_short_lengths_gives_ratio_one(tmp_path):
    """全 CNT が L < W=334 nm（短尺）の場合、G 比は 1.0 に近い.

    短尺 CNT は Pe が小さく等方的 → G 比 ≈ 1.0。
    ここでは等方テーブル（cos2=0.5）を使って厳密に 1.0 を確認。
    """
    csv = tmp_path / "short.csv"
    csv.write_text("length_nm\n" + "\n".join(["150"] * 200))

    rng = np.random.default_rng(1)
    result = run_monte_carlo(
        n_cnt=200,
        h_film=25e-9,
        area_size=5e-6,
        length_csv=str(csv),
        drop_interval=24.0,
        pe_table=_minimal_pe_table(),
        rng=rng,
    )
    assert abs(result["g_ratio"] - 1.0) < 1e-6, f"Expected 1.0, got {result['g_ratio']}"


def test_mc_zero_gamma_gives_ratio_one(tmp_path):
    """等方テーブル（全 Pe → cos2=0.5）では G 比 = 1.0。"""
    csv = tmp_path / "any.csv"
    csv.write_text("length_nm\n" + "\n".join(["300"] * 100))

    rng = np.random.default_rng(2)
    result = run_monte_carlo(
        n_cnt=100,
        h_film=25e-9,
        area_size=5e-6,
        length_csv=str(csv),
        drop_interval=24.0,
        pe_table=_minimal_pe_table(),
        rng=rng,
    )
    assert abs(result["g_ratio"] - 1.0) < 1e-6, f"Expected 1.0, got {result['g_ratio']}"


def test_mc_result_keys(tmp_path):
    """run_monte_carlo の戻り値が必要なキーを持つこと。"""
    csv = tmp_path / "data.csv"
    csv.write_text("length_nm\n" + "\n".join(["200"] * 50))

    rng = np.random.default_rng(3)
    result = run_monte_carlo(
        n_cnt=50,
        h_film=25e-9,
        area_size=5e-6,
        length_csv=str(csv),
        drop_interval=24.0,
        pe_table=_minimal_pe_table(),
        rng=rng,
    )
    for key in ("angles", "lengths", "positions", "gamma_dots", "pe_values", "g_ratio", "g_ratio_per_layer"):
        assert key in result, f"Missing key: {key}"
    assert result["positions"].shape == (50, 3)
    assert result["lengths"].shape == (50,)
