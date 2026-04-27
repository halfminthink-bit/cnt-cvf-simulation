"""length_dist モジュールのユニットテスト."""

import numpy as np
import pytest

from cnt_cvf.length_dist import load_length_distribution, sample_lengths, sample_lognormal_lengths


def test_load_empty_csv(tmp_path):
    """空の CSV（ヘッダのみ）は ValueError を raise する."""
    f = tmp_path / "empty.csv"
    f.write_text("length_nm\n")
    with pytest.raises(ValueError, match="No valid length data"):
        load_length_distribution(str(f))


def test_load_missing_column(tmp_path):
    """length_nm 列が無い CSV は ValueError を raise する."""
    f = tmp_path / "bad.csv"
    f.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="'length_nm' column not found"):
        load_length_distribution(str(f))


def test_load_returns_meters(tmp_path):
    """length_nm 列 [nm] が m 単位に変換されること."""
    f = tmp_path / "data.csv"
    f.write_text("length_nm\n200\n400\n600\n")
    arr = load_length_distribution(str(f))
    np.testing.assert_allclose(arr, [200e-9, 400e-9, 600e-9])


def test_sample_empty_raises():
    """空の distribution は ValueError を raise する."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must not be empty"):
        sample_lengths(np.array([]), n=10, rng=rng)


def test_sample_nonpositive_n_raises():
    """n <= 0 は ValueError を raise する."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be positive"):
        sample_lengths(np.array([1e-7, 2e-7]), n=0, rng=rng)


def test_sample_empirical_cdf():
    """N が大きいとき、サンプルの分布が元の分布を再現する（2-sample KS 検定）."""
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(42)
    original = np.array([100e-9, 200e-9, 300e-9, 400e-9, 500e-9] * 100)
    samples = sample_lengths(original, n=10000, rng=rng)

    # 2-sample KS 検定（有意水準 1%）
    stat, pval = ks_2samp(samples, original)
    assert pval > 0.01, f"KS p-value too small: {pval:.4f}"


def test_sample_values_in_distribution():
    """サンプルは全て元の distribution の要素であること（置換サンプリング）."""
    rng = np.random.default_rng(7)
    original = np.array([1e-7, 2e-7, 3e-7])
    samples = sample_lengths(original, n=1000, rng=rng)
    assert set(samples).issubset(set(original))


# ── sample_lognormal_lengths のテスト ──────────────────────────────────────


def test_lognormal_invalid_median():
    """median_nm <= 0 は ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="median_nm"):
        sample_lognormal_lengths(-1.0, 0.6, 100, rng)


def test_lognormal_invalid_sigma():
    """sigma_log <= 0 は ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="sigma_log"):
        sample_lognormal_lengths(200.0, -0.5, 100, rng)


def test_lognormal_invalid_n():
    """n <= 0 は ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be positive"):
        sample_lognormal_lengths(200.0, 0.6, 0, rng)


def test_lognormal_all_above_min():
    """全サンプルが min_length_nm 以上（デフォルト 100 nm）であること."""
    rng = np.random.default_rng(42)
    arr = sample_lognormal_lengths(150.0, 0.6, 5000, rng)
    assert np.all(arr >= 100e-9), "Some samples below min_length_nm=100nm"


def test_lognormal_output_shape():
    """戻り値は shape (n,) の m 単位配列であること."""
    rng = np.random.default_rng(1)
    arr = sample_lognormal_lengths(200.0, 0.5, 300, rng)
    assert arr.shape == (300,)
    assert arr.dtype == float


def test_lognormal_median_approx():
    """大サンプル時、実測 median が指定 median_nm の ±30% 以内であること."""
    rng = np.random.default_rng(99)
    median_nm = 300.0
    arr = sample_lognormal_lengths(median_nm, 0.6, 20000, rng)
    measured_median_nm = float(np.median(arr) * 1e9)
    rel_err = abs(measured_median_nm - median_nm) / median_nm
    assert rel_err < 0.30, f"Median deviation too large: {measured_median_nm:.1f} nm vs {median_nm} nm"


def test_lognormal_units_are_meters():
    """戻り値は nm ではなく m 単位（典型的な値が 1e-7 オーダー）."""
    rng = np.random.default_rng(5)
    arr = sample_lognormal_lengths(200.0, 0.4, 100, rng)
    assert np.median(arr) < 1e-6, "Values look like nm instead of m"
