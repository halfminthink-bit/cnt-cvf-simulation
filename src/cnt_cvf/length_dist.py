"""CNT 長さ分布の読み込みと empirical CDF サンプリング（Step 3-B/3-C）."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_length_distribution(csv_path: str) -> np.ndarray:
    """CSV から length_nm 列を読み込み、m 単位の配列を返す.

    Args:
        csv_path: CSV ファイルのパス。length_nm 列を含むこと。

    Returns:
        長さの 1-D 配列 [m]。shape (N,)。正値のみ含む。

    Raises:
        ValueError: CSV が空、または length_nm 列が存在しない場合。
    """
    df = pd.read_csv(csv_path)
    if "length_nm" not in df.columns:
        raise ValueError(f"'length_nm' column not found in {csv_path}")
    lengths_nm = df["length_nm"].dropna().values
    if len(lengths_nm) == 0:
        raise ValueError(f"No valid length data in {csv_path}")
    lengths_m = lengths_nm * 1e-9
    if np.any(lengths_m <= 0):
        raise ValueError("Non-positive length found in data")
    return lengths_m


def sample_lognormal_lengths(
    median_nm: float,
    sigma_log: float,
    n: int,
    rng: np.random.Generator,
    min_length_nm: float = 100.0,
) -> np.ndarray:
    """log-normal 分布から CNT 長さをサンプルする [m].

    .. math::

        L \\sim \\exp\\!\\bigl(\\mathcal{N}(\\ln\\mu_L,\\, \\sigma_{\\log})\\bigr)

    中央値 = exp(ln μ_L) = μ_L。min_length_nm 未満は棄却して再サンプル。

    Args:
        median_nm: 中央値 [nm]。正値であること。
        sigma_log: 対数標準偏差 [-]。正値であること。
        n: サンプル数。正値であること。
        rng: 乱数生成器。
        min_length_nm: 最小長さ [nm]。この値未満は棄却して再サンプル。正値であること。

    Returns:
        長さの 1-D 配列 [m]。shape (n,)。全要素 >= min_length_nm * 1e-9。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if median_nm <= 0:
        raise ValueError(f"median_nm must be positive, got {median_nm}")
    if sigma_log <= 0:
        raise ValueError(f"sigma_log must be positive, got {sigma_log}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if min_length_nm <= 0:
        raise ValueError(f"min_length_nm must be positive, got {min_length_nm}")

    mu_log = np.log(median_nm)  # log-normal の μ パラメータ
    result = np.empty(n)
    filled = 0
    while filled < n:
        remaining = n - filled
        batch = max(remaining * 2, 100)
        raw_nm = np.exp(rng.normal(mu_log, sigma_log, size=batch))
        valid = raw_nm[raw_nm >= min_length_nm]
        take = min(len(valid), remaining)
        result[filled : filled + take] = valid[:take] * 1e-9
        filled += take

    return result


def sample_lengths(
    distribution: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """empirical CDF からランダムサンプルを返す（リサンプリング）.

    Args:
        distribution: 元の長さ分布 [m]。shape (M,)。空でないこと。
        n: サンプル数。正値であること。
        rng: 乱数生成器。

    Returns:
        サンプルされた長さの配列 [m]。shape (n,)。

    Raises:
        ValueError: distribution が空、または n が非正の場合。
    """
    if len(distribution) == 0:
        raise ValueError("distribution must not be empty")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return rng.choice(distribution, size=n, replace=True)
