"""
Tabular Beta distribution parameter estimation for the WS coefficient.

This script is based on the paper ``Nonparametric Significance Test of the Weighted Similarity Coefficient``
and covers the following stages:

1. Generating WS samples from random rankings.
2. Fitting Beta distribution parameters (alpha_n, beta_n) to the samples.
3. Saving the parameter table to a CSV file.

Examples
--------
Run from the command line:

.. code-block:: bash

    python main.py --n-min 3 --n-max 100 --samples 100000 --output ws_beta_params.csv

Notes
-----
Large values of ``--samples`` combined with a wide range of *n* will
significantly increase computation time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta, rankdata
from tqdm import tqdm


EPSILON = 1e-6


def compute_ws(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the WS similarity coefficient for two rankings.

    The WS coefficient measures the similarity between two rankings,
    giving higher weight to differences at the top of the ranking
    through an exponential decay factor :math:`2^{-x_i}`.

    Parameters
    ----------
    x : np.ndarray
        First ranking (1-D array of ranks).
    y : np.ndarray
        Second ranking (1-D array of ranks, same length as `x`).

    Returns
    -------
    float
        WS coefficient value in the range [0, 1], where 1 indicates
        identical rankings.

    Raises
    ------
    ValueError
        If `x` and `y` have different shapes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("Rankings x and y must have the same shape.")

    n_items = len(x)
    numerator = np.abs(x - y)
    denominator = np.maximum(np.abs(1 - x), np.abs(n_items - x))

    # For n=1 the denominator may contain zeros; the studied range is n >= 3.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            denominator == 0,
            0.0,
            np.power(2.0, -x) * numerator / denominator,
        )

    return float(1.0 - np.sum(terms))


def generate_single_ws(n: int, decimals: int = 2) -> float:
    """
    Generate a single WS value from two random rankings of length *n*.

    Random values are rounded to ``decimals`` decimal places before
    ranking, which increases the probability of ties.  Ties are resolved
    by ``rankdata(..., method="average")``.

    Parameters
    ----------
    n : int
        Number of items in each ranking.
    decimals : int, optional
        Number of decimal places used when rounding random values
        (default is 2).

    Returns
    -------
    float
        A single WS coefficient value.
    """
    random_values = np.round(np.random.rand(2, n), decimals)
    random_ranks = rankdata(random_values, axis=1, method="average")
    return compute_ws(random_ranks[0], random_ranks[1])


def generate_ws_samples(
    n: int,
    samples: int,
    decimals: int = 2,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Generate a Monte Carlo sample of WS values for a given ranking length.

    Parameters
    ----------
    n : int
        Number of items in each ranking.
    samples : int
        Number of Monte Carlo samples to draw.
    decimals : int, optional
        Decimal precision for rounding random values (default is 2).
    n_jobs : int, optional
        Number of parallel joblib workers.  ``-1`` uses all available
        CPU cores (default is -1).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(samples,)`` containing WS values.
    """
    values = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(generate_single_ws)(n, decimals) for _ in range(samples)
    )
    return np.asarray(values, dtype=float)


def fit_beta_parameters(
    ws_samples: np.ndarray,
    epsilon: float = EPSILON,
) -> Tuple[float, float, float, float]:
    """
    Fit a four-parameter Beta distribution to WS samples.

    Samples are clipped to the open interval ``(epsilon, 1 - epsilon)``
    before fitting to avoid boundary issues with the Beta PDF.

    Parameters
    ----------
    ws_samples : np.ndarray
        1-D array of WS coefficient values.
    epsilon : float, optional
        Small positive constant used for clipping (default is 1e-6).

    Returns
    -------
    alpha : float
        Shape parameter *alpha* of the fitted Beta distribution.
    beta_param : float
        Shape parameter *beta* of the fitted Beta distribution.
    loc : float
        Location parameter returned by ``scipy.stats.beta.fit``.
    scale : float
        Scale parameter returned by ``scipy.stats.beta.fit``.
    """
    clipped = np.clip(ws_samples, epsilon, 1.0 - epsilon)
    alpha, beta_param, loc, scale = beta.fit(clipped, floc = 0, fscale = 1)
    return float(alpha), float(beta_param), float(loc), float(scale)


def estimate_parameters_for_n_values(
    n_values: Iterable[int],
    samples: int,
    decimals: int = 2,
    n_jobs: int = -1,
    epsilon: float = EPSILON,
    include_loc_scale: bool = False,
) -> pd.DataFrame:
    """
    Estimate Beta distribution parameters for a sequence of ranking lengths.

    For each value of *n* in `n_values`, a Monte Carlo sample of WS
    values is generated and a Beta distribution is fitted.  The resulting
    parameters are collected into a ``DataFrame``.

    Parameters
    ----------
    n_values : Iterable[int]
        Ranking lengths to evaluate.
    samples : int
        Number of Monte Carlo samples per ranking length.
    decimals : int, optional
        Decimal precision for rounding random values (default is 2).
    n_jobs : int, optional
        Number of parallel joblib workers (default is -1, i.e. all cores).
    epsilon : float, optional
        Clipping constant for Beta fitting (default is 1e-6).
    include_loc_scale : bool, optional
        If ``True``, the returned ``DataFrame`` also contains the ``loc``
        and ``scale`` columns from ``scipy.stats.beta.fit``
        (default is ``False``).

    Returns
    -------
    pd.DataFrame
        Table with at least the columns ``n``, ``alpha``, ``beta``,
        ``a``, and ``b`` (the last two are aliases kept for backward
        compatibility with the original notebook).
    """
    rows = []

    for n in tqdm(list(n_values), desc="Estimating parameters"):
        try:
            ws_samples = generate_ws_samples(
                n=n,
                samples=samples,
                decimals=decimals,
                n_jobs=n_jobs,
            )
            alpha, beta_param, loc, scale = fit_beta_parameters(ws_samples, epsilon=epsilon)

            row = {
                "n": n,
                "alpha": alpha,
                "beta": beta_param,
                "a": alpha,
                "b": beta_param,
            }
            if include_loc_scale:
                row["loc"] = loc
                row["scale"] = scale
            rows.append(row)

        except Exception as exc:  # continue on error, as in the notebook
            print(f"Fitting failed for n={n}: {exc}")

    return pd.DataFrame(rows)


def save_parameters(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save the parameter table to a CSV file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Beta distribution parameters (as returned by
        :func:`estimate_parameters_for_n_values`).
    output_path : str or Path
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def get_alpha_beta_from_table(
    n: int,
    table: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Look up the Beta shape parameters for a given ranking length.

    Parameters
    ----------
    n : int
        Ranking length to query.
    table : pd.DataFrame
        Parameter table (must contain a column ``n`` and either
        ``alpha``/``beta`` or ``a``/``b``).

    Returns
    -------
    alpha : float
        Shape parameter *alpha_n*.
    beta : float
        Shape parameter *beta_n*.

    Raises
    ------
    ValueError
        If no row with the requested *n* exists in `table`.
    """
    row = table.loc[table["n"] == n]
    if row.empty:
        raise ValueError(f"No parameters found for n={n} in the table.")

    alpha_col = "alpha" if "alpha" in table.columns else "a"
    beta_col = "beta" if "beta" in table.columns else "b"
    return float(row.iloc[0][alpha_col]), float(row.iloc[0][beta_col])


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes ``n_min``, ``n_max``,
        ``samples``, ``decimals``, ``n_jobs``, ``output``, and
        ``include_loc_scale``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate tabular Beta distribution parameters "
            "(alpha_n, beta_n) for WS coefficient values."
        ),
    )
    parser.add_argument(
        "--n-min", type=int, default=3,
        help="Smallest ranking length n (default: 3).",
    )
    parser.add_argument(
        "--n-max", type=int, default=100,
        help="Largest ranking length n, inclusive (default: 100).",
    )
    parser.add_argument(
        "--samples", type=int, default=100000,
        help="Number of Monte Carlo samples per ranking length (default: 100000).",
    )
    parser.add_argument(
        "--decimals", type=int, default=2,
        help="Decimal places when rounding random values (default: 2).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of joblib workers; -1 means all cores (default: -1).",
    )
    parser.add_argument(
        "--output", type=str, default="ws_beta_parameters.csv",
        help="Output CSV file path (default: ws_beta_parameters.csv).",
    )
    parser.add_argument(
        "--include-loc-scale", action="store_true",
        help="Also store loc and scale parameters from beta.fit.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for command-line execution.

    Parses arguments, validates inputs, runs the parameter estimation
    pipeline, and writes results to the specified CSV file.

    Raises
    ------
    ValueError
        If any of the numeric arguments are out of range.
    """
    args = parse_args()

    if args.n_min < 2:
        raise ValueError("--n-min must be at least 2.")
    if args.n_max < args.n_min:
        raise ValueError("--n-max must be greater than or equal to --n-min.")
    if args.samples <= 0:
        raise ValueError("--samples must be a positive integer.")

    n_values = range(args.n_min, args.n_max + 1)

    df = estimate_parameters_for_n_values(
        n_values=n_values,
        samples=args.samples,
        decimals=args.decimals,
        n_jobs=args.n_jobs,
        include_loc_scale=args.include_loc_scale,
    )

    save_parameters(df, args.output)
    print(f"Results saved to: {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()