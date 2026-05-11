# ws-ssd-test

Companion code for the paper:

> **Nonparametric Significance Test of the Weighted Similarity Coefficient**
> Wojciech Sałabun, Andrii Shekhovtsov, Jean Dezert
> *Journal of Computational Science* (accepted)

This repository provides tools for the **SSD test** — a nonparametric significance test for the Weighted Similarity (WS) coefficient — and for estimating the Beta distribution parameters that enable its analytical approximation.

## Background

The WS coefficient is an asymmetric similarity measure for rankings that assigns exponentially decreasing weights to positions, so that disagreements near the top are penalized more heavily than those at the bottom:

$$\text{WS}(x, y) = 1 - \sum_{i=1}^{n} 2^{-x_i} \cdot \frac{|x_i - y_i|}{\max(|1 - x_i|,\; |n - x_i|)}$$

Despite its widespread use in multi-criteria decision analysis (MCDA), prior applications have reported raw WS scores without formal significance assessment. The **SSD (Sałabun–Shekhovtsov–Dezert) test** fills this gap by constructing an empirical null distribution through repeated comparison of the reference ranking with randomly generated rankings, yielding an empirical *p*-value:

$$\hat{p}_K = \frac{1}{K} \sum_{i=1}^{K} \mathbb{I}\{WS_i \geq WS_{\text{obs}}\}$$

The paper proves that this estimator is **unbiased**, **consistent**, and **asymptotically normal**, and provides finite-sample concentration bounds via Hoeffding's inequality.

### Beta approximation

To avoid the computational cost of Monte Carlo sampling in large-scale or real-time settings, the null distribution of the WS statistic can be approximated by a Beta distribution with shape parameters $\alpha_n$ and $\beta_n$ fitted for each ranking length $n$. The approximate *p*-value is then computed analytically:

$$\hat{p}_{\text{beta}} = 1 - F_{\text{Beta}}(WS_{\text{obs}};\; \alpha_n, \beta_n)$$

The paper demonstrates that this approximation is reliable for ranking lengths $n \geq 6$.

## Repository contents

| File | Description |
|---|---|
| `main.py` | Estimation of tabular $(\alpha_n, \beta_n)$ parameters via Monte Carlo simulation and MLE fitting |
| `ws_beta_parameters.csv` | Pre-computed parameter table for $n \in [3, 100]$ (1 million samples per $n$) |

## Requirements

- Python 3.9+
- NumPy, SciPy, pandas, joblib, tqdm

```bash
pip install numpy scipy pandas joblib tqdm
```

## Usage

### Reproducing the parameter table

```bash
python main.py --n-min 3 --n-max 100 --samples 10000000 --output ws_beta_parameters.csv
```

| Argument | Default | Description |
|---|---|---|
| `--n-min` | 3 | Smallest ranking length $n$ |
| `--n-max` | 100 | Largest ranking length $n$ (inclusive) |
| `--samples` | 100 000 | Monte Carlo samples per $n$ |
| `--decimals` | 2 | Decimal places for rounding (controls tie frequency) |
| `--n-jobs` | -1 | Parallel workers (`-1` = all CPU cores) |
| `--output` | `ws_beta_parameters.csv` | Output CSV path |
| `--include-loc-scale` | off | Also store `loc` and `scale` from `beta.fit` |

> **Note:** Reproducing the table used in the paper (10M samples × 98 values of $n$) requires substantial computation time.

### Computing a *p*-value from the pre-computed table

```python
import pandas as pd
from scipy.stats import beta
from main import compute_ws, get_alpha_beta_from_table

# Observed rankings
x = [1, 2, 7, 3, 4, 5, 6, 8, 9, 10]
y = [1, 3, 4, 5, 2, 6, 7, 9, 8, 10]

ws_obs = compute_ws(x, y)  # 0.8981

# Load pre-computed Beta parameters for n=10
table = pd.read_csv("ws_beta_parameters.csv")
alpha_n, beta_n = get_alpha_beta_from_table(n=10, table=table)

# Analytical p-value
p_value = 1 - beta.cdf(ws_obs, alpha_n, beta_n)
print(f"WS_obs = {ws_obs:.4f}, p-value = {p_value:.4f}")
```

### Running the full nonparametric SSD test

```python
import numpy as np
from scipy.stats import rankdata
from main import compute_ws

x = np.array([1, 2, 7, 3, 4, 5, 6, 8, 9, 10], dtype=float)
y = np.array([1, 3, 4, 5, 2, 6, 7, 9, 8, 10], dtype=float)
ws_obs = compute_ws(x, y)

K = 100_000
n = len(x)
count = 0
for _ in range(K):
    r = np.round(np.random.rand(n), 2)
    pi = rankdata(r, method="average")
    if compute_ws(x, pi) >= ws_obs:
        count += 1

p_value = count / K
print(f"WS_obs = {ws_obs:.4f}, p-value = {p_value:.4f}")
```

## Output format

The CSV parameter table contains one row per ranking length:

| Column | Description |
|---|---|
| `n` | Ranking length |
| `alpha` | Beta shape parameter $\alpha_n$ |
| `beta` | Beta shape parameter $\beta_n$ |
| `a`, `b` | Aliases (backward compatibility) |

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{salabun2025ssd,
  title   = {Nonparametric Significance Test of the Weighted Similarity Coefficient},
  author  = {Sa{\l}abun, Wojciech and Shekhovtsov, Andrii and Dezert, Jean},
  journal = {Journal of Computational Science},
  year    = {in press},
  note    = {Accepted}
}
```

## Acknowledgements

This work was supported by the National Science Centre, Poland (grant 2024/55/D/ST6/01627).

## License

See `LICENSE` for details.
