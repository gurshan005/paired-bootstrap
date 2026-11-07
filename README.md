# Paired block bootstrap

A Python module for comparing two daily profit-and-loss series measured on the same dates. It estimates differences in the mean and annualized Sharpe ratio, with percentile confidence intervals from a seeded circular moving-block bootstrap. A separate Holm–Bonferroni helper adjusts for comparisons across a fixed family of hypotheses.

The bootstrap draws consecutive observations in blocks and applies the same sampled indices to both series. This preserves pairing and some short-range dependence. The choice of block length affects the interval. Both the intervals and the one-sided null tests are exploratory.

## Run a synthetic example

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
```

```python
from paired_bootstrap import paired_circular_block_bootstrap

sample = [2, -1, 3, -2, 1, 0] * 6
baseline = [1, 0, 2, -2, 1, -1] * 6
result = paired_circular_block_bootstrap(
    sample, baseline, block_sessions=4, resamples=500, seed=17
)
print(result["mean_daily_pnl_difference"])
```

The test suite checks seeded reproducibility, symmetry when swapping the two series, index alignment, zero variance, and ties in the Holm adjustment. The library requires NumPy. The tests also use pandas for labeled inputs. All examples use synthetic observations.
