"""Deterministic inference helpers for paired time series.

These routines are safeguards against over-interpreting a backtest, not proof
of tradable alpha.  The bootstrap keeps candidate and baseline observations
paired and preserves short-run dependence with circular moving blocks.  Its
percentile intervals and one-sided tail fractions remain exploratory because
they do not correct for every research choice, regime shift, or data defect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_BLOCK_SESSIONS = 20
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20_260_729
DEFAULT_ANNUALIZATION = 252


def _as_paired_vectors(
    candidate_daily_pnl: Sequence[float],
    baseline_daily_pnl: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite one-dimensional vectors after checking optional indexes."""

    candidate_index = getattr(candidate_daily_pnl, "index", None)
    baseline_index = getattr(baseline_daily_pnl, "index", None)
    if not callable(candidate_index) and not callable(baseline_index):
        if candidate_index is not None and baseline_index is not None:
            if not np.array_equal(
                np.asarray(candidate_index), np.asarray(baseline_index)
            ):
                raise ValueError("candidate and baseline indexes must match in order")

    candidate = np.asarray(candidate_daily_pnl, dtype=float)
    baseline = np.asarray(baseline_daily_pnl, dtype=float)
    if candidate.ndim != 1 or baseline.ndim != 1:
        raise ValueError("candidate and baseline P&L must be one-dimensional")
    if not len(candidate) or candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline P&L must be nonempty equal vectors")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("candidate and baseline P&L must contain only finite values")
    return candidate, baseline


def _annualized_sharpe(values: np.ndarray, annualization: int) -> float:
    """Calculate a finite Sharpe, conservatively mapping zero variance to zero."""

    if len(values) < 2:
        return 0.0
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    standard_deviation = float(np.std(scaled, ddof=1))
    tolerance = np.finfo(float).eps * max(1.0, abs(float(np.mean(scaled))))
    if standard_deviation <= tolerance:
        return 0.0
    return float(
        math.sqrt(annualization) * float(np.mean(scaled)) / standard_deviation
    )


def _bootstrap_summary(
    observed: float,
    samples: np.ndarray,
    null_samples: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "observed_difference": float(observed),
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_probability_greater_than_zero": float(np.mean(samples > 0.0)),
        # The add-one correction prevents a simulated null tail of zero.  Unlike
        # the percentile support probability above, this tail is calculated from
        # a distribution on which the relevant equality null has been imposed.
        "one_sided_null_p_value": float(
            (1 + np.count_nonzero(null_samples >= observed))
            / (len(null_samples) + 1)
        ),
    }


def paired_circular_block_bootstrap(
    candidate_daily_pnl: Sequence[float],
    baseline_daily_pnl: Sequence[float],
    *,
    block_sessions: int = DEFAULT_BLOCK_SESSIONS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    annualization: int = DEFAULT_ANNUALIZATION,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare daily P&L and Sharpe with paired circular moving blocks.

    Inputs must represent the same sessions in the same chronological order.
    When both inputs expose indexes (for example, pandas Series), their indexes
    must also match exactly.  Every bootstrap draw applies one circular block
    index to both paths, preserving pairing.  Daily-P&L Sharpe uses sample
    standard deviation and maps zero-variance samples to zero so all reported
    statistics remain finite.

    ``one_sided_null_p_value`` is an add-one empirical upper-tail probability.
    For mean P&L, the paired difference is centered to zero.  For Sharpe, each
    path is shifted to a common pooled sample Sharpe while its residuals,
    volatility, and paired dependence are retained.  This null-imposed bootstrap
    supports the preregistered Holm screen, but remains exploratory rather than
    a substitute for new prospective data.
    """

    candidate, baseline = _as_paired_vectors(
        candidate_daily_pnl, baseline_daily_pnl
    )
    if isinstance(block_sessions, bool) or not isinstance(block_sessions, int):
        raise ValueError("block_sessions must be an integer")
    if not 1 <= block_sessions <= len(candidate):
        raise ValueError("block_sessions must be in [1, sample length]")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100")
    if isinstance(annualization, bool) or not isinstance(annualization, int):
        raise ValueError("annualization must be a positive integer")
    if annualization < 1:
        raise ValueError("annualization must be a positive integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    observations = len(candidate)
    blocks_needed = math.ceil(observations / block_sessions)
    offsets = np.arange(block_sessions)
    rng = np.random.default_rng(seed)
    mean_differences = np.empty(resamples)
    sharpe_differences = np.empty(resamples)
    null_mean_differences = np.empty(resamples)
    null_sharpe_differences = np.empty(resamples)

    paired_difference = candidate - baseline
    candidate_mean_null = candidate - float(np.mean(paired_difference)) / 2.0
    baseline_mean_null = baseline + float(np.mean(paired_difference)) / 2.0

    candidate_sharpe = _annualized_sharpe(candidate, annualization)
    baseline_sharpe = _annualized_sharpe(baseline, annualization)
    common_sharpe = (candidate_sharpe + baseline_sharpe) / 2.0
    candidate_std = float(np.std(candidate, ddof=1)) if observations > 1 else 0.0
    baseline_std = float(np.std(baseline, ddof=1)) if observations > 1 else 0.0
    candidate_target_mean = common_sharpe * candidate_std / math.sqrt(annualization)
    baseline_target_mean = common_sharpe * baseline_std / math.sqrt(annualization)
    candidate_sharpe_null = candidate - float(np.mean(candidate)) + candidate_target_mean
    baseline_sharpe_null = baseline - float(np.mean(baseline)) + baseline_target_mean

    for draw in range(resamples):
        starts = rng.integers(0, observations, size=blocks_needed)
        indexes = ((starts[:, None] + offsets) % observations).ravel()[:observations]
        candidate_sample = candidate[indexes]
        baseline_sample = baseline[indexes]
        mean_differences[draw] = float(
            np.mean(candidate_sample - baseline_sample)
        )
        sharpe_differences[draw] = _annualized_sharpe(
            candidate_sample, annualization
        ) - _annualized_sharpe(baseline_sample, annualization)
        null_mean_differences[draw] = float(
            np.mean(candidate_mean_null[indexes] - baseline_mean_null[indexes])
        )
        null_sharpe_differences[draw] = _annualized_sharpe(
            candidate_sharpe_null[indexes], annualization
        ) - _annualized_sharpe(baseline_sharpe_null[indexes], annualization)

    observed_mean = float(np.mean(candidate - baseline))
    observed_sharpe = _annualized_sharpe(
        candidate, annualization
    ) - _annualized_sharpe(baseline, annualization)
    return {
        "mean_daily_pnl_difference": _bootstrap_summary(
            observed_mean, mean_differences, null_mean_differences, confidence
        ),
        "annualized_sharpe_difference": _bootstrap_summary(
            observed_sharpe,
            sharpe_differences,
            null_sharpe_differences,
            confidence,
        ),
        "method": {
            "name": "paired_circular_moving_block_bootstrap",
            "block_sessions": block_sessions,
            "resamples": resamples,
            "seed": seed,
            "annualization": annualization,
            "confidence": confidence,
            "zero_variance_sharpe": 0.0,
            "mean_null": "centered paired difference",
            "sharpe_null": "common pooled sample Sharpe with paired residuals retained",
            "inference_scope": "exploratory",
        },
    }


def holm_bonferroni(
    candidate_p_values: Mapping[str, float], *, alpha: float = 0.05
) -> dict[str, Any]:
    """Apply deterministic Holm--Bonferroni step-down decisions.

    The caller should include every preregistered candidate in the fixed family.
    A candidate that failed an earlier development gate can therefore remain in
    the family with a p-value of ``1.0`` instead of silently shrinking the trial
    count.  Ties are ordered by candidate name and receive identical adjusted
    p-values when their raw p-values are identical.
    """

    if not candidate_p_values:
        raise ValueError("candidate_p_values cannot be empty")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")

    normalized: dict[str, float] = {}
    for candidate, value in candidate_p_values.items():
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("candidate names must be nonempty strings")
        p_value = float(value)
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("candidate p-values must be finite and in [0, 1]")
        normalized[candidate] = p_value

    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_adjusted = 0.0
    step_down_open = True
    decisions: dict[str, dict[str, Any]] = {}
    adjusted_by_raw_value: dict[float, float] = {}

    for offset, (candidate, p_value) in enumerate(ordered):
        rank = offset + 1
        remaining = family_size - offset
        threshold = alpha / remaining
        running_adjusted = max(running_adjusted, remaining * p_value)
        adjusted = min(1.0, running_adjusted)
        adjusted_by_raw_value[p_value] = max(
            adjusted_by_raw_value.get(p_value, 0.0), adjusted
        )
        reject = step_down_open and p_value <= threshold
        if not reject:
            step_down_open = False
        decisions[candidate] = {
            "raw_p_value": p_value,
            "rank": rank,
            "holm_threshold": threshold,
            "holm_adjusted_p_value": adjusted,
            "reject_null": reject,
        }

    # The arbitrary alphabetical ordering used to break exact ties must not give
    # tied candidates different adjusted p-values.
    for decision in decisions.values():
        decision["holm_adjusted_p_value"] = adjusted_by_raw_value[
            decision["raw_p_value"]
        ]

    return {
        "method": "holm_bonferroni_step_down",
        "alpha": float(alpha),
        "family_size": family_size,
        "ordered_candidates": [candidate for candidate, _ in ordered],
        "decisions": decisions,
    }
