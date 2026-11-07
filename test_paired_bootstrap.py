import unittest

import numpy as np
import pandas as pd

from paired_bootstrap import (
    holm_bonferroni,
    paired_circular_block_bootstrap,
)


class PairedCircularBlockBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = [4, -1, 3, -2, 5, 0, -1, 3, 2, -2] * 4
        self.baseline = [1, -2, 2, -1, 1, 0, 0, 1, -1, -1] * 4

    def test_is_deterministic_for_both_statistics(self) -> None:
        first = paired_circular_block_bootstrap(
            self.candidate,
            self.baseline,
            block_sessions=5,
            resamples=500,
            seed=71,
        )
        second = paired_circular_block_bootstrap(
            self.candidate,
            self.baseline,
            block_sessions=5,
            resamples=500,
            seed=71,
        )
        self.assertEqual(first, second)
        self.assertIn("mean_daily_pnl_difference", first)
        self.assertIn("annualized_sharpe_difference", first)

    def test_swapping_paths_reverses_observed_statistics_and_intervals(self) -> None:
        forward = paired_circular_block_bootstrap(
            self.candidate,
            self.baseline,
            block_sessions=5,
            resamples=500,
            seed=19,
        )
        reverse = paired_circular_block_bootstrap(
            self.baseline,
            self.candidate,
            block_sessions=5,
            resamples=500,
            seed=19,
        )
        for metric in (
            "mean_daily_pnl_difference",
            "annualized_sharpe_difference",
        ):
            self.assertAlmostEqual(
                forward[metric]["observed_difference"],
                -reverse[metric]["observed_difference"],
            )
            forward_interval = forward[metric]["confidence_interval"]
            reverse_interval = reverse[metric]["confidence_interval"]
            self.assertAlmostEqual(forward_interval[0], -reverse_interval[1])
            self.assertAlmostEqual(forward_interval[1], -reverse_interval[0])

    def test_rejects_unpaired_and_invalid_vectors(self) -> None:
        invalid_pairs = (
            ([1, 2], [1]),
            ([], []),
            ([[1, 2]], [[1, 2]]),
            ([1, np.nan], [1, 2]),
        )
        for candidate, baseline in invalid_pairs:
            with self.subTest(candidate=candidate, baseline=baseline):
                with self.assertRaises(ValueError):
                    paired_circular_block_bootstrap(
                        candidate, baseline, block_sessions=1, resamples=100
                    )

    def test_rejects_differently_ordered_series_indexes(self) -> None:
        candidate = pd.Series([1.0, 2.0], index=["2025-01-02", "2025-01-03"])
        baseline = pd.Series([1.0, 2.0], index=["2025-01-03", "2025-01-02"])
        with self.assertRaisesRegex(ValueError, "indexes must match in order"):
            paired_circular_block_bootstrap(
                candidate, baseline, block_sessions=1, resamples=100
            )

    def test_zero_variance_sharpe_remains_finite(self) -> None:
        result = paired_circular_block_bootstrap(
            [5.0] * 40,
            [0.0] * 40,
            block_sessions=20,
            resamples=100,
            seed=3,
        )
        sharpe = result["annualized_sharpe_difference"]
        self.assertEqual(sharpe["observed_difference"], 0.0)
        self.assertEqual(sharpe["confidence_interval"], [0.0, 0.0])
        self.assertTrue(
            all(
                np.isfinite(value)
                for value in (
                    sharpe["observed_difference"],
                    *sharpe["confidence_interval"],
                    sharpe["bootstrap_probability_greater_than_zero"],
                    sharpe["one_sided_null_p_value"],
                )
            )
        )

    def test_mean_p_value_uses_a_zero_centered_null(self) -> None:
        result = paired_circular_block_bootstrap(
            [3.0, 1.0, 4.0, 2.0] * 10,
            [2.0, 0.0, 3.0, 1.0] * 10,
            block_sessions=5,
            resamples=500,
            seed=17,
        )

        mean = result["mean_daily_pnl_difference"]

        self.assertEqual(mean["observed_difference"], 1.0)
        self.assertEqual(mean["one_sided_null_p_value"], 1 / 501)


class HolmBonferroniTests(unittest.TestCase):
    def test_step_down_thresholds_failures_and_tie_order_are_deterministic(self) -> None:
        result = holm_bonferroni(
            {"zeta": 0.01, "alpha": 0.01, "middle": 0.03, "dev_failure": 1.0},
            alpha=0.05,
        )
        self.assertEqual(
            result["ordered_candidates"],
            ["alpha", "zeta", "middle", "dev_failure"],
        )
        decisions = result["decisions"]
        self.assertAlmostEqual(decisions["alpha"]["holm_threshold"], 0.0125)
        self.assertAlmostEqual(decisions["zeta"]["holm_threshold"], 0.05 / 3)
        self.assertTrue(decisions["alpha"]["reject_null"])
        self.assertTrue(decisions["zeta"]["reject_null"])
        self.assertFalse(decisions["middle"]["reject_null"])
        self.assertFalse(decisions["dev_failure"]["reject_null"])
        self.assertEqual(
            decisions["alpha"]["holm_adjusted_p_value"],
            decisions["zeta"]["holm_adjusted_p_value"],
        )

    def test_threshold_equality_rejects(self) -> None:
        result = holm_bonferroni(
            {"first": 0.01, "second": 0.025, "third": 0.04}, alpha=0.05
        )
        self.assertTrue(result["decisions"]["second"]["reject_null"])
        self.assertTrue(result["decisions"]["third"]["reject_null"])

    def test_rejects_invalid_family_values(self) -> None:
        for values in ({}, {"x": -0.01}, {"x": 1.01}, {"x": np.nan}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    holm_bonferroni(values)


if __name__ == "__main__":
    unittest.main()
