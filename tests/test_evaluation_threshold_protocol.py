"""
Tests for threshold protocol helpers in scripts.evaluation.
"""

import unittest

from omegaconf import OmegaConf

from scripts.evaluation.reporting.threshold_protocol import (
    build_evaluation_threshold_protocol,
    build_primary_metric_selector,
    enforce_post_threshold_mode,
    make_fixed_protocol,
    make_sweep_protocol,
    make_sweep_protocol_from_spec,
    parse_thresholds,
    select_primary_threshold,
)


class TestThresholdParsing(unittest.TestCase):
    def test_parse_range_notation(self):
        thresholds = parse_thresholds("0.1:0.3:0.1")
        self.assertEqual(thresholds, [0.1, 0.2, 0.3])

    def test_parse_csv_notation(self):
        thresholds = parse_thresholds("0.15, 0.5,0.9")
        self.assertEqual(thresholds, [0.15, 0.5, 0.9])

    def test_invalid_empty_spec_raises(self):
        with self.assertRaises(ValueError):
            parse_thresholds("   ")


class TestProtocolBuilders(unittest.TestCase):
    def test_make_fixed_protocol(self):
        protocol = make_fixed_protocol(0.5)
        self.assertEqual(protocol.mode, "fixed")
        self.assertEqual(protocol.thresholds, [0.5])
        self.assertIsNone(protocol.optimize_metric)

    def test_make_sweep_protocol_from_spec(self):
        protocol = make_sweep_protocol_from_spec("0.1:0.3:0.1", optimize_metric="dice")
        self.assertEqual(protocol.mode, "sweep")
        self.assertEqual(protocol.thresholds, [0.1, 0.2, 0.3])
        self.assertEqual(protocol.optimize_metric, "dice")

    def test_enforce_post_threshold_rejects_sweep(self):
        protocol = make_sweep_protocol([0.3, 0.5, 0.7], optimize_metric="f1")
        with self.assertRaises(ValueError):
            enforce_post_threshold_mode(protocol)

    def test_enforce_post_threshold_accepts_fixed(self):
        protocol = make_fixed_protocol(0.5)
        enforced = enforce_post_threshold_mode(protocol)
        self.assertEqual(enforced.mode, "fixed")
        self.assertEqual(enforced.thresholds, [0.5])


class TestPrimaryThresholdSelection(unittest.TestCase):
    def test_select_primary_threshold_for_sweep(self):
        protocol = make_sweep_protocol([0.3, 0.5, 0.7], optimize_metric="f1")
        finalized_results = {
            0.3: {"metrics": {"f1": {"all_slices": {"mean": 0.6}, "foreground_only": {"mean": 0.6}}}},
            0.5: {"metrics": {"f1": {"all_slices": {"mean": 0.72}, "foreground_only": {"mean": 0.7}}}},
            0.7: {"metrics": {"f1": {"all_slices": {"mean": 0.68}, "foreground_only": {"mean": 0.67}}}},
        }
        selected = select_primary_threshold(finalized_results, protocol)
        self.assertEqual(selected, 0.5)

    def test_select_primary_threshold_for_fixed(self):
        protocol = make_fixed_protocol(0.42)
        finalized_results = {
            0.42: {"metrics": {"dice": {"all_slices": {"mean": 0.8}, "foreground_only": {"mean": 0.81}}}},
        }
        selected = select_primary_threshold(finalized_results, protocol)
        self.assertEqual(selected, 0.42)


class TestEvaluationThresholdProtocol(unittest.TestCase):
    def test_build_fixed_protocol_from_config(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "mode": "fixed",
                        "thresholds": "0.1:0.9:0.1",
                        "fixed_threshold": 0.5,
                        "primary": {
                            "level": "volume",
                            "metric": "DiceNativeCoefficient",
                            "statistic": "mean",
                            "direction": "max",
                        },
                    }
                }
            }
        )

        protocol = build_evaluation_threshold_protocol(cfg)

        self.assertEqual(protocol.mode, "fixed")
        self.assertEqual(protocol.thresholds, [0.5])
        self.assertEqual(protocol.fixed_threshold, 0.5)
        self.assertEqual(protocol.primary.level, "volume")

    def test_build_sweep_protocol_includes_fixed_threshold(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "mode": "sweep",
                        "thresholds": "0.1:0.3:0.1",
                        "fixed_threshold": 0.5,
                        "primary": {
                            "level": "volume",
                            "metric": "DiceNativeCoefficient",
                            "statistic": "mean",
                            "direction": "max",
                        },
                    }
                }
            }
        )

        protocol = build_evaluation_threshold_protocol(cfg)

        self.assertEqual(protocol.mode, "sweep")
        self.assertEqual(protocol.thresholds, [0.1, 0.2, 0.3, 0.5])

    def test_build_oracle_protocol_from_list_thresholds(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "mode": "oracle_per_case",
                        "thresholds": [0.2, 0.4, 0.6],
                        "fixed_threshold": 0.5,
                        "primary": {
                            "level": "volume",
                            "metric": "DiceNativeCoefficient",
                            "statistic": "median",
                            "direction": "max",
                        },
                    }
                }
            }
        )

        protocol = build_evaluation_threshold_protocol(cfg)

        self.assertEqual(protocol.mode, "oracle_per_case")
        self.assertEqual(protocol.thresholds, [0.2, 0.4, 0.5, 0.6])
        self.assertEqual(protocol.primary.statistic, "median")

    def test_build_sweep_with_oracle_protocol(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "mode": "sweep_with_oracle",
                        "thresholds": "0.2,0.4",
                        "fixed_threshold": 0.5,
                        "primary": {
                            "level": "slice",
                            "metric": "dice",
                            "statistic": "mean",
                            "direction": "max",
                        },
                    }
                }
            }
        )

        protocol = build_evaluation_threshold_protocol(cfg)

        self.assertEqual(protocol.mode, "sweep_with_oracle")
        self.assertEqual(protocol.thresholds, [0.2, 0.4, 0.5])
        self.assertEqual(protocol.primary.level, "slice")

    def test_invalid_protocol_mode_raises(self):
        cfg = OmegaConf.create(
            {"evaluation": {"threshold_protocol": {"mode": "invalid"}}}
        )

        with self.assertRaises(ValueError):
            build_evaluation_threshold_protocol(cfg)

    def test_invalid_primary_selector_level_raises(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "primary": {
                            "level": "case",
                            "metric": "DiceNativeCoefficient",
                            "statistic": "mean",
                            "direction": "max",
                        }
                    }
                }
            }
        )

        with self.assertRaises(ValueError):
            build_primary_metric_selector(cfg)

    def test_invalid_primary_selector_statistic_raises(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "primary": {
                            "level": "volume",
                            "metric": "DiceNativeCoefficient",
                            "statistic": "p95",
                            "direction": "max",
                        }
                    }
                }
            }
        )

        with self.assertRaises(ValueError):
            build_primary_metric_selector(cfg)

    def test_invalid_primary_selector_direction_raises(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "threshold_protocol": {
                        "primary": {
                            "level": "volume",
                            "metric": "DiceNativeCoefficient",
                            "statistic": "mean",
                            "direction": "closest",
                        }
                    }
                }
            }
        )

        with self.assertRaises(ValueError):
            build_primary_metric_selector(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)

