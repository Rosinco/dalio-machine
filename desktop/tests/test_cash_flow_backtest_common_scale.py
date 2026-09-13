"""Independent fixtures for matched-window comparison denominators."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "common_scale", Path(__file__).parents[1] / "scripts/cash-flow-backtest-common-scale.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CommonScaleTests(unittest.TestCase):
    @staticmethod
    def row(window, training, prediction=3, actual=2):
        return {
            "train_cash": "|".join(map(str, training)), "window": str(window),
            "prediction": str(prediction), "actual": str(actual),
            "lower": str(prediction * 0.9), "upper": str(prediction * 1.1),
        }

    def test_different_original_scales_share_latest_five_scale(self):
        five = module.measure(self.row(5, [1] * 5))
        ten = module.measure(self.row(10, [1] * 5 + [100] * 5, prediction=4))
        self.assertEqual(five["scale"], 1)
        self.assertEqual(ten["scale"], 1)
        self.assertEqual(five["normalizedAbsoluteError"], 1)
        self.assertEqual(ten["normalizedAbsoluteError"], 2)
        self.assertAlmostEqual(five["normalizedWidth"], 0.6)
        self.assertAlmostEqual(ten["normalizedWidth"], 0.8)
        self.assertAlmostEqual(ten["normalizedIntervalScore"], 16.8)

    def test_zero_common_scale_does_not_use_nonzero_older_cash(self):
        five = module.measure(self.row(5, [0] * 5))
        ten = module.measure(self.row(10, [0] * 5 + [100] * 5))
        for value in [five, ten]:
            self.assertEqual(value["scale"], 0)
            self.assertIsNone(value["normalizedAbsoluteError"])
            self.assertIsNone(value["normalizedWidth"])
            self.assertIsNone(value["normalizedIntervalScore"])
            self.assertEqual(value["outcome"], "below")

    def test_normalizer_does_not_use_target(self):
        a = module.measure(self.row(5, [-5, 0, 5, -10, 10], actual=1))
        b = module.measure(self.row(5, [-5, 0, 5, -10, 10], actual=1e12))
        self.assertEqual(a["scale"], 6)
        self.assertEqual(a["scale"], b["scale"])


if __name__ == "__main__":
    unittest.main()
