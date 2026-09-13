"""Origin-only numerical/source guards for the offline segmentation experiment."""
import copy
import importlib.util
import math
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "segmentation_features", Path(__file__).parents[1] / "scripts/cash-flow-segmentation-features.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

COLUMNS = [
    "revenues", "operating_income", "cash_flow_from_operating_activities", "free_cash_flow",
    "total_equity", "net_debt", "total_assets", "cash_and_equivalents", "current_assets",
    "current_liabilities", "intangible_assets", "tangible_assets",
]
INDEX = {"columns": COLUMNS, "sources": [{"id": "vintage", "as_of": "2026-08-10"}]}


def packed(year, ratio=1, **changes):
    values = dict(zip(COLUMNS, [100, 10, 15, 10, 80, 25, 200, 10, 50, 20, 20, 100], strict=True))
    values.update(changes)
    amounts = [None if values[key] is None else values[key] * ratio for key in COLUMNS]
    return [year, 5, f"{year}-01-01", f"{year}-12-31", f"{year + 1}-02-01", "EUR", ratio, "vintage", *amounts]


def company(**changes):
    return {"id": "1", "annual": [packed(year, **changes) for year in range(2015, 2021)], "withheld": []}


class SegmentationFeatureTests(unittest.TestCase):
    def features(self, value=None, **kwargs):
        return module.extract_features(module.decode_training(value or company(), INDEX, 2019), **kwargs)

    def test_native_conversion_and_exact_ratios(self):
        value = company(ratio=4)
        features = self.features(value)
        self.assertEqual(features["tangible_assets_to_revenue"], 1)
        self.assertEqual(features["tangible_assets_to_total_assets"], 0.5)
        self.assertEqual(features["tangible_book_to_ebit"], 6)
        self.assertEqual(features["ebit_margin"], 0.1)
        self.assertEqual(features["ebit_margin_volatility"], 0)
        self.assertEqual(features["fcf_volatility"], 0)
        self.assertEqual(features["cfo_volatility"], 0)
        self.assertEqual(features["revenue_volatility"], 0)
        self.assertEqual(features["net_debt_to_mean_ebit"], 2.5)
        self.assertEqual(features["net_debt_to_total_assets"], 0.125)
        self.assertEqual(features["working_capital_proxy_to_revenue"], 0.2)
        self.assertAlmostEqual(features["cash_gap_level"], 1 / 3)
        self.assertEqual(features["cash_gap_volatility"], 0)
        self.assertEqual(features["cash_sign_regime"], "positive")
        self.assertEqual(features["tangible_book_to_ebit_status"], "valid")
        self.assertTrue(all(count == 5 for count in features["valid_counts"].values()))

    def test_future_poison_changes_nothing_including_fingerprint(self):
        original = company()
        changed = copy.deepcopy(original)
        changed["annual"][-1] = packed(2020, revenues=1e18, free_cash_flow=-1e18, operating_income=1e16)
        changed["annual"].append([2021, "deliberately malformed future payload"])
        left = module.decode_training(original, INDEX, 2019)
        right = module.decode_training(changed, INDEX, 2019)
        self.assertEqual(left, right)
        self.assertEqual(module.extract_features(left), module.extract_features(right))
        self.assertEqual(module.training_fingerprint(left, INDEX["columns"]), module.training_fingerprint(right, INDEX["columns"]))

    def test_lineage_mismatch_is_rejected(self):
        value = company()
        good = dict(expected_years=[2019, 2018, 2017, 2016, 2015], expected_source_ids=["vintage"] * 5,
                    expected_cash=[10] * 5, expected_currency="EUR")
        self.assertEqual(len(module.decode_training(value, INDEX, 2019, **good)), 5)
        for key, changed in [("expected_years", [2018] * 5), ("expected_source_ids", ["wrong"] * 5),
                             ("expected_cash", [11] * 5), ("expected_currency", "USD")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.decode_training(value, INDEX, 2019, **(good | {key: changed}))

    def test_missing_feature_is_not_zero_or_partial_median(self):
        value = company()
        value["annual"][0] = packed(2015, tangible_assets=None)
        features = self.features(value)
        self.assertIsNone(features["tangible_assets_to_revenue"])
        self.assertIsNone(features["tangible_assets_to_total_assets"])
        self.assertEqual(features["valid_counts"]["tangible_assets_to_revenue"], 4)
        self.assertEqual(features["ebit_margin"], 0.1)

    def test_nonpositive_tangible_book_and_ebit_are_not_asset_light(self):
        for changes, expected in [({"total_equity": 10}, "nonpositive_tangible_book"),
                                  ({"operating_income": -1}, "nonpositive_ebit"),
                                  ({"operating_income": 0}, "nonpositive_ebit"),
                                  ({"operating_income": 0.5}, "near_zero_positive_ebit")]:
            with self.subTest(changes=changes):
                features = self.features(company(**changes))
                self.assertIsNone(features["tangible_book_to_ebit"])
                self.assertEqual(features["tangible_book_to_ebit_status"], expected)
                self.assertIn(expected, features["tangible_book_to_ebit_flags"])

    def test_signs_and_population_volatility(self):
        value = company()
        cash = [-10, -5, 0, 5, 10]
        for row, amount in zip(value["annual"], cash, strict=False):
            row[8 + COLUMNS.index("free_cash_flow")] = amount
        features = self.features(value)
        self.assertEqual(features["cash_sign_regime"], "mixed")
        self.assertAlmostEqual(features["fcf_volatility"], math.sqrt(50) / 6)
        zero = self.features(company(free_cash_flow=0, cash_flow_from_operating_activities=0))
        self.assertIsNone(zero["fcf_volatility"])
        self.assertIsNone(zero["cfo_volatility"])
        self.assertIsNone(zero["cash_gap_level"])
        self.assertEqual(zero["cash_sign_regime"], "zero")

    def test_source_guards_gap_withheld_currency_ratio_and_duplicates(self):
        for problem in ["missing", "withheld", "currency", "ratio", "overlap", "duplicate", "source", "all_zero"]:
            value = company()
            if problem == "missing":
                value["annual"].pop(0)
            elif problem == "withheld":
                value["withheld"] = [{"year": 2015, "period": 5}]
            elif problem == "currency":
                value["annual"][0][5] = "USD"
            elif problem == "ratio":
                value["annual"][0][6] = 0
            elif problem == "overlap":
                value["annual"][1][2] = "2015-12-01"
            elif problem == "duplicate":
                value["annual"].append(value["annual"][0])
            elif problem == "source":
                value["annual"][0][7] = "unknown"
            else:
                value["annual"][0][8:] = [0] * len(COLUMNS)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                module.decode_training(value, INDEX, 2019)

    def test_tiny_or_negative_revenue_does_not_make_invalid_ratios(self):
        for revenue in [0, -100, None]:
            with self.subTest(revenue=revenue):
                features = self.features(company(revenues=revenue))
                self.assertIsNone(features["tangible_assets_to_revenue"])
                self.assertIsNone(features["ebit_margin"])
                self.assertIsNone(features["revenue_volatility"])
                self.assertIsNone(features["working_capital_proxy_to_revenue"])

    def test_native_fingerprint_ignores_listing_and_conversion_units(self):
        first = module.decode_training(company(), INDEX, 2019)
        second = module.decode_training(company(ratio=4), INDEX, 2019)
        self.assertEqual(module.training_fingerprint(first, COLUMNS), module.training_fingerprint(second, COLUMNS))
        self.assertNotEqual(module.training_fingerprint(first, COLUMNS),
                            module.training_fingerprint(module.decode_training(company(total_assets=201), INDEX, 2019), COLUMNS))

    def test_residual_dispersion_separates_linear_growth_from_level_variation(self):
        value = company()
        for index, row in enumerate(value["annual"]):
            row[8 + COLUMNS.index("free_cash_flow")] = 10 + 5 * index
        features = self.features(value)
        self.assertGreater(features["fcf_volatility"], 0.3)
        self.assertAlmostEqual(features["fcf_trend_residual_volatility"], 0)
        value["annual"][2][8 + COLUMNS.index("free_cash_flow")] += 10
        self.assertGreater(self.features(value)["fcf_trend_residual_volatility"], 0.1)

    def test_leverage_keeps_net_cash_and_missing_history_requirements(self):
        value = company(net_debt=-20)
        features = self.features(value)
        self.assertEqual(features["net_debt_to_mean_ebit"], -2)
        self.assertEqual(features["net_debt_to_total_assets"], -0.1)
        value["annual"][0][8 + COLUMNS.index("operating_income")] = None
        features = self.features(value)
        self.assertIsNone(features["net_debt_to_mean_ebit"])
        self.assertEqual(features["net_debt_to_total_assets"], -0.1)


if __name__ == "__main__":
    unittest.main()
