import importlib.util
import unittest
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location('expanded_export', Path(__file__).parents[1]/'scripts/export-expanded-kpis.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ExpandedKpiExportTests(unittest.TestCase):
    def frame(self, rows):
        return pd.DataFrame(rows, columns=['ins_id', 'n', 's', 'scope'])

    def test_scope_coalescing_preserves_zero_negative_and_observed_value(self):
        values, counts = EXPORT.resolve_cells(self.frame([(1, 0, None, 'nordic'), (1, 0, None, 'global'), (2, None, None, 'nordic'), (2, -3, None, 'global')]), ['1', '2', '3'], 'number')
        self.assertEqual(values, [0, -3, None])
        self.assertEqual(counts['scopeAvailabilityDifferenceCount'], 1)
        self.assertEqual(counts['conflictCount'], 0)

    def test_unequal_observed_scopes_are_withheld_without_precedence(self):
        values, counts = EXPORT.resolve_cells(self.frame([(1, 4, None, 'nordic'), (1, 5, None, 'global')]), ['1'], 'number')
        self.assertEqual(values, [None])
        self.assertEqual(counts['conflictCount'], 1)

    def test_date_channel_ignores_numeric_timestamp_and_checks_calendar(self):
        values, _ = EXPORT.resolve_cells(self.frame([(1, 638900000000000000, '2026-08-10', 'nordic'), (2, 1, '2026-02-30', 'nordic')]), ['1', '2'], 'date')
        self.assertEqual(values, ['2026-08-10', None])

    def test_nonnumeric_and_unexpected_string_do_not_become_financial_number(self):
        values, counts = EXPORT.resolve_cells(self.frame([(1, float('inf'), None, 'nordic'), (2, 2, 'unexpected', 'nordic')]), ['1', '2'], 'number')
        self.assertEqual(values, [None, None])
        self.assertEqual(counts['invalidValueCount'], 1)
        self.assertEqual(counts['conflictCount'], 1)

    def test_formatted_numeric_companions_are_retained_at_printed_precision(self):
        values, counts = EXPORT.resolve_cells(self.frame([(1, 41.4000015, '41,4%', 'nordic'), (2, 95.400002, '95%', 'nordic'), (3, 345.399994, '345', 'nordic'), (4, 3.5, '35%', 'nordic')]), ['1', '2', '3', '4'], 'percent')
        self.assertEqual(values, [41.4000015, 95.400002, 345.399994, None])
        self.assertEqual(counts['conflictCount'], 1)

    def test_metric_format_does_not_override_variant_units(self):
        self.assertEqual(EXPORT.variant_semantics(151, 'last', 'default', {})[:2], ('price', 'quote'))
        self.assertEqual(EXPORT.variant_semantics(151, '1year', 'return', {})[:2], ('percent', 'none'))
        self.assertEqual(EXPORT.variant_semantics(53, 'last', 'psh', {'format': 'MCURR'})[:2], ('per_share', 'report'))
        self.assertEqual(EXPORT.variant_semantics(53, '5year', 'cagr', {'format': 'MCURR'})[:2], ('percent', 'none'))

    def test_known_provider_unit_and_label_errors_have_explicit_overrides(self):
        self.assertEqual(EXPORT.variant_semantics(148, 'last', 'latest', {'format': 'CURR'})[:2], ('percent', 'none'))
        self.assertEqual(EXPORT.variant_semantics(50, 'last', 'latest', {'format': 'MCURR'})[:2], ('millions', 'quote'))
        self.assertEqual(EXPORT.variant_semantics(50, 'last', 'usd', {'format': 'MCURR'})[:2], ('millions', 'USD'))
        self.assertEqual(EXPORT.variant_semantics(49, 'last', 'latest', {})[:2], ('millions', 'quote'))
        self.assertEqual(EXPORT.variant_semantics(40, 'last', 'latest', {})[:2], ('multiple', 'none'))
        self.assertEqual(EXPORT.variant_semantics(27, 'last', 'latest', {})[:2], ('percent', 'none'))
        self.assertEqual(EXPORT.LABELS[23], 'Free cash flow per share')
        self.assertEqual(EXPORT.LABELS[41], 'Net debt %')
        self.assertEqual(EXPORT.LABELS[71], 'EBITDA per share')

    def test_defective_ncav_family_is_explicitly_excluded(self):
        self.assertEqual(set(EXPORT.EXCLUDED), {307, 308, 309, 310})

    def test_unknown_monetary_amounts_are_withheld_regardless_of_numeric_unit(self):
        for k, calc in [(110, 'ValueBuy'), (110, 'ValueSell'), (110, 'ValueNet'), (144, 'Amount'), (146, 'SumValue'), (146, 'AvgValue')]:
            _, basis, _ = EXPORT.variant_semantics(k, '1month', calc, {})
            self.assertEqual(basis, 'unverified')
            values, withheld = EXPORT.apply_currency_policy([0, 12, -4, None], ['1', '2', '3', '4'], ['SEK']*4, ['SEK']*4, k, '1month', calc, basis)
            self.assertEqual(values, [None]*4)
            self.assertEqual(withheld, 3)

    def test_price_levels_percent_distances_and_turnover_have_distinct_units(self):
        cases = [(153, 'high', ('price', 'quote')), (153, 'low', ('price', 'quote')), (153, 'pricehigh', ('percent', 'none')), (153, 'pricelow', ('percent', 'none')), (153, 'default', ('percent', 'none')), (157, 'mean', ('price', 'quote')), (161, 'diff', ('price', 'quote')), (313, 'mill', ('millions', 'quote')), (313, 'mean', ('count', 'none')), (321, 'diff', ('percent', 'none')), (322, 'diff', ('percent', 'none'))]
        for k, calc, expected in cases:
            self.assertEqual(EXPORT.variant_semantics(k, 'last', calc, {})[:2], expected)


if __name__ == '__main__': unittest.main()
