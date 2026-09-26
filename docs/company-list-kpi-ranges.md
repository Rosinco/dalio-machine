# Min and Max filters in company Lists

Companies → Lists shows Min and Max inputs directly beneath every numeric KPI
heading, including downloaded provider KPIs and each terminal-credit variant.
Use either bound independently or both together. Values equal to a bound are
included. All active column ranges and other list conditions must match.

For example, P/E Min `5` and Max `15`, together with dividend yield Min `3`,
keeps companies with observed P/E from 5 through 15 and dividend yield of at
least 3%. Enter percentages as displayed percentages, not decimal fractions.
Amounts use the displayed millions or per-share units and require a selected
currency; Atlas does not convert currencies to satisfy a range.

Blank means no limit on that side. Signed decimal numbers, zero, a decimal point
or Swedish decimal comma are accepted. A comma means a decimal separator, not
thousands grouping. Values use their stored precision, before display rounding.
Text and date columns retain their existing sorting and applicable category
filters rather than numeric range inputs.

A company without an observed value for an actively filtered KPI is excluded.
Loading and failed datasets cannot satisfy a range. Invalid or incomplete input,
Min greater than Max, or missing currency produces an inline message and no
matches for that range. Header controls remain available at zero matches.

Changes apply immediately and save with list preferences. Saved views include
the exact KPI, period, calculation, bounds and currency. Reordering a column
preserves its range. Editing its KPI, period or calculation clears the range;
removing the column or applying a new column preset removes attached ranges.
Independent conditions in the Filters panel retain their own exact KPI identity.
The 32-column allowance supports a range per numeric column separately from the
12 advanced conditions. Clear an individual range, all KPI ranges, or all list
filters. Export CSV uses the entire resulting filtered list.

The ranges are optional additions to v2 column preferences, including saved-view
columns. Raw bound strings preserve partial or invalid input across restart.
Reading existing preferences does not write storage; only explicit list actions
save. Older app versions do not understand these new range fields, so use the
new version when editing range-bearing views.

This feature filters existing observations. It does not change data packs,
recalculate source KPIs, reprice starter valuations, or modify company Value
drafts and reviewed studies.
