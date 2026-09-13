# ADR 0041: Customizable company lists and explicit candidate screens

Date: 2026-09-13

Status: Accepted by user direction; release verification is recorded in the handoff.

## Context

The user wants a dense list of companies worth investigating, with actual values
in individually chosen KPI columns. Their examples include a searchable KPI
selector with categories, periods, calculations and definitions. The existing
research screen provides detailed evidence but does not offer configurable
numeric columns or a personal watchlist.

## Decision

Add **Companies → Lists**, also accessible for a selected branch. Keep the
existing Screen and company Value workspaces. Lists use the same verified,
source-bound research artifact and inherit its financial-pack and directory
identity checks. This feature does not change the underlying financial histories,
starter valuations, uncertainty calibration or reviewed studies.

Users choose and order KPI columns, sort values, apply explicit filters, star
listings and save named views. The KPI selector shows supported calculations and
periods with definitions. Annual observations are labelled fiscal-year history;
they are not called current trailing twelve months. Window calculations require
the stated number of comparable valid observations. Missing amounts remain
missing; negative cash, zero observations and unavailable ratios remain distinct.

Available fields cover downloaded annual cash flow, revenue, EBIT, margins,
balance-sheet and asset proxies, comparable quarterly changes, saved quote dates,
and standard-starter valuation context. EPS, P/E, dividend yield, price performance,
R12 and longer KPI histories must await a separately verified import. A broad
raw download is not automatically an application-ready metric catalogue.

Sorting keeps missing observations last in both directions. Monetary amounts
retain their currency and unit; different currencies must be grouped or explicitly
restricted, not treated as comparable numbers. Numeric filters require compatible
units. Each cell exposes its period, calculation and missing reason where relevant.

Candidate presets declare their conditions. The cash-and-margin preset selects
operating businesses with five-period evidence, five positive provider-FCF and
EBIT observations, and a positive saved price no greater than 70% of a positive
Mid starter value. These are descriptive research filters, not validated business
quality gates or investment recommendations. All listings and missing/manual
research routes remain accessible. A personal watchlist records the user's own
choices separately from calculated matches.

Valuation columns use the existing frozen standard starter and dated price/share
proxy, separate from edited or reviewed valuations. Terminal value contributes
once, NPV subtracts the saved equity price, and the 30% margin is explicit.
Provider FCF does not establish owner cash. The preset is not a current-price
alert or a validated forecasting strategy; source dates stay visible.

Persist only list preferences under a separate versioned local-storage key.
Preserve unknown watchlist IDs across changes of pack. Malformed or future-version
preferences must not be overwritten merely by opening the page. Storage failure
must be visible. Browsing lists must never mount Value, migrate drafts, modify
revisions or write formal deep-dive outcomes.

## Validation

Check independent numeric results and exact preset membership from the saved
artifact, missing/signed observations, supported period calculations, sorting,
filters, picker keyboard behavior, saved views, watchlist persistence and profile
navigation. Verify narrow-screen scrolling, source mismatch withholding, offline
operation, unchanged authored valuations and a full Windows process restart.
Record build, tests and installation separately; retain the preceding release.
