"""Dated, explicitly derived listing valuations. No static FX or repaired shares."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from datetime import date, timedelta

METHOD = "reported-shares-publication-close-v1"
ENTRY_DAYS = 30
FX_DAYS = 7
PRICE_LIMIT = 1e10
FLAGS = (
    "missing_publication",
    "missing_shares",
    "missing_price",
    "missing_currency",
    "short_period",
    "share_basis",
    "scale_suspect",
    "receipt_basis",
    "missing_fx",
)
QUALITY_FLAGS = {"short_period", "share_basis", "scale_suspect", "receipt_basis"}


def positive(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def select_quote(prices, published, as_of):
    """First valid daily close in the existing engine's 30-calendar-day window."""
    if not published:
        return None
    end = min((date.fromisoformat(published) + timedelta(days=ENTRY_DAYS)).isoformat(), as_of)
    at = bisect_left(prices, (published, -math.inf))
    for i in range(at, len(prices)):
        day, price = prices[i]
        if day > end:
            break
        if positive(price) and price < PRICE_LIMIT:
            return day, float(price)
    return None


def select_fx(series, price_date):
    if not price_date:
        return None
    at = bisect_right(series, price_date, key=lambda x: x[0]) - 1
    if at < 0:
        return None
    r = series[at]
    age = (date.fromisoformat(price_date) - date.fromisoformat(r[0])).days
    return r if age <= FX_DAYS and positive(r[1]) else None


def market_record(annual, *, shares, currency, quote, fx, basis_ok, receipt=False):
    published, stamp = annual["published"], annual["as_of"]
    if quote:
        qday, price = quote
        if (
            not published
            or not published <= qday <= stamp
            or (date.fromisoformat(qday) - date.fromisoformat(published)).days > ENTRY_DAYS
            or not positive(price)
            or price >= PRICE_LIMIT
        ):
            raise ValueError(
                "Price must be a valid close within the publication window and snapshot"
            )
    else:
        qday, price = None, None
    if fx:
        fday, rate, method, ids = fx
        if (
            not qday
            or not fday <= qday
            or (date.fromisoformat(qday) - date.fromisoformat(fday)).days > FX_DAYS
            or not positive(rate)
        ):
            raise ValueError("FX must be dated at or shortly before the price")
        if method == "identity" and (currency != "SEK" or rate != 1 or fday != qday or ids):
            raise ValueError("Invalid SEK identity conversion")
        if method not in ["identity", "direct", "usd_cross"]:
            raise ValueError("Unsupported FX method")
    else:
        fday, rate, method, ids = None, None, None, []
    shares = float(shares) if positive(shares) else None
    currency = (
        currency
        if isinstance(currency, str)
        and len(currency) == 3
        and currency.isalpha()
        and currency.isupper()
        and currency.isascii()
        else None
    )
    flags = []
    if not published:
        flags.append("missing_publication")
    if shares is None:
        flags.append("missing_shares")
    if quote is None:
        flags.append("missing_price")
    if currency is None:
        flags.append("missing_currency")
    days = (date.fromisoformat(annual["end"]) - date.fromisoformat(annual["start"])).days + 1
    if not 330 <= days <= 400:
        flags.append("short_period")
    if not basis_ok:
        flags.append("share_basis")
    if receipt:
        flags.append("receipt_basis")
    candidate = shares * price if shares and price else None
    profit, equity = annual.get("profit"), annual.get("equity")
    if candidate is not None and (
        not positive(candidate)
        or positive(profit)
        and candidate / profit < 1
        or positive(equity)
        and candidate / equity < 0.05
    ):
        flags.append("scale_suspect")
    local = candidate if not flags else None
    if fx is None:
        flags.append("missing_fx")
    sek = local * rate if local is not None and rate is not None else None
    if sek is not None and not positive(sek):
        raise ValueError("SEK valuation overflow")
    return dict(
        year=annual["year"],
        source_id=annual["source_id"],
        currency=currency,
        shares=shares,
        price=price,
        price_date=qday,
        local=local,
        fx_rate=rate,
        fx_date=fday,
        fx_method=method,
        fx_instruments=ids,
        sek=sek,
        flags=sorted(flags),
    )


def market_coverage(rows):
    return dict(
        count=len(rows),
        local=sum(r["local"] is not None for r in rows),
        sek=sum(r["sek"] is not None for r in rows),
        flagged=sum(bool(QUALITY_FLAGS.intersection(r["flags"])) for r in rows),
    )
