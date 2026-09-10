//! Version-2 market histories: validate the financial join, dated FX and arithmetic.
use super::{check, company_id, currency, day, hash_id, integer, Result};
use serde_json::Value;
use std::collections::HashSet;

const FLAGS: [&str; 9] = [
    "missing_publication",
    "missing_shares",
    "missing_price",
    "missing_currency",
    "short_period",
    "share_basis",
    "scale_suspect",
    "receipt_basis",
    "missing_fx",
];
const QUALITY: [&str; 4] = [
    "short_period",
    "share_basis",
    "scale_suspect",
    "receipt_basis",
];
fn positive(v: &Value) -> bool {
    v.as_f64().is_some_and(|n| n.is_finite() && n > 0.)
}
fn file(v: &Value) -> Result<()> {
    check(
        v["sha256"].as_str().is_some_and(hash_id)
            && v["bytes"]
                .as_u64()
                .is_some_and(|n| n > 0 && n <= 10_000_000_000)
            && v["path"].as_str().is_some_and(|p| {
                !p.is_empty()
                    && p.len() <= 2000
                    && !p.contains(['\\', ':'])
                    && p.split('/').all(|s| !s.is_empty() && s != "." && s != "..")
            }),
        "Invalid market source file",
    )
}
fn ordinal(s: &str) -> i64 {
    let y = s[..4].parse::<i64>().unwrap();
    let m = s[5..7].parse::<usize>().unwrap();
    let d = s[8..].parse::<i64>().unwrap();
    let prior = y - 1;
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    prior * 365 + prior / 4 - prior / 100
        + prior / 400
        + [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][m - 1]
        + d
        + i64::from(leap && m > 2)
}
fn near(a: &Value, b: Option<f64>) -> bool {
    match b {
        None => a.is_null(),
        Some(b) => {
            b.is_finite()
                && a.as_f64()
                    .is_some_and(|a| a > 0. && (a - b).abs() <= 1e-8 * b.abs().max(1.))
        }
    }
}

pub(super) fn validate_index(index: &Value) -> Result<()> {
    let companies = index["companies"].as_object().unwrap();
    if index["version"] == 1 {
        return check(
            index.get("market").is_none() && companies.values().all(|c| c.get("market").is_none()),
            "Market data requires financial pack version 2",
        );
    }
    let m = &index["market"];
    check(
        m["version"] == 1
            && m["method"] == "reported-shares-publication-close-v1"
            && m["base_pack"].as_str().is_some_and(hash_id)
            && m["entry_days"] == 30
            && m["fx_days"] == 7,
        "Unsupported market-cap method",
    )?;
    let sources = m["sources"].as_array().ok_or("Missing market sources")?;
    check(
        !sources.is_empty() && sources.len() <= 1000,
        "Invalid market source bounds",
    )?;
    let mut seen = HashSet::new();
    for s in sources {
        let id = s["id"].as_str().ok_or("Invalid market source ID")?;
        check(
            seen.insert(id)
                && index["sources"].as_array().unwrap().iter().any(|r| {
                    r["id"] == id && r["frequency"] == "annual" && r["as_of"] == s["as_of"]
                }),
            "Market source must match an annual source",
        )?;
        file(&s["prices"])?;
        file(&s["instruments"])?;
        let pairs = s["fx_pairs"]
            .as_object()
            .ok_or("Missing market FX definitions")?;
        check(
            pairs.len() <= 256
                && pairs.iter().all(|(k, v)| {
                    company_id(k)
                        && v.as_str().is_some_and(|p| {
                            p.len() == 7
                                && p.is_ascii()
                                && &p[3..4] == "/"
                                && p[..3]
                                    .bytes()
                                    .chain(p[4..].bytes())
                                    .all(|b| b.is_ascii_uppercase())
                        })
                }),
            "Invalid market FX pairs",
        )?;
    }
    check(
        index["sources"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| s["frequency"] == "annual")
            .all(|s| seen.contains(s["id"].as_str().unwrap())),
        "Missing market snapshot source",
    )?;
    file(&m["basis_code"])?;
    let mut totals = [0u64; 6];
    for c in companies.values() {
        let v = &c["market"];
        for (i, k) in ["count", "local", "sek", "flagged"].iter().enumerate() {
            check(integer(&v[k]), "Invalid market coverage")?;
            totals[i] += v[k].as_u64().unwrap();
        }
        let count = v["count"].as_u64().unwrap();
        let local = v["local"].as_u64().unwrap();
        check(
            v["count"] == c["annual"]["count"]
                && local <= count
                && v["sek"].as_u64().unwrap() <= local
                && v["flagged"].as_u64().unwrap() <= count - local,
            "Market coverage bounds do not reconcile",
        )?;
        totals[4] += u64::from(local > 0);
        totals[5] += u64::from(v["sek"].as_u64().unwrap() > 0);
    }
    for (i, k) in ["count", "local", "sek", "flagged", "with_local", "with_sek"]
        .iter()
        .enumerate()
    {
        check(
            m["summary"][k] == totals[i],
            "Market summary does not reconcile",
        )?;
    }
    check(
        m["summary"]["listings"] == companies.len(),
        "Market listing count does not reconcile",
    )
}

pub(super) fn validate_company(index: &Value, company: &Value, id: &str) -> Result<()> {
    if index["version"] == 1 {
        return check(
            company.get("market").is_none(),
            "Market rows require version 2",
        );
    }
    let rows = company["market"]
        .as_array()
        .ok_or("Missing market history")?;
    let annual = company["annual"].as_array().unwrap();
    check(
        rows.len() == annual.len(),
        "Market history must cover every annual observation",
    )?;
    let mut totals = [rows.len() as u64, 0, 0, 0];
    for (r, a) in rows.iter().zip(annual) {
        for k in [
            "year",
            "source_id",
            "currency",
            "shares",
            "price",
            "price_date",
            "local",
            "sek",
            "fx_rate",
            "fx_date",
            "fx_method",
            "fx_instruments",
            "flags",
        ] {
            check(r.get(k).is_some(), "Missing market field")?;
        }
        check(
            r["year"] == a[0] && r["source_id"] == a[7],
            "Market history does not match annual report",
        )?;
        let source = index["market"]["sources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["id"] == r["source_id"])
            .ok_or("Unknown market source")?;
        check(
            r["currency"].is_null() || currency(&r["currency"]),
            "Invalid market currency",
        )?;
        for k in ["shares", "price", "local", "sek", "fx_rate"] {
            check(r[k].is_null() || positive(&r[k]), "Invalid market amount")?;
        }
        if r["price"].is_null() {
            check(r["price_date"].is_null(), "Missing price carries a date")?;
        } else {
            let p = r["price_date"]
                .as_str()
                .ok_or("Missing market price date")?;
            let published = a[4]
                .as_str()
                .ok_or("No report publication for market price")?;
            check(
                day(p)
                    && r["price"].as_f64().unwrap() < 1e10
                    && p >= published
                    && p <= source["as_of"].as_str().unwrap()
                    && ordinal(p) - ordinal(published) <= 30,
                "Market price outside publication window",
            )?;
        }
        let flags = r["flags"]
            .as_array()
            .ok_or("Missing market quality flags")?;
        let flags: Vec<&str> = flags.iter().map(|f| f.as_str().unwrap_or("")).collect();
        check(
            flags.iter().all(|f| FLAGS.contains(f))
                && flags.iter().collect::<HashSet<_>>().len() == flags.len(),
            "Invalid market quality flags",
        )?;
        let has = |f: &str| flags.contains(&f);
        check(
            has("missing_publication") == a[4].is_null()
                && has("missing_shares") == r["shares"].is_null()
                && has("missing_price") == r["price"].is_null()
                && has("missing_currency") == r["currency"].is_null(),
            "Market missingness flags do not reconcile",
        )?;
        let span = ordinal(a[3].as_str().unwrap()) - ordinal(a[2].as_str().unwrap()) + 1;
        check(
            has("short_period") == !(330..=400).contains(&span),
            "Market fiscal-period flag disagrees",
        )?;
        let candidate = r["shares"]
            .as_f64()
            .zip(r["price"].as_f64())
            .map(|(s, p)| s * p);
        let suspect = candidate.is_some_and(|c| {
            !c.is_finite()
                || c <= 0.
                || (positive(&a[10]) && c / a[10].as_f64().unwrap() < 1.)
                || (positive(&a[13]) && c / a[13].as_f64().unwrap() < 0.05)
        });
        check(
            has("scale_suspect") == suspect,
            "Market scale flag disagrees with report",
        )?;
        let local = if flags.iter().any(|f| *f != "missing_fx") {
            None
        } else {
            candidate
        };
        check(
            near(&r["local"], local),
            "Market cap does not equal shares times price",
        )?;
        let ids = r["fx_instruments"]
            .as_array()
            .ok_or("Missing FX source IDs")?;
        check(
            ids.len() <= 2
                && ids.iter().all(|v| v.as_str().is_some_and(company_id))
                && ids
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<HashSet<_>>()
                    .len()
                    == ids.len(),
            "Invalid FX source IDs",
        )?;
        if r["fx_rate"].is_null() {
            check(
                r["fx_date"].is_null()
                    && r["fx_method"].is_null()
                    && ids.is_empty()
                    && has("missing_fx"),
                "Missing FX carries a conversion",
            )?;
        } else {
            let f = r["fx_date"].as_str().ok_or("Missing FX date")?;
            let p = r["price_date"].as_str().ok_or("FX without price date")?;
            check(
                !has("missing_fx")
                    && currency(&r["currency"])
                    && day(f)
                    && f <= p
                    && ordinal(p) - ordinal(f) <= 7,
                "FX is stale, unavailable or later than the price",
            )?;
            let ccy = r["currency"].as_str().unwrap();
            let pairs: Vec<&str> = ids
                .iter()
                .map(|k| {
                    source["fx_pairs"][k.as_str().unwrap()]
                        .as_str()
                        .unwrap_or("")
                })
                .collect();
            match r["fx_method"].as_str().unwrap_or("") {
                "identity" => check(
                    ccy == "SEK" && r["fx_rate"] == 1.0 && f == p && ids.is_empty(),
                    "Invalid SEK identity",
                )?,
                "direct" => check(
                    pairs == vec![format!("{ccy}/SEK")],
                    "Invalid direct FX direction",
                )?,
                "usd_cross" => check(
                    pairs == vec!["USD/SEK".to_string(), format!("USD/{ccy}")],
                    "Invalid cross FX direction",
                )?,
                _ => return Err("Unsupported FX method".into()),
            }
        }
        let sek = local.zip(r["fx_rate"].as_f64()).map(|(v, f)| v * f);
        check(
            near(&r["sek"], sek),
            "SEK value does not equal dated conversion",
        )?;
        totals[1] += u64::from(!r["local"].is_null());
        totals[2] += u64::from(!r["sek"].is_null());
        totals[3] += u64::from(flags.iter().any(|f| QUALITY.contains(f)));
    }
    for (i, k) in ["count", "local", "sek", "flagged"].iter().enumerate() {
        check(
            index["companies"][id]["market"][k] == totals[i],
            "Market rows disagree with coverage",
        )?;
    }
    Ok(())
}
