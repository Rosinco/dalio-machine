//! Bounded identity catalogue. Financial documents remain separate.
use super::business::company_id;
use super::{array, check, date, is_id, object, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn text(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() <= 2000)
}
fn file(v: &Value) -> Result<()> {
    check(
        text(&v["path"])
            && v["path"].as_str().is_some_and(|s| {
                !s.contains(['\\', ':'])
                    && s.split('/').all(|p| !p.is_empty() && p != "." && p != "..")
            })
            && v["sha256"].as_str().is_some_and(is_id)
            && v["bytes"].as_u64().is_some_and(|n| n <= 9007199254740991),
        "Invalid catalogue source file.",
    )
}
pub fn validate(raw: &Value, business: Option<&Value>, inventory: &str) -> Result<()> {
    check(
        raw["version"] == 1
            && raw["as_of"]
                .as_str()
                .is_some_and(|s| date(s) && s <= inventory)
            && super::taxonomy::export_timestamp(&raw["exported_at"]),
        "Invalid company catalogue date or version.",
    )?;
    let types = array(&raw["included_types"])?;
    let mut actual: Vec<u64> = types.iter().map(|t| t.as_u64().unwrap_or(99)).collect();
    actual.sort();
    check(
        actual == vec![0, 1, 3, 8, 9, 10],
        "Invalid company instrument types.",
    )?;
    let countries = object(&raw["countries"])?;
    check(countries.len() <= 300, "Too many catalogue countries.")?;
    let mut codes = BTreeSet::new();
    for (key, country) in countries {
        check(
            company_id(key)
                && country["id"] == *key
                && text(&country["name"])
                && country["name"].as_str() != Some("")
                && text(&country["name_en"])
                && country["name_en"].as_str() != Some("")
                && country.get("iso2").is_some()
                && (country["iso2"].is_null()
                    || country["iso2"]
                        .as_str()
                        .is_some_and(|s| super::is_code(s) && codes.insert(s))),
            "Invalid catalogue country.",
        )?;
    }
    let snapshots = array(&raw["snapshots"])?;
    check(
        !snapshots.is_empty() && snapshots.len() <= 1000,
        "Invalid catalogue snapshot inventory.",
    )?;
    let mut sources = BTreeMap::new();
    let mut previous = "";
    for s in snapshots {
        let stamp = s["as_of"].as_str().unwrap_or("");
        check(
            date(stamp) && stamp > previous && Some(stamp) <= raw["as_of"].as_str(),
            "Invalid catalogue snapshot order.",
        )?;
        previous = stamp;
        for k in ["instrument_count", "company_count", "excluded_count"] {
            check(
                s[k].as_u64().is_some_and(|n| n <= 9007199254740991),
                "Invalid catalogue source counts.",
            )?;
        }
        check(
            s["company_count"].as_u64().unwrap() + s["excluded_count"].as_u64().unwrap()
                == s["instrument_count"].as_u64().unwrap(),
            "Catalogue source counts do not reconcile.",
        )?;
        file(&s["instruments"])?;
        file(&s["countries"])?;
        sources.insert(stamp, s["company_count"].as_u64().unwrap());
    }
    check(
        Some(previous) == raw["as_of"].as_str(),
        "Catalogue latest snapshot is missing.",
    )?;
    let listings = object(&raw["listings"])?;
    check(listings.len() <= 100000, "Too many company listings.")?;
    let mut counts = BTreeMap::<&str, u64>::new();
    for (key, row) in listings {
        check(
            company_id(key)
                && row["id"] == *key
                && row["country_id"].as_str().is_some_and(company_id)
                && row["instrument_type"]
                    .as_u64()
                    .is_some_and(|t| [0, 1, 3, 8, 9, 10].contains(&t)),
            "Invalid company listing identity.",
        )?;
        for k in ["sector_id", "branch_id"] {
            check(
                row.get(k).is_some()
                    && (row[k].is_null() || row[k].as_str().is_some_and(company_id)),
                "Invalid source classification ID.",
            )?;
        }
        for k in [
            "name",
            "ticker",
            "isin",
            "stock_currency",
            "report_currency",
        ] {
            check(
                row.get(k).is_some() && (row[k].is_null() || text(&row[k])),
                "Invalid listing text metadata.",
            )?;
        }
        let code = countries
            .get(row["country_id"].as_str().unwrap())
            .map_or(&Value::Null, |c| &c["iso2"]);
        check(
            row.get("listing_country") == Some(code)
                && row.get("listing_date").is_some()
                && (row["listing_date"].is_null()
                    || row["listing_date"].as_str().is_some_and(date)),
            "Invalid listing country or date.",
        )?;
        let stamp = row["source_as_of"].as_str().unwrap_or("");
        check(
            sources.contains_key(stamp),
            "Unknown company source snapshot.",
        )?;
        *counts.entry(stamp).or_default() += 1;
    }
    check(
        sources
            .iter()
            .all(|(stamp, n)| counts.get(stamp).copied().unwrap_or(0) <= *n)
            && counts.get(previous).copied().unwrap_or(0) == sources[previous],
        "Company catalogue counts do not reconcile with its source snapshots.",
    )?;
    if let Some(b) = business {
        for key in object(&b["companies"])?.keys() {
            check(
                listings.contains_key(key),
                "A financial profile is missing from the company catalogue.",
            )?;
        }
    }
    Ok(())
}
