//! Business documents are inert data. Resource names never become source paths.
use super::{array, check, date, fields, is_code, is_id, number, object, timestamp, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;

const AMOUNTS: [&str; 16] = [
    "revenues",
    "operating_income",
    "profit_to_equity_holders",
    "cash_flow_from_operating_activities",
    "free_cash_flow",
    "total_equity",
    "net_debt",
    "total_assets",
    "cash_and_equivalents",
    "current_assets",
    "current_liabilities",
    "non_current_assets",
    "non_current_liabilities",
    "intangible_assets",
    "tangible_assets",
    "financial_assets",
];
const RATIOS: [&str; 5] = [
    "operating_margin",
    "operating_cash_margin",
    "return_on_capital",
    "equity_ratio",
    "net_debt_to_equity",
];

pub fn company_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 10
        && !id.starts_with('0')
        && id.bytes().all(|c| c.is_ascii_digit())
}
fn business_date(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| date(s) && s >= "2000-01-01" && s <= "2300-12-31")
}
fn currency(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 3 && s.bytes().all(|c| c.is_ascii_uppercase()))
}
fn texts(value: &Value, max: usize) -> Result<()> {
    let rows = array(value)?;
    check(
        rows.len() <= max && rows.iter().all(Value::is_string),
        "Business text list is invalid.",
    )
}
fn studies(value: &Value, sources: &BTreeSet<String>) -> Result<()> {
    let rows = array(value)?;
    check(rows.len() <= 100, "Too many archived studies.")?;
    for row in rows {
        fields(row, &["title", "basis", "source_id"])?;
        check(
            business_date(&row["as_of"])
                && sources.contains(row["source_id"].as_str().unwrap())
                && row["text"]
                    .as_str()
                    .is_some_and(|text| text.len() <= 1000000),
            "An archived study is invalid.",
        )?;
    }
    Ok(())
}
pub fn validate(raw: &Value) -> Result<()> {
    check(
        raw["version"] == 1 && business_date(&raw["as_of"]) && timestamp(&raw["exported_at"]),
        "Unsupported business version or snapshot date.",
    )?;
    let countries = object(&raw["countries"])?;
    check(
        countries.len() <= 300 && countries.iter().all(|(k, v)| is_code(k) && v.is_string()),
        "Business country coverage is invalid.",
    )?;
    let mut sources = BTreeSet::new();
    let source_rows = array(&raw["sources"])?;
    check(source_rows.len() <= 1000, "Too many business sources.")?;
    for source in source_rows {
        fields(source, &["id", "label", "path", "sha256"])?;
        check(
            sources.insert(source["id"].as_str().unwrap().to_owned())
                && is_id(source["sha256"].as_str().unwrap())
                && source["bytes"]
                    .as_u64()
                    .is_some_and(|n| n <= 9007199254740991),
            "A business source is invalid.",
        )?;
    }
    studies(&raw["research"], &sources)?;
    texts(&raw["limitations"], 100)?;
    let branch = &raw["branch"];
    fields(
        branch,
        &[
            "id",
            "name",
            "source_name",
            "sector_id",
            "sector_name",
            "source_id",
            "description",
        ],
    )?;
    check(
        company_id(branch["id"].as_str().unwrap())
            && company_id(branch["sector_id"].as_str().unwrap())
            && business_date(&branch["as_of"])
            && sources.contains(branch["source_id"].as_str().unwrap()),
        "Business branch metadata is invalid.",
    )?;
    let drivers = array(&branch["drivers"])?;
    check(drivers.len() <= 100, "Too many branch drivers.")?;
    for driver in drivers {
        fields(driver, &["title", "text", "indicator"])?;
    }
    let companies = object(&raw["companies"])?;
    check(
        !companies.is_empty()
            && companies.len() <= 300
            && raw["default_company"]
                .as_str()
                .is_some_and(|id| companies.contains_key(id)),
        "The company catalogue is invalid.",
    )?;
    for (id, c) in companies {
        fields(
            c,
            &[
                "id",
                "name",
                "listing_name",
                "ticker",
                "isin",
                "listing_country",
                "report_currency",
                "stock_currency",
                "sector_id",
                "branch_id",
                "overlap",
                "context_source_id",
                "instrument_source_id",
                "annual_source_id",
                "quarterly_source_id",
            ],
        )?;
        check(
            company_id(id)
                && c["id"] == *id
                && countries.contains_key(c["listing_country"].as_str().unwrap())
                && c["branch_id"] == branch["id"]
                && c["sector_id"] == branch["sector_id"]
                && business_date(&c["context_as_of"])
                && currency(&c["report_currency"])
                && currency(&c["stock_currency"]),
            "Company listing metadata is invalid.",
        )?;
        for key in [
            "context_source_id",
            "instrument_source_id",
            "annual_source_id",
            "quarterly_source_id",
        ] {
            check(
                sources.contains(c[key].as_str().unwrap()),
                "Unknown company source.",
            )?;
        }
        texts(&c["segments"], 100)?;
        studies(&c["research"], &sources)?;
        for (kind, limit) in [("annual", 400), ("quarterly", 2000)] {
            let rows = array(&c[kind])?;
            check(rows.len() <= limit, "Too many financial periods.")?;
            let mut previous = 0;
            for row in rows {
                let year = row["year"].as_u64().unwrap_or(0);
                let period = row["period"].as_u64().unwrap_or(0);
                check(
                    (2000..=2300).contains(&year)
                        && if kind == "annual" {
                            period == 5
                        } else {
                            (1..=4).contains(&period)
                        },
                    "Invalid business financial period.",
                )?;
                check(
                    year * 5 + period > previous,
                    "Duplicate or unordered business financial periods.",
                )?;
                previous = year * 5 + period;
                check(
                    business_date(&row["start"])
                        && business_date(&row["end"])
                        && row["start"].as_str() <= row["end"].as_str()
                        && row["end"].as_str() <= raw["as_of"].as_str()
                        && row.get("report_date").is_some()
                        && (row["report_date"].is_null()
                            || business_date(&row["report_date"])
                                && row["report_date"].as_str() >= row["end"].as_str()
                                && row["report_date"].as_str() <= raw["as_of"].as_str()),
                    "Invalid business financial period dates.",
                )?;
                check(
                    currency(&row["currency"])
                        && row.get("currency_ratio").is_some()
                        && number(&row["currency_ratio"]),
                    "Invalid report currency.",
                )?;
                let values = object(&row["values"])?;
                let original = object(&row["raw"])?;
                check(
                    AMOUNTS.iter().all(|k| original.get(*k).is_some_and(number))
                        && AMOUNTS
                            .iter()
                            .chain(RATIOS.iter())
                            .all(|k| values.get(*k).is_some_and(number)),
                    "Business financial measures are incomplete or invalid.",
                )?;
                check(
                    kind == "annual" || row["values"]["return_on_capital"].is_null(),
                    "Quarterly return on capital must be unavailable.",
                )?;
                check(
                    row["currency_ratio"].as_f64().is_some_and(|r| r > 0.0)
                        || values.values().all(Value::is_null),
                    "Missing currency conversion must leave financial values unavailable.",
                )?;
            }
        }
    }
    Ok(())
}

fn period_key(row: &Value) -> String {
    format!(
        "{}|{}|{}",
        row["year"],
        row["start"].as_str().unwrap(),
        row["end"].as_str().unwrap()
    )
}
pub fn index(raw: &Value) -> Value {
    let companies = raw["companies"].as_object().unwrap();
    let groups: Vec<BTreeSet<String>> = companies
        .values()
        .map(|c| {
            c["annual"]
                .as_array()
                .unwrap()
                .iter()
                .map(period_key)
                .collect()
        })
        .collect();
    let common = groups[0]
        .iter()
        .rev()
        .find(|k| groups.iter().all(|g| g.contains(*k)));
    let mut result = raw.clone();
    result.as_object_mut().unwrap().remove("research");
    result["common_year"] = common
        .map(|k| json!(k.split('|').next().unwrap().parse::<u32>().unwrap()))
        .unwrap_or(Value::Null);
    for c in result["companies"].as_object_mut().unwrap().values_mut() {
        let annual = c["annual"].as_array().unwrap();
        let comparison = annual
            .iter()
            .find(|r| common.is_some_and(|k| period_key(r) == *k))
            .cloned()
            .unwrap_or(Value::Null);
        let latest_annual = annual.last().cloned().unwrap_or(Value::Null);
        let latest_quarter = c["quarterly"]
            .as_array()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or(Value::Null);
        c["research_count"] = json!(c["research"].as_array().unwrap().len());
        c["comparison"] = comparison;
        c["latest_annual"] = latest_annual;
        c["latest_quarter"] = latest_quarter;
        for key in ["annual", "quarterly", "research"] {
            c.as_object_mut().unwrap().remove(key);
        }
    }
    result
}
