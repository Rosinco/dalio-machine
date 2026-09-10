//! Validated directory metadata and reversible, source-bound classification corrections.
use super::business::company_id;
use super::{array, check, date, fields, is_id, object, timestamp, Result};
use serde_json::Value;
use std::collections::BTreeSet;

fn text(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| !s.is_empty() && s.len() <= 20000)
}
fn identifier(v: &Value) -> bool {
    v.as_str().is_some_and(company_id)
}
fn day(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.starts_with("20") && date(s))
}
fn path(v: &Value) -> bool {
    text(v)
        && v.as_str().is_some_and(|s| {
            !s.contains(['\\', ':']) && s.split('/').all(|p| !p.is_empty() && p != "." && p != "..")
        })
}
fn file(v: &Value) -> Result<()> {
    check(
        path(&v["path"])
            && v["sha256"].as_str().is_some_and(is_id)
            && v["bytes"].as_u64().is_some_and(|n| n <= 9007199254740991),
        "Invalid taxonomy source file.",
    )
}
fn dives(v: &Value) -> Result<()> {
    let rows = array(v)?;
    check(rows.len() <= 10000, "Too many taxonomy deep dives.")?;
    let mut seen = BTreeSet::new();
    for row in rows {
        check(
            path(&row["folder"]) && text(&row["label"]),
            "Invalid taxonomy deep-dive metadata.",
        )?;
        let folder = row["folder"].as_str().unwrap();
        check(seen.insert(folder), "Duplicate taxonomy deep-dive folder.")?;
        let docs = array(&row["documents"])?;
        check(
            !docs.is_empty() && docs.len() <= 1000,
            "Invalid deep-dive document inventory.",
        )?;
        let mut paths = BTreeSet::new();
        for doc in docs {
            file(doc)?;
            let p = doc["path"].as_str().unwrap();
            check(
                p.starts_with(&format!("{folder}/")) && paths.insert(p),
                "Invalid or duplicate deep-dive source path.",
            )?;
        }
    }
    Ok(())
}
pub(super) fn export_timestamp(value: &Value) -> bool {
    timestamp(value)
        && value.as_str().is_some_and(|s| {
            let local = s
                .strip_suffix('Z')
                .or_else(|| s.strip_suffix("+00:00"))
                .unwrap();
            let fraction = &local[19..];
            s.starts_with("20")
                && (fraction.is_empty()
                    || fraction.starts_with('.')
                        && fraction.len() > 1
                        && fraction[1..].bytes().all(|b| b.is_ascii_digit()))
        })
}
pub fn validate(raw: &Value, business: Option<&Value>, business_hash: Option<&str>) -> Result<()> {
    check(
        (raw["version"] == 1 || raw["version"] == 2)
            && day(&raw["as_of"])
            && export_timestamp(&raw["exported_at"]),
        "Unsupported taxonomy version or inventory date.",
    )?;
    let sectors = object(&raw["sectors"])?;
    check(
        !sectors.is_empty() && sectors.len() <= 100,
        "Invalid taxonomy sectors.",
    )?;
    for (key, s) in sectors {
        check(
            company_id(key)
                && s["id"] == *key
                && text(&s["name_sv"])
                && text(&s["name_en"])
                && s["slug"].as_str().is_some_and(|slug| {
                    !slug.is_empty()
                        && slug.split('-').all(|p| {
                            !p.is_empty()
                                && p.bytes()
                                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
                        })
                }),
            "Invalid taxonomy sector metadata.",
        )?;
    }
    let branches = object(&raw["branches"])?;
    check(
        !branches.is_empty() && branches.len() <= 1000,
        "Invalid taxonomy branches.",
    )?;
    for (key, b) in branches {
        check(
            company_id(key)
                && b["id"] == *key
                && identifier(&b["sector_id"])
                && sectors.contains_key(b["sector_id"].as_str().unwrap())
                && text(&b["name_sv"])
                && text(&b["name_en"])
                && path(&b["study_path"])
                && ["scaffold", "graduated"].contains(&b["study_status"].as_str().unwrap_or(""))
                && ["none", "graduated"].contains(&b["corpus_status"].as_str().unwrap_or("")),
            "Invalid taxonomy branch metadata.",
        )?;
        file(&b["overview"])?;
        check(
            b["overview"]["path"] == format!("{}/README.md", b["study_path"].as_str().unwrap()),
            "Invalid branch overview path.",
        )?;
        dives(&b["deep_dives"])?;
        let shared = array(&b["shared_study_ids"])?;
        check(
            shared.len() <= 1000 && shared.iter().all(text),
            "Invalid shared study identifiers.",
        )?;
    }
    let groups = object(&raw["shared_studies"])?;
    check(groups.len() <= 1000, "Too many shared studies.")?;
    for (key, g) in groups {
        check(
            !key.is_empty()
                && key.len() <= 81
                && key.as_bytes()[0].is_ascii_lowercase()
                && key
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
                && g["id"] == *key
                && path(&g["path"]),
            "Invalid shared study metadata.",
        )?;
        file(&g["overview"])?;
        check(
            g["overview"]["path"]
                .as_str()
                .unwrap()
                .starts_with(&format!("studies/sectors/{}/", g["path"].as_str().unwrap())),
            "Invalid shared study overview.",
        )?;
        let members = array(&g["branch_ids"])?;
        check(
            members.len() >= 2 && members.len() <= 1000,
            "Invalid shared study membership.",
        )?;
        let mut seen = BTreeSet::new();
        for member in members {
            check(identifier(member), "Invalid shared study branch.")?;
            let bid = member.as_str().unwrap();
            check(
                seen.insert(bid)
                    && branches.get(bid).is_some_and(|b| {
                        b["shared_study_ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|id| id == key)
                    }),
                "Invalid shared taxonomy study membership.",
            )?;
        }
        dives(&g["deep_dives"])?;
    }
    for (key, b) in branches {
        let mut seen = BTreeSet::new();
        for id in b["shared_study_ids"].as_array().unwrap() {
            let id = id.as_str().unwrap();
            check(
                seen.insert(id)
                    && groups.get(id).is_some_and(|g| {
                        g["branch_ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|bid| bid == key)
                    }),
                "Invalid shared taxonomy study membership.",
            )?;
        }
    }
    dives(&raw["unmapped_deep_dives"])?;
    let sources = array(&raw["sources"])?;
    check(sources.len() <= 1000, "Too many taxonomy sources.")?;
    for source in sources {
        file(source)?;
    }
    let notes = array(&raw["notes"])?;
    check(
        notes.len() <= 1000
            && notes.iter().all(text)
            && raw["corrections_sha256"].as_str().is_some_and(is_id),
        "Invalid taxonomy notes or correction checksum.",
    )?;
    let with_catalogue = raw["version"] == 2;
    if with_catalogue {
        super::catalogue::validate(&raw["catalogue"], business, raw["as_of"].as_str().unwrap())?;
    } else {
        check(
            raw.get("catalogue").is_none(),
            "A v1 taxonomy cannot include an unvalidated company catalogue.",
        )?;
    }
    let expected_hash = business_hash.map_or(Value::Null, Value::from);
    let expected_date = if with_catalogue {
        raw["catalogue"]["as_of"].clone()
    } else {
        business.map_or(Value::Null, |b| b["as_of"].clone())
    };
    check(
        raw.get("business_sha256") == Some(&expected_hash)
            && raw.get("classification_as_of") == Some(&expected_date),
        "Taxonomy does not match the selected business document.",
    )?;
    let classifications = object(&raw["classifications"])?;
    let companies = if with_catalogue {
        Some(object(&raw["catalogue"]["listings"])?)
    } else {
        business.and_then(|b| b["companies"].as_object())
    };
    check(
        classifications.len() <= if with_catalogue { 100000 } else { 300 }
            && classifications.len() == companies.map_or(0, |c| c.len()),
        "Taxonomy company coverage does not match the selected directory.",
    )?;
    for (key, c) in classifications {
        fields(c, &["company_id", "status"])?;
        for k in [
            "source_sector_id",
            "source_branch_id",
            "sector_id",
            "branch_id",
        ] {
            check(
                c.get(k).is_some() && (c[k].is_null() || identifier(&c[k])),
                "Invalid classification identifier.",
            )?;
        }
        check(
            company_id(key)
                && c["company_id"] == *key
                && companies.and_then(|c| c.get(key)).is_some_and(|original| {
                    c["source_sector_id"] == original["sector_id"]
                        && c["source_branch_id"] == original["branch_id"]
                }),
            "Taxonomy original classification does not match its source document.",
        )?;
        let source_branch = c["source_branch_id"].as_str();
        let source_parent = source_branch.and_then(|id| branches.get(id));
        if !with_catalogue {
            check(
                source_parent.is_some_and(|b| b["sector_id"] == c["source_sector_id"]),
                "Invalid original branch parent.",
            )?;
        }
        let mut target = source_branch.filter(|id| branches.contains_key(*id));
        let mut status = if target.is_none() {
            "unclassified"
        } else if source_parent.unwrap()["sector_id"] == c["source_sector_id"] {
            "source"
        } else {
            "sector_mismatch"
        };
        check(
            c.get("correction").is_some(),
            "Missing taxonomy correction state.",
        )?;
        let r = &c["correction"];
        if !r.is_null() {
            fields(
                r,
                &[
                    "company_id",
                    "expected_sector_id",
                    "expected_branch_id",
                    "branch_id",
                    "reason",
                    "source",
                ],
            )?;
            check(
                r["company_id"] == *key
                    && identifier(&r["expected_branch_id"])
                    && identifier(&r["expected_sector_id"])
                    && identifier(&r["branch_id"])
                    && branches.contains_key(r["branch_id"].as_str().unwrap())
                    && !r["reason"].as_str().unwrap().trim().is_empty()
                    && !r["source"].as_str().unwrap().trim().is_empty()
                    && day(&r["reviewed_at"])
                    && r["reviewed_at"].as_str() <= raw["as_of"].as_str(),
                "Invalid reviewed taxonomy correction.",
            )?;
            if !with_catalogue {
                check(
                    branches
                        .get(r["expected_branch_id"].as_str().unwrap())
                        .is_some_and(|b| b["sector_id"] == r["expected_sector_id"]),
                    "Invalid correction original branch parent.",
                )?;
            }
            if c["source_branch_id"] == r["branch_id"]
                && source_parent.is_some_and(|b| b["sector_id"] == c["source_sector_id"])
            {
                status = "aligned";
            } else if c["source_branch_id"] == r["expected_branch_id"]
                && c["source_sector_id"] == r["expected_sector_id"]
            {
                status = "corrected";
                target = r["branch_id"].as_str();
            } else {
                status = "needs_review";
            }
        }
        let expected_branch = target.map_or(Value::Null, Value::from);
        let expected_sector = target.map_or(&Value::Null, |id| &branches[id]["sector_id"]);
        check(
            c["status"] == status
                && c["branch_id"] == expected_branch
                && &c["sector_id"] == expected_sector,
            "Taxonomy effective classification disagrees with its correction.",
        )?;
    }
    Ok(())
}
