//! Validated, immutable local research files. No network or source-database access.
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

pub const MAX_BYTES: usize = 32 * 1024 * 1024;
const CATEGORIES: [&str; 5] = [
    "real_stuff",
    "production",
    "exchange",
    "promises",
    "enforcer",
];
type Result<T> = std::result::Result<T, String>;

#[derive(Deserialize)]
struct Document {
    source_file: String,
    sha256: String,
    content: String,
}
#[derive(Deserialize)]
struct Package {
    format: String,
    schema_version: u32,
    fundamentals: Document,
    liquidity: Option<Document>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Release {
    pub id: String,
    pub as_of: String,
    pub generated_at: String,
    pub fundamentals_sha256: String,
    pub liquidity_as_of: Option<String>,
    pub liquidity_sha256: Option<String>,
    pub country_count: usize,
    pub indicator_count: usize,
}
#[derive(Serialize)]
pub struct Library {
    pub releases: Vec<Release>,
    pub unreadable: usize,
}
struct Validated {
    package: Package,
    fundamentals: Value,
    liquidity: Option<Value>,
    release: Release,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn check(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn is_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn is_code(code: &str) -> bool {
    code.len() == 2 && code.bytes().all(|b| b.is_ascii_uppercase())
}
fn number(v: &Value) -> bool {
    v.is_null() || v.as_f64().is_some_and(f64::is_finite)
}
fn score(v: &Value) -> bool {
    v.is_null()
        || v.as_f64()
            .is_some_and(|n| n.is_finite() && (0.0..=100.0).contains(&n))
}
fn string(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() <= 20000)
}
fn object(v: &Value) -> Result<&Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| "Research data contains an invalid object.".into())
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array()
        .ok_or_else(|| "Research data contains an invalid list.".into())
}
fn date(s: &str) -> bool {
    if s.len() != 10 || !s.is_ascii() || &s[4..5] != "-" || &s[7..8] != "-" {
        return false;
    }
    let (Ok(y), Ok(m), Ok(d)) = (
        s[..4].parse::<u32>(),
        s[5..7].parse::<u32>(),
        s[8..].parse::<u32>(),
    ) else {
        return false;
    };
    let days = match m {
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    };
    (1800..=2300).contains(&y) && d > 0 && d <= days
}
fn timestamp(v: &Value) -> bool {
    let Some(s) = v.as_str() else {
        return false;
    };
    if !s.is_ascii()
        || s.len() < 20
        || !date(&s[..10])
        || &s[10..11] != "T"
        || &s[13..14] != ":"
        || &s[16..17] != ":"
    {
        return false;
    }
    s[11..13].parse::<u8>().is_ok_and(|n| n < 24)
        && s[14..16].parse::<u8>().is_ok_and(|n| n < 60)
        && s[17..19].parse::<u8>().is_ok_and(|n| n < 60)
        && (s.ends_with('Z') || s.ends_with("+00:00"))
}
fn fields(v: &Value, keys: &[&str]) -> Result<()> {
    for key in keys {
        check(string(&v[key]), "Research text metadata is invalid.")?;
    }
    Ok(())
}
fn measures(v: &Value, keys: &[&str]) -> Result<()> {
    for key in keys {
        check(number(&v[key]), "A research measure is invalid.")?;
    }
    Ok(())
}
fn trace(v: &Value) -> Result<()> {
    fields(v, &["availability_status"])?;
    check(
        array(&v["input_release_ids"])?
            .iter()
            .all(|r| r.as_u64().is_some()),
        "Source references are invalid.",
    )?;
    if let Some(statuses) = v.get("input_statuses") {
        check(
            array(statuses)?.iter().all(string),
            "Observation statuses are invalid.",
        )?;
    }
    check(
        v["interpretation_limit"].is_null() || string(&v["interpretation_limit"]),
        "Interpretation metadata is invalid.",
    )
}
fn document(doc: &Document) -> Result<Value> {
    check(
        !doc.source_file.is_empty()
            && doc.source_file.len() <= 160
            && !doc.source_file.contains(['/', '\\']),
        "Source names must be filenames.",
    )?;
    check(
        is_id(&doc.sha256) && digest(doc.content.as_bytes()) == doc.sha256,
        "The research file's checksum does not match its contents.",
    )?;
    serde_json::from_str(&doc.content).map_err(|_| "A source document is not valid JSON.".into())
}
fn validate_fundamentals(raw: &Value) -> Result<()> {
    check(
        raw["version"] == 1,
        "This fundamentals format requires a different Atlas version.",
    )?;
    check(
        raw["as_of"].as_str().is_some_and(date) && timestamp(&raw["generated_at"]),
        "The fundamentals release date is invalid.",
    )?;
    let categories = array(&raw["categories"])?;
    check(
        categories.len() == 5 && CATEGORIES.iter().all(|k| categories.contains(&json!(k))),
        "The five fundamentals categories are required.",
    )?;
    let indicators = array(&raw["indicators"])?;
    check(
        !indicators.is_empty() && indicators.len() <= 500,
        "The indicator catalogue is invalid.",
    )?;
    let mut seen = std::collections::HashSet::new();
    for indicator in indicators {
        let name = indicator["name"]
            .as_str()
            .ok_or("An indicator has no name.")?;
        check(
            !name.is_empty() && name.len() <= 100 && seen.insert(name),
            "Indicator names must be unique.",
        )?;
        check(
            indicator["category"]
                .as_str()
                .is_some_and(|k| CATEGORIES.contains(&k)),
            "An indicator has an unknown category.",
        )?;
        for key in ["label", "unit", "description", "uncertainty", "cadence"] {
            check(string(&indicator[key]), "Indicator metadata is incomplete.")?;
        }
        check(
            indicator["higher_is_better"].is_boolean()
                && indicator["scored"].is_boolean()
                && indicator["forward"].is_boolean(),
            "Indicator direction metadata is invalid.",
        )?;
        check(
            array(&indicator["sources"])?.iter().all(string),
            "Indicator sources are invalid.",
        )?;
    }
    let countries = object(&raw["countries"])?;
    check(
        !countries.is_empty() && countries.len() <= 300 && countries.contains_key("SE"),
        "The package must include a valid country panel with Sweden.",
    )?;
    let population = array(&raw["ranking_population"])?;
    check(
        !population.is_empty()
            && population
                .iter()
                .all(|v| v.as_str().is_some_and(|s| countries.contains_key(s))),
        "The ranking population is invalid.",
    )?;
    let unique: std::collections::HashSet<_> =
        population.iter().map(|v| v.as_str().unwrap()).collect();
    check(
        unique.len() == population.len(),
        "Ranking countries must be unique.",
    )?;
    for (code, country) in countries {
        check(
            is_code(code)
                && string(&country["name"])
                && string(&country["iso3"])
                && country["on_map"].is_boolean(),
            "Country metadata is invalid.",
        )?;
        check(
            country["currency"].is_null() || string(&country["currency"]),
            "Country currency is invalid.",
        )?;
        check(
            string(&country["data_quality"]["flag"]),
            "Country data-quality metadata is missing.",
        )?;
        check(
            country["data_quality"]["note"].is_null() || string(&country["data_quality"]["note"]),
            "Data-quality notes are invalid.",
        )?;
        if !country["cycle"].is_null() {
            fields(&country["cycle"], &["long_term_label", "short_term_label"])?;
            measures(
                &country["cycle"],
                &["long_term_confidence", "short_term_confidence"],
            )?;
        }
        for key in CATEGORIES {
            let category = &country["categories"][key];
            check(
                category.is_object() && score(&category["score"]),
                "A category score is outside 0–100.",
            )?;
            let available = category["n_available"]
                .as_u64()
                .ok_or("Category coverage is invalid.")?;
            let total = category["n_total"]
                .as_u64()
                .ok_or("Category coverage is invalid.")?;
            check(
                available <= total && total <= 500,
                "Category coverage is invalid.",
            )?;
        }
        for cell in object(&country["indicators"])?.values() {
            check(
                cell.is_object()
                    && cell.get("value").is_some()
                    && number(&cell["value"])
                    && score(&cell["pct"])
                    && cell["is_forecast"].is_boolean(),
                "An indicator observation is invalid.",
            )?;
            for key in ["source", "date", "trend", "uncertainty"] {
                check(
                    cell[key].is_null() || string(&cell[key]),
                    "Observation metadata is invalid.",
                )?;
            }
        }
        for history in object(&country["history"])?.values() {
            let points = array(history)?;
            check(points.len() <= 10000, "An annual history is too large.")?;
            let mut years = std::collections::HashSet::new();
            for point in points {
                let year = point["year"].as_i64().ok_or("A history year is invalid.")?;
                check(
                    (1800..=2300).contains(&year)
                        && years.insert(year)
                        && number(&point["value"])
                        && point["is_forecast"].is_boolean(),
                    "A history has duplicate years or invalid values.",
                )?;
            }
        }
        for pressure in array(&country["pressures"])? {
            for key in ["title", "constraint", "rule_id", "uncertainty"] {
                check(string(&pressure[key]), "Pressure metadata is invalid.")?;
            }
            check(
                array(&pressure["forced_options"])?.iter().all(string),
                "Pressure options are invalid.",
            )?;
            for spillover in array(&pressure["spillovers"])? {
                fields(spillover, &["target", "text", "channel"])?;
            }
            check(
                number(&pressure["confidence"]),
                "Pressure confidence is invalid.",
            )?;
        }
    }
    check(
        array(&raw["trade"])?.len() <= 100000,
        "The trade panel is too large.",
    )?;
    for row in array(&raw["trade"])? {
        check(
            row["iso2"].as_str().is_some_and(is_code)
                && row["partner"].as_str().is_some_and(is_code)
                && row["year"].as_u64().is_some(),
            "A trade row is invalid.",
        )?;
        for key in ["x_share", "m_share", "x_usd", "m_usd"] {
            check(number(&row[key]), "A trade value is invalid.")?;
        }
    }
    Ok(())
}
fn validate_liquidity(raw: &Value) -> Result<()> {
    check(
        raw["version"] == 1 && raw["methodology_version"] == "liquidity-diagnostics-v1",
        "This liquidity methodology requires a different Atlas version.",
    )?;
    check(
        raw["as_of"].as_str().is_some_and(date) && timestamp(&raw["as_known_at"]),
        "Liquidity release metadata is invalid.",
    )?;
    fields(raw, &["snapshot_sha256", "methodology_sha256"])?;
    check(
        raw["complete_snapshot_available_at"].is_null()
            || timestamp(&raw["complete_snapshot_available_at"]),
        "Liquidity availability date is invalid.",
    )?;
    for key in [
        "broad_money",
        "central_bank_divergence",
        "offshore_credit",
        "input_releases",
        "horizons",
        "interpretation_limits",
    ] {
        check(
            array(&raw[key])?.len() <= 2000,
            "A liquidity panel is invalid.",
        )?;
    }
    for key in [
        "money_summary",
        "mmf",
        "repo",
        "coverage",
        "formulas",
        "evidence_integrity",
    ] {
        object(&raw[key])?;
    }
    for row in array(&raw["broad_money"])?
        .iter()
        .chain(array(&raw["offshore_credit"])?)
    {
        trace(row)?;
        fields(row, &["currency", "title", "unit"])?;
        for key in ["period", "movement"] {
            check(
                row[key].is_null() || string(&row[key]),
                "Liquidity period metadata is invalid.",
            )?;
        }
        measures(
            row,
            &[
                "annual_log_growth_pct",
                "acceleration_3m_pp",
                "acceleration_1q_pp",
                "latest_value",
            ],
        )?;
        if let Some(history) = row.get("history") {
            let points = array(history)?;
            check(points.len() <= 20000, "A liquidity history is too large.")?;
            let mut periods = std::collections::HashSet::new();
            for point in points {
                let d = point["date"]
                    .as_str()
                    .ok_or("A liquidity history date is invalid.")?;
                check(
                    date(d) && periods.insert(&d[..7]) && number(&point["annual_log_growth_pct"]),
                    "A liquidity history has duplicate periods or invalid values.",
                )?;
            }
        }
    }
    for row in array(&raw["broad_money"])? {
        fields(row, &["country"])?;
    }
    for row in array(&raw["central_bank_divergence"])? {
        trace(row)?;
        fields(row, &["country", "currency"])?;
        check(
            row["period"].is_null() || string(&row["period"]),
            "Liquidity period metadata is invalid.",
        )?;
        measures(
            row,
            &[
                "money_annual_log_growth_pct",
                "central_bank_assets_annual_log_growth_pct",
                "money_minus_assets_growth_gap_pp",
            ],
        )?;
    }
    for row in array(&raw["input_releases"])? {
        check(
            row["release_id"].as_u64().is_some(),
            "Source reference is invalid.",
        )?;
        fields(
            row,
            &[
                "source_family",
                "partition_key",
                "content_sha256",
                "available_at",
            ],
        )?;
        for key in ["published_at", "source_url"] {
            check(
                row[key].is_null() || string(&row[key]),
                "Source metadata is invalid.",
            )?;
        }
    }
    for row in array(&raw["horizons"])? {
        fields(row, &["horizon", "supported_context"])?;
    }
    check(
        object(&raw["formulas"])?.values().all(string),
        "Calculation rules are invalid.",
    )?;
    for row in object(&raw["coverage"])?
        .values()
        .chain(std::iter::once(&raw["money_summary"]))
    {
        let ready = row["ready"].as_u64().ok_or("Coverage is invalid.")?;
        let expected = row["expected"].as_u64().ok_or("Coverage is invalid.")?;
        check(ready <= expected, "Coverage is invalid.")?;
    }
    let summary = &raw["money_summary"];
    fields(summary, &["interpretation_limit"])?;
    check(
        summary["common_period"].is_null() || string(&summary["common_period"]),
        "Money-summary period is invalid.",
    )?;
    measures(
        summary,
        &[
            "median_annual_log_growth_pct",
            "positive_growth_breadth",
            "accelerating_breadth",
        ],
    )?;
    let mmf = &raw["mmf"];
    trace(mmf)?;
    measures(
        mmf,
        &[
            "mmf_annual_log_growth_pct",
            "mmf_minus_m2_growth_gap_pp",
            "mmf_assets_to_m2_scale_pct",
        ],
    )?;
    check(
        mmf["period"].is_null() || string(&mmf["period"]),
        "MMF period is invalid.",
    )?;
    for row in array(&mmf["asset_allocation"])? {
        fields(row, &["title"])?;
        measures(row, &["share_of_total_pct"])?;
    }
    let ratios = &mmf["published_repo_counterparty_categories"];
    fields(ratios, &["availability_status", "interpretation_limit"])?;
    check(
        ratios["period"].is_null() || string(&ratios["period"]),
        "MMF category period is invalid.",
    )?;
    for row in array(&ratios["ratios"])? {
        fields(row, &["title"])?;
        measures(row, &["share_of_repo_pct"])?;
    }
    let repo = &raw["repo"];
    trace(repo)?;
    measures(
        repo,
        &[
            "effr_pct",
            "fragmentation_5d_median_bp",
            "fragmentation_robust_z",
        ],
    )?;
    check(
        repo["period_date"].is_null() || string(&repo["period_date"]),
        "Repo date is invalid.",
    )?;
    for row in array(&repo["venues"])? {
        fields(row, &["venue", "status"])?;
        measures(row, &["rate_pct", "effr_premium_5d_median_bp"])?;
    }
    for row in array(&repo["volume_context"])? {
        fields(row, &["title", "unit", "measure_kind"])?;
        measures(row, &["latest_value"])?;
        for key in ["latest_date", "status"] {
            check(
                row[key].is_null() || string(&row[key]),
                "Repo activity metadata is invalid.",
            )?;
        }
    }
    Ok(())
}
fn validate(text: &str) -> Result<Validated> {
    check(
        text.len() <= MAX_BYTES,
        "Research files must be smaller than 32 MB.",
    )?;
    let package: Package = serde_json::from_str(text)
        .map_err(|_| "Choose a Macro Atlas research file (.atlas.json).".to_string())?;
    check(
        package.format == "macro-atlas-research" && package.schema_version == 1,
        "This research package requires a different Atlas version.",
    )?;
    let fundamentals = document(&package.fundamentals)?;
    validate_fundamentals(&fundamentals)?;
    let liquidity = package.liquidity.as_ref().map(document).transpose()?;
    if let Some(raw) = &liquidity {
        validate_liquidity(raw)?;
    }
    let liquidity_hash = package.liquidity.as_ref().map(|d| d.sha256.clone());
    let release = Release {
        id: digest(
            format!(
                "macro-atlas-research-v1\n{}\n{}",
                package.fundamentals.sha256,
                liquidity_hash.as_deref().unwrap_or("")
            )
            .as_bytes(),
        ),
        as_of: fundamentals["as_of"].as_str().unwrap().into(),
        generated_at: fundamentals["generated_at"].as_str().unwrap().into(),
        fundamentals_sha256: package.fundamentals.sha256.clone(),
        liquidity_as_of: liquidity
            .as_ref()
            .map(|v| v["as_of"].as_str().unwrap().into()),
        liquidity_sha256: liquidity_hash,
        country_count: fundamentals["countries"].as_object().unwrap().len(),
        indicator_count: fundamentals["indicators"].as_array().unwrap().len(),
    };
    Ok(Validated {
        package,
        fundamentals,
        liquidity,
        release,
    })
}
pub fn inspect(text: &str) -> Result<Release> {
    Ok(validate(text)?.release)
}

pub struct Archive {
    root: PathBuf,
}
impl Archive {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        check(is_id(id), "Invalid research release identifier.")?;
        Ok(self.root.join(format!("{id}.atlas.json")))
    }
    pub fn read(&self, id: &str) -> Result<String> {
        let path = self.path(id)?;
        check(
            fs::metadata(&path)
                .map_err(|_| "Saved research could not be opened.")?
                .len()
                <= MAX_BYTES as u64,
            "Saved research exceeds the file limit.",
        )?;
        let text = fs::read_to_string(path).map_err(|_| "Saved research could not be opened.")?;
        check(
            inspect(&text)?.id == id,
            "Saved research does not match its identifier.",
        )?;
        Ok(text)
    }
    pub fn import(&self, text: &str) -> Result<Release> {
        let release = inspect(text)?;
        let destination = self.path(&release.id)?;
        fs::create_dir_all(&self.root)
            .map_err(|e| format!("Could not create the research library: {e}"))?;
        if destination.exists() {
            return inspect(&self.read(&release.id)?);
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let temporary = self.root.join(format!(
            ".import-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| -> Result<()> {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|e| e.to_string())?;
            file.write_all(text.as_bytes())
                .and_then(|_| file.sync_all())
                .map_err(|e| e.to_string())?;
            // Publish a complete file without replacing any earlier archive entry.
            match fs::hard_link(&temporary, &destination) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    self.read(&release.id)?;
                    Ok(())
                }
                Err(e) => Err(format!("Could not save the research file: {e}")),
            }
        })();
        let _ = fs::remove_file(&temporary);
        result?;
        Ok(release)
    }
    pub fn list(&self) -> Result<Library> {
        if !self.root.exists() {
            return Ok(Library {
                releases: vec![],
                unreadable: 0,
            });
        }
        let mut library = Library {
            releases: vec![],
            unreadable: 0,
        };
        for entry in fs::read_dir(&self.root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".atlas.json") {
                match self.read(id).and_then(|text| inspect(&text)) {
                    Ok(release) => library.releases.push(release),
                    Err(_) => library.unreadable += 1,
                }
            }
        }
        library
            .releases
            .sort_by(|a, b| b.generated_at.cmp(&a.generated_at).then(b.id.cmp(&a.id)));
        Ok(library)
    }
    pub fn resource(&self, id: &str, resource: &str) -> Result<Value> {
        let validated = validate(&self.read(id)?)?;
        let raw = validated.fundamentals;
        match resource {
            "index" => {
                let mut index = raw.clone();
                for country in index["countries"].as_object_mut().unwrap().values_mut() {
                    country.as_object_mut().unwrap().remove("history");
                }
                index["manifest"] = json!({"sha256":validated.release.fundamentals_sha256,"source_file":validated.package.fundamentals.source_file,
                    "source_bytes":validated.package.fundamentals.content.len(),"country_files":{}});
                Ok(index)
            }
            "history" => Ok(Value::Object(
                raw["countries"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v["history"].clone()))
                    .collect(),
            )),
            "liquidity" => Ok(validated.liquidity.unwrap_or(Value::Null)),
            _ if resource.starts_with("country:") => {
                let code = &resource[8..];
                check(is_code(code), "Invalid country identifier.")?;
                raw["countries"]
                    .get(code)
                    .cloned()
                    .ok_or_else(|| "This release does not include that country.".into())
            }
            _ => Err("Unknown research resource.".into()),
        }
    }
}
