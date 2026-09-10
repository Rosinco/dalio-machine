//! Immutable, indexed financial companion packs.
mod market;
use flate2::read::GzDecoder;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

type Result<T> = std::result::Result<T, String>;
pub const MAX_BYTES: u64 = 512 * 1024 * 1024;
const COLUMNS: [&str; 23] = [
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
    "gross_income",
    "profit_before_tax",
    "net_sales",
    "total_liabilities_and_equity",
    "cash_flow_from_investing_activities",
    "cash_flow_from_financing_activities",
    "cash_flow_for_the_year",
];
fn currency(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| s.len() == 3 && s.bytes().all(|b| b.is_ascii_uppercase()))
}
fn check(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn hash_id(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn company_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 10 && !s.starts_with('0') && s.bytes().all(|b| b.is_ascii_digit())
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn digest_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(err)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(err)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn decode(bytes: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    GzDecoder::new(bytes)
        .take(limit as u64 + 1)
        .read_to_end(&mut output)
        .map_err(err)?;
    check(
        output.len() <= limit,
        "Financial data exceeds its supported size.",
    )?;
    Ok(output)
}
fn day(s: &str) -> bool {
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
    let max = match m {
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    };
    (1800..=2300).contains(&y) && d > 0 && d <= max
}
fn integer(v: &Value) -> bool {
    v.as_u64().is_some_and(|n| n <= 10_000_000)
}
fn validate_index(v: &Value) -> Result<()> {
    check(
        v["format"] == "macro-atlas-financials"
            && (v["version"] == 1 || v["version"] == 2)
            && v["taxonomy_sha256"].as_str().is_some_and(hash_id)
            && v["as_of"].as_str().is_some_and(day),
        "Unsupported financial pack or source binding.",
    )?;
    let generated = v["generated_at"]
        .as_str()
        .ok_or("Missing financial generation date")?;
    check(
        generated.is_ascii()
            && generated.len() >= 20
            && day(&generated[..10])
            && &generated[10..11] == "T"
            && (generated.ends_with('Z') || generated.ends_with("+00:00"))
            && generated[11..13].parse::<u32>().is_ok_and(|h| h < 24)
            && &generated[13..14] == ":"
            && generated[14..16].parse::<u32>().is_ok_and(|m| m < 60)
            && &generated[16..17] == ":"
            && generated[17..19].parse::<u32>().is_ok_and(|s| s < 60),
        "Invalid financial generation date",
    )?;
    let companies = v["companies"]
        .as_object()
        .ok_or("Missing financial coverage")?;
    let columns = v["columns"].as_array().ok_or("Missing statement columns")?;
    let sources = v["sources"].as_array().ok_or("Missing financial sources")?;
    check(
        companies.len() <= 100000
            && !columns.is_empty()
            && columns.len() <= 40
            && !sources.is_empty()
            && sources.len() <= 1000,
        "Financial pack bounds are invalid.",
    )?;
    check(
        columns.iter().map(|c| c.as_str().unwrap_or("")).eq(COLUMNS),
        "Invalid financial column names.",
    )?;
    let mut annual = 0;
    let mut quarterly = 0;
    let mut withheld = 0;
    let mut with_reports = 0;
    for (id, c) in companies {
        check(
            company_id(id) && c["sha256"].as_str().is_some_and(hash_id) && integer(&c["withheld"]),
            "Invalid financial company coverage.",
        )?;
        check(
            c["withheld"].as_u64().unwrap() <= 10000
                && c["currencies"].as_array().is_some_and(|items| {
                    items.len() <= 100
                        && items.iter().all(currency)
                        && items
                            .iter()
                            .map(|v| v.as_str().unwrap())
                            .collect::<std::collections::HashSet<_>>()
                            .len()
                            == items.len()
                }),
            "Invalid reporting currencies",
        )?;
        for freq in ["annual", "quarterly"] {
            let f = &c[freq];
            for key in ["count", "gaps", "unavailable"] {
                check(integer(&f[key]), "Invalid financial coverage counts.")?;
            }
            check(
                f["count"].as_u64().unwrap() <= 2000
                    && f["gaps"].as_u64().unwrap() <= 2000
                    && f["unavailable"].as_u64() <= f["count"].as_u64(),
                "Financial coverage exceeds supported bounds.",
            )?;
            if f["count"] == 0 {
                check(
                    ["first", "last", "last_period", "end", "published"]
                        .iter()
                        .all(|k| f.get(k).is_some_and(Value::is_null))
                        && f["gaps"] == 0,
                    "Empty financial coverage has a period.",
                )?;
            } else {
                check(
                    f["first"]
                        .as_u64()
                        .is_some_and(|y| (2000..=2300).contains(&y))
                        && f["last"]
                            .as_u64()
                            .is_some_and(|y| (2000..=2300).contains(&y))
                        && f["first"].as_u64() <= f["last"].as_u64()
                        && (if freq == "annual" {
                            f["last_period"] == 5
                        } else {
                            f["last_period"]
                                .as_u64()
                                .is_some_and(|p| (1..=4).contains(&p))
                        })
                        && f["end"]
                            .as_str()
                            .is_some_and(|d| day(d) && d <= v["as_of"].as_str().unwrap())
                        && (f["published"].is_null()
                            || f["published"].as_str().is_some_and(|d| {
                                day(d)
                                    && d >= f["end"].as_str().unwrap()
                                    && d <= v["as_of"].as_str().unwrap()
                            })),
                    "Invalid financial coverage periods.",
                )?;
            }
        }
        annual += c["annual"]["count"].as_u64().unwrap();
        quarterly += c["quarterly"]["count"].as_u64().unwrap();
        withheld += c["withheld"].as_u64().unwrap();
        with_reports += u64::from(
            c["annual"]["count"].as_u64().unwrap() + c["quarterly"]["count"].as_u64().unwrap() > 0,
        );
    }
    let totals = &v["summary"];
    for key in [
        "listings",
        "with_reports",
        "annual",
        "quarterly",
        "withheld",
        "source_rows",
        "outside_directory",
        "superseded",
    ] {
        check(integer(&totals[key]), "Invalid financial summary.")?;
    }
    check(
        totals["listings"] == companies.len()
            && totals["with_reports"] == with_reports
            && totals["annual"] == annual
            && totals["quarterly"] == quarterly
            && totals["withheld"] == withheld
            && totals["source_rows"].as_u64().unwrap()
                == totals["outside_directory"].as_u64().unwrap()
                    + totals["superseded"].as_u64().unwrap()
                    + annual
                    + quarterly
                    + withheld,
        "Financial source counts do not reconcile.",
    )?;
    let mut seen = std::collections::HashSet::new();
    let mut source_totals = [0u64; 4];
    for s in sources {
        check(
            s["id"]
                .as_str()
                .is_some_and(|id| !id.is_empty() && id.len() <= 100 && seen.insert(id))
                && s["as_of"]
                    .as_str()
                    .is_some_and(|d| day(d) && d <= v["as_of"].as_str().unwrap())
                && s["sha256"].as_str().is_some_and(hash_id)
                && ["annual", "quarterly"].contains(&s["frequency"].as_str().unwrap_or("")),
            "Invalid financial source.",
        )?;
        check(
            s["path"].as_str().is_some_and(|p| {
                !p.is_empty()
                    && p.len() <= 2000
                    && !p.contains(['\\', ':'])
                    && p.split('/').all(|c| !c.is_empty() && c != "." && c != "..")
            }) && s["bytes"].as_u64().is_some_and(|b| b <= 2_000_000_000),
            "Invalid financial source file",
        )?;
        for (i, key) in ["rows", "outside_directory", "usable", "withheld"]
            .iter()
            .enumerate()
        {
            check(integer(&s[key]), "Invalid financial source counts")?;
            source_totals[i] += s[key].as_u64().unwrap();
        }
        check(
            s["outside_directory"].as_u64().unwrap()
                + s["usable"].as_u64().unwrap()
                + s["withheld"].as_u64().unwrap()
                <= s["rows"].as_u64().unwrap(),
            "Financial source counts exceed source file",
        )?;
    }
    check(
        source_totals
            == [
                totals["source_rows"].as_u64().unwrap(),
                totals["outside_directory"].as_u64().unwrap(),
                annual + quarterly,
                withheld,
            ],
        "Financial source totals do not reconcile",
    )?;
    market::validate_index(v)?;
    Ok(())
}

pub struct Pack {
    path: PathBuf,
    id: String,
    conn: Connection,
    index: Value,
    bytes: u64,
    modified: SystemTime,
}
impl Pack {
    pub fn open(path: &Path, id: &str) -> Result<Self> {
        check(hash_id(id), "Invalid financial pack ID.")?;
        let meta = fs::metadata(path).map_err(err)?;
        check(
            meta.is_file() && meta.len() <= MAX_BYTES && digest_file(path)? == id,
            "Financial pack checksum does not match its contents.",
        )?;
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(err)?;
        conn.set_limit(rusqlite::limits::Limit::SQLITE_LIMIT_LENGTH, 20_000_000)
            .map_err(err)?;
        conn.execute_batch("PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;")
            .map_err(err)?;
        let tables: i64=conn.query_row("SELECT count(*) FROM sqlite_schema WHERE type='table' AND name IN ('metadata','companies')",[],|r|r.get(0)).map_err(err)?;
        check(tables == 2, "This is not an Atlas financial pack.")?;
        let blob: Vec<u8> = conn
            .query_row("SELECT payload FROM metadata WHERE key='index'", [], |r| {
                r.get(0)
            })
            .map_err(err)?;
        let index: Value = serde_json::from_slice(&decode(&blob, 16_000_000)?).map_err(err)?;
        validate_index(&index)?;
        let count: i64 = conn
            .query_row("SELECT count(*) FROM companies", [], |r| r.get(0))
            .map_err(err)?;
        check(
            count as usize == index["companies"].as_object().unwrap().len(),
            "Financial table coverage does not reconcile.",
        )?;
        Ok(Self {
            path: path.to_owned(),
            id: id.into(),
            conn,
            index,
            bytes: meta.len(),
            modified: meta.modified().map_err(err)?,
        })
    }
    fn unchanged(&self) -> Result<()> {
        let meta = fs::metadata(&self.path).map_err(err)?;
        check(
            meta.len() == self.bytes && meta.modified().map_err(err)? == self.modified,
            "The financial pack changed while open; reopen the application.",
        )
    }
    pub fn index(&self) -> Value {
        let mut v = self.index.clone();
        v["id"] = json!(self.id);
        v["bytes"] = json!(self.bytes);
        v
    }
    pub fn company(&self, id: &str) -> Result<Value> {
        self.unchanged()?;
        check(
            company_id(id) && self.index["companies"].get(id).is_some(),
            "Unknown company in financial pack.",
        )?;
        let (blob, hash): (Vec<u8>, String) = self
            .conn
            .query_row(
                "SELECT payload,sha256 FROM companies WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(err)?
            .ok_or("Company history is missing")?;
        let bytes = decode(&blob, 2_000_000)?;
        check(
            self.index["companies"][id]["sha256"] == hash && digest(&bytes) == hash,
            "Company financial checksum failed.",
        )?;
        let v: Value = serde_json::from_slice(&bytes).map_err(err)?;
        check(v["id"] == id, "Financial listing identity mismatch.")?;
        let mut currencies = std::collections::BTreeSet::new();
        for frequency in ["annual", "quarterly"] {
            let rows = v[frequency].as_array().ok_or("Invalid financial history")?;
            check(
                rows.len() as u64
                    == self.index["companies"][id][frequency]["count"]
                        .as_u64()
                        .unwrap(),
                "Financial period counts disagree.",
            )?;
            let mut previous = 0;
            for row in rows {
                let cells = row.as_array().ok_or("Invalid financial row")?;
                check(
                    cells.len() == 8 + self.index["columns"].as_array().unwrap().len(),
                    "Wrong financial field count.",
                )?;
                let y = cells[0].as_u64().unwrap_or(0);
                let p = cells[1].as_u64().unwrap_or(0);
                check(
                    (2000..=2300).contains(&y)
                        && (if frequency == "annual" {
                            p == 5
                        } else {
                            (1..=4).contains(&p)
                        })
                        && y * 5 + p > previous,
                    "Invalid financial period order.",
                )?;
                previous = y * 5 + p;
                let source = self.index["sources"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|s| s["id"] == cells[7] && s["frequency"] == frequency)
                    .ok_or("Unknown financial source")?;
                let stamp = source["as_of"].as_str().unwrap();
                check(
                    cells[2].as_str().is_some_and(day)
                        && cells[3].as_str().is_some_and(|d| day(d) && d <= stamp)
                        && cells[2].as_str() <= cells[3].as_str()
                        && (cells[4].is_null()
                            || cells[4].as_str().is_some_and(|d| {
                                day(d) && d >= cells[3].as_str().unwrap() && d <= stamp
                            })),
                    "Invalid financial dates.",
                )?;
                check(
                    cells[5]
                        .as_str()
                        .is_some_and(|s| s.len() == 3 && s.bytes().all(|b| b.is_ascii_uppercase()))
                        && (cells[6].is_null() || cells[6].is_number())
                        && cells[8..].iter().all(|v| v.is_null() || v.is_number()),
                    "Invalid financial currency or values.",
                )?;
                currencies.insert(cells[5].as_str().unwrap());
            }
            let coverage = &self.index["companies"][id][frequency];
            let first = rows.first().unwrap_or(&Value::Null);
            let last = rows.last().unwrap_or(&Value::Null);
            let position = |r: &Value| {
                if frequency == "annual" {
                    r[0].as_u64().unwrap()
                } else {
                    r[0].as_u64().unwrap() * 4 + r[1].as_u64().unwrap() - 1
                }
            };
            let gaps = if rows.is_empty() {
                0
            } else {
                position(last) - position(first) + 1 - rows.len() as u64
            };
            let unavailable = rows
                .iter()
                .filter(|r| {
                    r[6].as_f64().is_none_or(|n| n <= 0.)
                        || r.as_array().unwrap()[8..].iter().all(Value::is_null)
                })
                .count();
            check(
                coverage["first"] == first[0]
                    && coverage["last"] == last[0]
                    && coverage["last_period"] == last[1]
                    && coverage["end"] == last[3]
                    && coverage["published"] == last[4]
                    && coverage["gaps"] == gaps
                    && coverage["unavailable"] == unavailable,
                "Financial reports disagree with coverage index",
            )?;
        }
        check(
            currencies
                == self.index["companies"][id]["currencies"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_str().unwrap())
                    .collect(),
            "Financial reporting currency coverage disagrees",
        )?;
        check(
            v["withheld"].as_array().is_some_and(|r| {
                r.len() as u64 == self.index["companies"][id]["withheld"].as_u64().unwrap()
            }),
            "Withheld report counts disagree.",
        )?;
        for r in v["withheld"].as_array().unwrap() {
            check(
                r["year"].as_i64().is_some()
                    && r["period"].as_i64().is_some()
                    && self.index["sources"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|s| s["id"] == r["source_id"])
                    && ["start", "end", "published", "reason"]
                        .iter()
                        .all(|k| r[k].as_str().is_some_and(|s| s.len() <= 2000)),
                "Invalid withheld report metadata",
            )?;
        }
        market::validate_company(&self.index, &v, id)?;
        Ok(v)
    }
    pub fn annual(&self, ids: &[String]) -> Result<Value> {
        let unique: std::collections::BTreeSet<_> = ids.iter().collect();
        check(
            !ids.is_empty() && ids.len() <= 32 && unique.len() == ids.len(),
            "Request between 1 and 32 distinct listings.",
        )?;
        let mut rows = Vec::new();
        let mut bytes = 0;
        for id in ids {
            // Verify the original complete payload before projecting annual rows.
            let mut company = self.company(id)?;
            let annual = company["annual"].take();
            let market = company["market"].take();
            bytes += annual.to_string().len() + market.to_string().len();
            check(bytes <= 8_000_000, "Annual batch exceeds 8 MB.")?;
            let mut row = json!({"id": id, "annual": annual});
            if self.index["version"] == 2 {
                row["market"] = market;
            }
            rows.push(row);
        }
        Ok(json!({"pack": self.id, "companies": rows}))
    }
    pub fn check_all(&self) -> Result<()> {
        for id in self.index["companies"].as_object().unwrap().keys() {
            self.company(id).map_err(|e| format!("Listing {id}: {e}"))?;
        }
        Ok(())
    }
}

struct Upload {
    path: PathBuf,
    expected: u64,
    written: u64,
}
pub struct Store {
    included: PathBuf,
    imported: PathBuf,
    cache: HashMap<String, Pack>,
    uploads: HashMap<String, Upload>,
}
impl Store {
    pub fn new(included: PathBuf, imported: PathBuf) -> Self {
        Self {
            included,
            imported,
            cache: HashMap::new(),
            uploads: HashMap::new(),
        }
    }
    fn candidates(&self) -> Vec<(String, PathBuf)> {
        let mut paths = HashMap::new();
        for root in [&self.included, &self.imported] {
            if let Ok(files) = fs::read_dir(root) {
                for f in files.flatten() {
                    let path = f.path();
                    if path.extension().is_some_and(|e| e == "sqlite") {
                        if let Some(id) = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .filter(|s| hash_id(s))
                        {
                            paths.insert(id.to_string(), path);
                        }
                    }
                }
            }
        }
        paths.into_iter().take(100).collect()
    }
    fn load(&mut self, id: &str) -> Result<&Pack> {
        check(hash_id(id), "Invalid financial pack ID.")?;
        if !self.cache.contains_key(id) {
            let path = self
                .candidates()
                .into_iter()
                .find(|(key, _)| key == id)
                .ok_or("Financial pack is not installed")?
                .1;
            self.cache.insert(id.into(), Pack::open(&path, id)?);
        }
        let pack = &self.cache[id];
        pack.unchanged()?;
        Ok(pack)
    }
    pub fn index(&mut self, taxonomy: &str) -> Result<Value> {
        check(hash_id(taxonomy), "Invalid directory binding.")?;
        let mut result = Value::Null;
        let mut errors = Vec::new();
        for (id, _) in self.candidates() {
            match self.load(&id) {
                Ok(pack) if pack.index["taxonomy_sha256"] == taxonomy => {
                    if result.is_null()
                        || pack.index["generated_at"].as_str() > result["generated_at"].as_str()
                    {
                        result = pack.index();
                    }
                }
                Ok(_) => (),
                Err(e) => errors.push(e),
            }
        }
        if result.is_null() && !errors.is_empty() {
            return Err(errors.join("; "));
        }
        Ok(result)
    }
    pub fn company(&mut self, pack: &str, id: &str) -> Result<Value> {
        self.load(pack)?.company(id)
    }
    pub fn annual(&mut self, pack: &str, ids: &[String]) -> Result<Value> {
        self.load(pack)?.annual(ids)
    }
    pub fn begin(&mut self, bytes: u64) -> Result<String> {
        check(
            bytes > 0 && bytes <= MAX_BYTES && self.uploads.len() < 2,
            "Financial pack must be smaller than 512 MB.",
        )?;
        fs::create_dir_all(&self.imported).map_err(err)?;
        let token = digest(format!("{:?}-{}", SystemTime::now(), std::process::id()).as_bytes());
        let path = self.imported.join(format!("upload-{token}.part"));
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(err)?;
        self.uploads.insert(
            token.clone(),
            Upload {
                path,
                expected: bytes,
                written: 0,
            },
        );
        Ok(token)
    }
    pub fn append(&mut self, token: &str, offset: u64, bytes: &[u8]) -> Result<()> {
        let u = self
            .uploads
            .get_mut(token)
            .ok_or("Unknown financial import")?;
        check(
            offset == u.written
                && !bytes.is_empty()
                && bytes.len() <= 524288
                && u.written + bytes.len() as u64 <= u.expected,
            "Invalid financial import chunk.",
        )?;
        OpenOptions::new()
            .append(true)
            .open(&u.path)
            .map_err(err)?
            .write_all(bytes)
            .map_err(err)?;
        u.written += bytes.len() as u64;
        Ok(())
    }
    pub fn cancel(&mut self, token: &str) -> Result<()> {
        if let Some(u) = self.uploads.remove(token) {
            fs::remove_file(u.path).map_err(err)?;
        }
        Ok(())
    }
    pub fn finish(&mut self, token: &str) -> Result<Value> {
        let u = self
            .uploads
            .remove(token)
            .ok_or("Unknown financial import")?;
        let result = (|| {
            check(u.written == u.expected, "Financial import is incomplete.")?;
            let id = digest_file(&u.path)?;
            let pack = Pack::open(&u.path, &id)?;
            pack.check_all()?;
            drop(pack);
            let target = self.imported.join(format!("{id}.sqlite"));
            if target.exists() {
                check(
                    digest_file(&target)? == id,
                    "Existing financial pack is damaged.",
                )?;
            } else {
                fs::hard_link(&u.path, &target).map_err(err)?;
            }
            self.cache.remove(&id);
            self.load(&id).map(Pack::index)
        })();
        let _ = fs::remove_file(u.path);
        result
    }
    pub fn export(&mut self, id: &str, folder: &Path) -> Result<PathBuf> {
        let pack = self.load(id)?;
        let target = folder.join(format!(
            "Macro-Atlas-Financials-{}-{}.sqlite",
            pack.index["as_of"].as_str().unwrap(),
            &id[..12]
        ));
        if target.exists() {
            check(
                digest_file(&target)? == id,
                "A different financial file has this name.",
            )?;
        } else {
            let mut source = File::open(&pack.path).map_err(err)?;
            let mut out = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)
                .map_err(err)?;
            std::io::copy(&mut source, &mut out).map_err(err)?;
        }
        Ok(target)
    }
}
