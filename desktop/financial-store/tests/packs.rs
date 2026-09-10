use atlas_financial_store::{digest_file, Pack, Store};
use flate2::{write::GzEncoder, Compression};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::{io::Write, path::PathBuf};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "atlas-financial-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn compressed(bytes: &[u8]) -> Vec<u8> {
    let mut w = GzEncoder::new(Vec::new(), Compression::default());
    w.write_all(bytes).unwrap();
    w.finish().unwrap()
}
fn fixture(root: &std::path::Path) -> (PathBuf, String) {
    let raw: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/financial.json")).unwrap();
    fixture_data(root, raw)
}
fn fixture_data(root: &std::path::Path, raw: Value) -> (PathBuf, String) {
    let path = root.join("build.sqlite");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch("CREATE TABLE metadata(key TEXT PRIMARY KEY,payload BLOB NOT NULL); CREATE TABLE companies(id TEXT PRIMARY KEY,payload BLOB NOT NULL,sha256 TEXT NOT NULL);").unwrap();
    let mut index = raw["index"].clone();
    for (id, payload) in raw["companies"].as_object().unwrap() {
        let bytes = payload.to_string().into_bytes();
        let hash = atlas_financial_store::digest(&bytes);
        index["companies"][id]["sha256"] = json!(hash);
        conn.execute(
            "INSERT INTO companies VALUES (?1,?2,?3)",
            rusqlite::params![id, compressed(&bytes), hash],
        )
        .unwrap();
    }
    conn.execute(
        "INSERT INTO metadata VALUES ('index',?1)",
        [compressed(index.to_string().as_bytes())],
    )
    .unwrap();
    drop(conn);
    let id = digest_file(&path).unwrap();
    let target = root.join(format!("{id}.sqlite"));
    std::fs::rename(path, &target).unwrap();
    (target, id)
}

#[test]
fn market_pack_keeps_dated_values_in_company_and_annual_reads() {
    let tmp = Temp::new();
    let raw: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/market.json")).unwrap();
    let (path, id) = fixture_data(&tmp.0, raw);
    let pack = Pack::open(&path, &id).unwrap();
    pack.check_all().unwrap();
    assert_eq!(pack.company("102").unwrap()["market"][0]["local"], 200.0);
    let batch = pack.annual(&["102".into()]).unwrap();
    assert_eq!(batch["companies"][0]["market"][0]["sek"], 2400.0);
    assert_eq!(
        batch["companies"][0]["market"][0]["price_date"],
        "2026-02-02"
    );
    assert_eq!(batch["companies"][0]["market"][0]["fx_date"], "2026-01-30");
}

#[test]
fn market_values_dates_units_and_flags_are_checked_even_with_valid_checksums() {
    for (key, value) in [
        ("local", json!(300)),
        ("sek", json!(300)),
        ("price_date", json!("2026-01-30")),
        ("fx_date", json!("2026-02-03")),
        ("fx_date", json!("2026-01-01")),
        ("source_id", json!("unknown")),
        ("flags", json!(["share_basis"])),
        ("flags", json!(["unknown"])),
        ("fx_instruments", json!(["123"])),
        ("fx_method", json!("static_fallback")),
    ] {
        let tmp = Temp::new();
        let mut raw: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/market.json")).unwrap();
        raw["companies"]["102"]["market"][0][key] = value;
        let (path, id) = fixture_data(&tmp.0, raw);
        let pack = Pack::open(&path, &id).unwrap();
        assert!(pack.check_all().is_err(), "accepted invalid {key}");
    }
}

#[test]
fn market_index_coverage_and_version_are_enforced() {
    for version in [1, 2] {
        let tmp = Temp::new();
        let mut raw: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/market.json")).unwrap();
        raw["index"]["version"] = json!(version);
        if version == 2 {
            raw["index"]["market"]["summary"]["sek"] = json!(2);
        }
        let (path, id) = fixture_data(&tmp.0, raw);
        assert!(Pack::open(&path, &id).is_err());
    }
}

#[test]
fn source_binding_and_per_company_reads_survive_reopening() {
    let tmp = Temp::new();
    let (path, id) = fixture(&tmp.0);
    let pack = Pack::open(&path, &id).unwrap();
    assert_eq!(pack.company("102").unwrap()["annual"][0][8], 200.0);
    assert!(pack.company("../102").is_err());
    drop(pack);
    let mut store = Store::new(tmp.0.clone(), tmp.0.join("imports"));
    assert!(store.index(&"b".repeat(64)).unwrap().is_null());
    assert_eq!(store.index(&"a".repeat(64)).unwrap()["id"], id);
    assert_eq!(store.company(&id, "102").unwrap()["id"], "102");
}

#[test]
fn annual_batch_is_bounded_source_checked_and_omits_quarterly_payloads() {
    let tmp = Temp::new();
    let (path, id) = fixture(&tmp.0);
    let mut store = Store::new(tmp.0.clone(), tmp.0.join("imports"));
    let rows = store.annual(&id, &["102".into()]).unwrap();
    assert_eq!(rows["pack"], id);
    assert_eq!(rows["companies"][0]["annual"][0][8], 200.0);
    assert!(rows["companies"][0].get("quarterly").is_none());
    assert!(store.annual(&id, &["102".into(), "102".into()]).is_err());
    assert!(store.annual(&id, &vec!["102".into(); 33]).is_err());
    assert!(store.annual(&id, &["999999".into()]).is_err());
    drop(store);
    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "UPDATE companies SET payload=?1 WHERE id='102'",
        [compressed(b"{}")],
    )
    .unwrap();
    drop(conn);
    let hash = digest_file(&path).unwrap();
    let pack = Pack::open(&path, &hash).unwrap();
    assert!(pack.annual(&["102".into()]).is_err());
}

#[test]
fn chunked_import_validates_before_publish_and_duplicate_is_immutable() {
    let tmp = Temp::new();
    let (path, id) = fixture(&tmp.0);
    let bytes = std::fs::read(path).unwrap();
    let mut store = Store::new(tmp.0.join("absent"), tmp.0.join("imports"));
    for _ in 0..2 {
        let token = store.begin(bytes.len() as u64).unwrap();
        assert!(store.append(&token, 1, &bytes[..8]).is_err());
        store.append(&token, 0, &bytes).unwrap();
        assert_eq!(store.finish(&token).unwrap()["id"], id);
    }
    assert_eq!(
        std::fs::read(tmp.0.join("imports").join(format!("{id}.sqlite"))).unwrap(),
        bytes
    );
    let token = store.begin(4).unwrap();
    store.append(&token, 0, b"junk").unwrap();
    assert!(store.finish(&token).is_err());
    assert!(store.company(&id, "102").is_ok());
    assert!(store.begin(600 * 1024 * 1024).is_err());
}

#[test]
fn tampered_file_and_payload_cannot_be_used() {
    let tmp = Temp::new();
    let (path, id) = fixture(&tmp.0);
    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "UPDATE companies SET payload=?1 WHERE id='102'",
        [compressed(b"{}")],
    )
    .unwrap();
    drop(conn);
    assert!(Pack::open(&path, &id).is_err());
    let different = digest_file(&path).unwrap();
    let pack = Pack::open(&path, &different).unwrap();
    assert!(pack.company("102").is_err());
    assert!(pack.check_all().is_err());
}

#[test]
fn inconsistent_coverage_is_rejected_before_import_is_published() {
    let tmp = Temp::new();
    let (path, id) = fixture(&tmp.0);
    let mut index = Pack::open(&path, &id).unwrap().index();
    index["companies"]["102"]["annual"]["gaps"] = json!(2);
    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "UPDATE metadata SET payload=?1 WHERE key='index'",
        [compressed(index.to_string().as_bytes())],
    )
    .unwrap();
    drop(conn);
    let bytes = std::fs::read(&path).unwrap();
    let mut store = Store::new(tmp.0.join("absent"), tmp.0.join("imports"));
    let token = store.begin(bytes.len() as u64).unwrap();
    store.append(&token, 0, &bytes).unwrap();
    assert!(store.finish(&token).is_err());
    assert!(store.index(&"a".repeat(64)).unwrap().is_null());
}
